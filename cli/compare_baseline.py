#!/usr/bin/env python3
"""Compare baseline model against a fine-tuned checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from evaluate import load

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import create_dataloaders, get_device, get_model_registry, setup_logging, load_config  # noqa: E402
from src.utils.text_preprocessing import SerbianTextPreprocessor  # noqa: E402

logger = logging.getLogger(__name__)


def _load_generation_defaults(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load generation defaults from config with safe fallback values."""
    fallback = {
        "num_beams": 8,
        "no_repeat_ngram_size": 10,
        "repetition_penalty": 5.0,
        "length_penalty": 1.0,
        "temperature": 0.0,
        "max_new_tokens": 128,
        "max_length": 256,
    }
    try:
        config = load_config(config_path)
    except Exception:
        return fallback

    generation = config.get("generation", {}) if isinstance(config, dict) else {}
    if not isinstance(generation, dict):
        return fallback

    defaults = dict(fallback)
    for key in fallback:
        if key in generation:
            defaults[key] = generation[key]
    return defaults


def evaluate_model(
    model_id_or_path: str,
    model_type: str,
    data_dir: str,
    split: str,
    batch_size: int,
    num_beams: int,
    no_repeat_ngram_size: int,
    repetition_penalty: float,
    length_penalty: float,
    temperature: float,
    max_new_tokens: int,
    max_length: int,
) -> Dict[str, float | int | str]:
    device = get_device()

    logger.info("Loading model for evaluation: %s", model_id_or_path)
    registry = get_model_registry()
    adapter = registry.create(
        model_type,
        model_name_or_path=model_id_or_path,
        language='sr',
        freeze_encoder=False,
    )
    processor = adapter.get_processor()
    model = adapter.unwrap()
    adapter.to(device)
    adapter.eval()

    try:
        if hasattr(model, "generation_config"):
            model.generation_config.forced_decoder_ids = None
            model.generation_config.suppress_tokens = None
            model.generation_config.begin_suppress_tokens = None
    except Exception:
        pass
    try:
        if hasattr(model.config, "forced_decoder_ids"):
            model.config.forced_decoder_ids = None
    except Exception:
        pass

    _, val_loader, test_loader = create_dataloaders(data_dir, processor, batch_size=batch_size)
    loader = val_loader if split == "val" else test_loader

    preproc = SerbianTextPreprocessor()
    wer_metric = load("wer")
    cer_metric = load("cer")

    predictions: List[str] = []
    references: List[str] = []

    for batch in loader:
        input_key = "input_features" if "input_features" in batch and batch["input_features"] is not None else "input_values"
        input_feats = batch[input_key].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            if model_type.lower() == "wav2vec2":
                logits = model(input_values=input_feats, attention_mask=attention_mask).logits
                generated = torch.argmax(logits, dim=-1)
                decoded = processor.batch_decode(generated)
            else:
                generate_kwargs = {
                    "input_features": input_feats,
                    "attention_mask": attention_mask,
                    "language": "sr",
                    "task": "transcribe",
                    "num_beams": num_beams,
                    "repetition_penalty": repetition_penalty,
                    "length_penalty": length_penalty,
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                    "max_length": max_length,
                    "suppress_tokens": None,
                    "begin_suppress_tokens": None,
                }
                if no_repeat_ngram_size > 0:
                    generate_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

                try:
                    generated = model.generate(**generate_kwargs)
                except TypeError:
                    forced = processor.get_decoder_prompt_ids(language="sr", task="transcribe")
                    generate_kwargs.pop("language", None)
                    generate_kwargs.pop("task", None)
                    generate_kwargs["forced_decoder_ids"] = forced
                    generated = model.generate(**generate_kwargs)

                decoded = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
        for index, text in enumerate(decoded):
            predictions.append(preproc.preprocess(text))
            references.append(preproc.preprocess(batch["text"][index]))

    wer_score = float(wer_metric.compute(predictions=predictions, references=references))
    cer_score = float(cer_metric.compute(predictions=predictions, references=references))

    return {
        "model": model_id_or_path,
        "split": split,
        "num_samples": len(predictions),
        "wer": wer_score,
        "cer": cer_score,
    }


def main() -> None:
    generation_defaults = _load_generation_defaults()

    parser = argparse.ArgumentParser(description="Compare baseline and fine-tuned ASR models")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Dataset directory")
    parser.add_argument("--model_type", type=str, default="whisper", help="Model type from registry")
    parser.add_argument(
        "--baseline_model",
        type=str,
        default="openai/whisper-base",
        help="HuggingFace model id for baseline",
    )
    parser.add_argument(
        "--finetuned_model",
        type=str,
        default="models/whisper/final_old",
        help="Path to fine-tuned model",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test", help="Evaluation split")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_beams", type=int, default=int(generation_defaults["num_beams"]), help="Beam search width")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=int(generation_defaults["no_repeat_ngram_size"]), help="No-repeat ngram size")
    parser.add_argument("--repetition_penalty", type=float, default=float(generation_defaults["repetition_penalty"]), help="Repetition penalty")
    parser.add_argument("--length_penalty", type=float, default=float(generation_defaults["length_penalty"]), help="Length penalty")
    parser.add_argument("--temperature", type=float, default=float(generation_defaults["temperature"]), help="Decoding temperature")
    parser.add_argument("--max_new_tokens", type=int, default=int(generation_defaults["max_new_tokens"]), help="Max new tokens")
    parser.add_argument("--max_length", type=int, default=int(generation_defaults["max_length"]), help="Max output length")
    parser.add_argument("--out_json", type=str, default="", help="Optional JSON output path")

    args = parser.parse_args()
    setup_logging("logs")

    baseline = evaluate_model(
        model_id_or_path=args.baseline_model,
        model_type=args.model_type,
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
        length_penalty=args.length_penalty,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
    )

    finetuned = evaluate_model(
        model_id_or_path=args.finetuned_model,
        model_type=args.model_type,
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
        length_penalty=args.length_penalty,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
    )

    result = {
        "baseline": baseline,
        "finetuned": finetuned,
        "delta_wer": float(baseline["wer"]) - float(finetuned["wer"]),
        "delta_cer": float(baseline["cer"]) - float(finetuned["cer"]),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
