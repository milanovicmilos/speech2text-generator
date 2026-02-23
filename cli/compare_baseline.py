#!/usr/bin/env python3
"""Compare baseline Whisper model against a fine-tuned Whisper checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import torch
from evaluate import load
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import create_dataloaders, get_device, setup_logging  # noqa: E402
from src.utils.text_preprocessing import SerbianTextPreprocessor  # noqa: E402

logger = logging.getLogger(__name__)


def evaluate_model(
    model_id_or_path: str,
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
    processor = WhisperProcessor.from_pretrained(model_id_or_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_id_or_path)
    model.to(device)
    model.eval()

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
        input_feats = batch["input_features"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

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

        with torch.no_grad():
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
    parser = argparse.ArgumentParser(description="Compare baseline and fine-tuned Whisper models")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Dataset directory")
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
    parser.add_argument("--num_beams", type=int, default=8, help="Beam search width")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=4, help="No-repeat ngram size")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Length penalty")
    parser.add_argument("--temperature", type=float, default=0.0, help="Decoding temperature")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens")
    parser.add_argument("--max_length", type=int, default=256, help="Max output length")
    parser.add_argument("--out_json", type=str, default="", help="Optional JSON output path")

    args = parser.parse_args()
    setup_logging("logs")

    baseline = evaluate_model(
        model_id_or_path=args.baseline_model,
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
