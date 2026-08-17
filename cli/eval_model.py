#!/usr/bin/env python3
"""
CLI script for evaluating ASR models.

Usage:
    python cli/evaluate.py --data_dir data/raw --model_dir models/whisper/final --split test
"""

import sys
import logging
import argparse
import json
import hashlib
from pathlib import Path
from typing import Any, Dict

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    setup_logging,
    get_device,
    create_dataloaders,
    get_model_registry,
    load_config,
    WhisperSpeechDataset,
    WhisperDataCollator,
)
from src.utils.metrics import evaluate_predictions
from src.utils.text_preprocessing import SerbianTextPreprocessor

logger = logging.getLogger(__name__)


def _canonical_sample_id(
    sample_id: Any,
    audio_path: Any,
    reference_raw: str,
    row_index: int,
) -> str:
    """Return a stable, non-empty sample identifier for row-level joins."""
    sid = str(sample_id or "").strip()
    if sid:
        return sid

    path_value = str(audio_path or "").strip()
    if path_value:
        normalized = Path(path_value).as_posix().lower()
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
        return f"pathsha1_{digest}"

    fallback = f"{reference_raw.strip()}|{row_index}"
    digest = hashlib.sha1(fallback.encode("utf-8")).hexdigest()[:16]
    return f"rowsha1_{digest}"


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


def main():
    """Main evaluation function."""
    generation_defaults = _load_generation_defaults()

    parser = argparse.ArgumentParser(description="Evaluate ASR model")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Data directory")
    parser.add_argument("--model_dir", type=str, default="models/whisper/final", help="Model directory")
    parser.add_argument("--model_type", type=str, default="whisper", help="Model type from registry")
    parser.add_argument("--split", choices=["val", "test", "all"], default="val", help="Split to evaluate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_beams", type=int, default=int(generation_defaults["num_beams"]), help="Beam search width")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=int(generation_defaults["no_repeat_ngram_size"]), help="No-repeat ngram size")
    parser.add_argument("--repetition_penalty", type=float, default=float(generation_defaults["repetition_penalty"]), help="Repetition penalty")
    parser.add_argument("--length_penalty", type=float, default=float(generation_defaults["length_penalty"]), help="Length penalty")
    parser.add_argument("--temperature", type=float, default=float(generation_defaults["temperature"]), help="Decoding temperature")
    parser.add_argument("--max_new_tokens", type=int, default=int(generation_defaults["max_new_tokens"]), help="Max new tokens")
    parser.add_argument("--max_length", type=int, default=int(generation_defaults["max_length"]), help="Max generated length")
    parser.add_argument("--metrics_out", type=str, default="", help="Optional path to save metrics JSON")
    parser.add_argument(
        "--predictions_out",
        type=str,
        default="",
        help="Optional path to save predictions as JSONL with reference/prediction pairs",
    )
    parser.add_argument(
        "--allow_auto_populate",
        action="store_true",
        help="Allow fallback chunk transcript auto-populate when text dir is empty",
    )
    
    args = parser.parse_args()
    
    setup_logging('logs')
    device = get_device()
    
    logger.info("=" * 80)
    logger.info("SERBIAN ASR - EVALUATION")
    logger.info("=" * 80)
    
    # Load model
    logger.info(f'Loading model from {args.model_dir} for evaluation')
    registry = get_model_registry()
    adapter = registry.create(
        args.model_type,
        model_name_or_path=args.model_dir,
        language='sr',
        freeze_encoder=False,
    )
    processor = adapter.get_processor()
    model = adapter.unwrap()
    adapter.to(device)
    adapter.eval()

    # Match validated decoding behavior from historical best setup.
    # Prevent model-level suppression from conflicting with anti-repetition decoding.
    try:
        if hasattr(model, 'generation_config'):
            model.generation_config.forced_decoder_ids = None
            model.generation_config.suppress_tokens = None
            model.generation_config.begin_suppress_tokens = None
    except Exception:
        pass
    try:
        if hasattr(model.config, 'forced_decoder_ids'):
            model.config.forced_decoder_ids = None
    except Exception:
        pass
    
    # Load data
    if args.split == "all":
        from pathlib import Path as _Path
        from torch.utils.data import DataLoader as _DataLoader

        base = _Path(args.data_dir)
        audio_dir = base / "audio" if (base / "audio").exists() else base
        text_dir = base / "text" if (base / "text").exists() else base

        dataset = WhisperSpeechDataset(
            audio_dir=str(audio_dir),
            text_dir=str(text_dir),
            processor=processor,
            augment=False,
        )
        loader = _DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=WhisperDataCollator(processor=processor, include_text=True),
        )
    else:
        _, val_loader, test_loader = create_dataloaders(
            args.data_dir,
            processor,
            batch_size=args.batch_size,
            allow_auto_populate=args.allow_auto_populate,
        )
        loader = val_loader if args.split == "val" else test_loader
    
    preproc = SerbianTextPreprocessor()
    preds_raw = []
    refs_raw = []
    prediction_rows = []
    
    logger.info(f'Running evaluation on {args.split} set')
    for batch in loader:
        input_feats = batch['input_features'].to(device)
        attn = batch.get('attention_mask')
        if attn is not None:
            attn = attn.to(device)
        with torch.no_grad():
            generate_kwargs = {
                "input_features": input_feats,
                "attention_mask": attn,
                "language": 'sr',
                "task": 'transcribe',
                "num_beams": args.num_beams,
                "repetition_penalty": args.repetition_penalty,
                "length_penalty": args.length_penalty,
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "max_length": args.max_length,
                "suppress_tokens": None,
                "begin_suppress_tokens": None,
            }
            if args.no_repeat_ngram_size > 0:
                generate_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
            try:
                generated = model.generate(**generate_kwargs)
            except TypeError:
                try:
                    forced = processor.get_decoder_prompt_ids(language='sr', task='transcribe')
                    generate_kwargs.pop("language", None)
                    generate_kwargs.pop("task", None)
                    generate_kwargs["forced_decoder_ids"] = forced
                    generated = model.generate(**generate_kwargs)
                except Exception:
                    generated = model.generate(input_feats, attention_mask=attn)
            except RuntimeError as runtime_error:
                error_text = str(runtime_error).lower()
                if 'bad allocation' in error_text or 'out of memory' in error_text:
                    logger.warning(
                        'Generation OOM/bad allocation detected; retrying with safe decode settings (num_beams=1).'
                    )
                    safe_kwargs = dict(generate_kwargs)
                    safe_kwargs['num_beams'] = 1
                    safe_kwargs.pop('no_repeat_ngram_size', None)
                    safe_kwargs['repetition_penalty'] = 1.0
                    safe_kwargs['length_penalty'] = 1.0
                    safe_kwargs['temperature'] = 0.0
                    generated = model.generate(**safe_kwargs)
                else:
                    raise
        decoded = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
        sample_ids = batch.get('sample_id')
        audio_paths = batch.get('audio_path')
        for i, text in enumerate(decoded):
            reference_raw = batch['text'][i]
            prediction_raw = text
            sample_id = sample_ids[i] if isinstance(sample_ids, list) and i < len(sample_ids) else ""
            audio_path = audio_paths[i] if isinstance(audio_paths, list) and i < len(audio_paths) else ""
            canonical_sample_id = _canonical_sample_id(
                sample_id=sample_id,
                audio_path=audio_path,
                reference_raw=reference_raw,
                row_index=len(prediction_rows),
            )

            preds_raw.append(prediction_raw)
            refs_raw.append(reference_raw)

            prediction_rows.append(
                {
                    "id": len(prediction_rows),
                    "sample_id": canonical_sample_id,
                    "audio_path": audio_path,
                    "prediction_raw": prediction_raw,
                    "reference_raw": reference_raw,
                }
            )
    
    metric_result = evaluate_predictions(predictions=preds_raw, references=refs_raw, preprocessor=preproc)
    preds_preprocessed = metric_result["predictions_preprocessed"]
    refs_preprocessed = metric_result["references_preprocessed"]
    wer_value = float(metric_result["wer"])
    cer_value = float(metric_result["cer"])

    for row, pred_proc, ref_proc in zip(prediction_rows, preds_preprocessed, refs_preprocessed):
        row["prediction_preprocessed"] = pred_proc
        row["reference_preprocessed"] = ref_proc
        # Backward-compatible keys
        row["prediction"] = pred_proc
        row["reference"] = ref_proc

    logger.info(f'Computed {len(preds_preprocessed)} predictions')

    logger.info('WER: %.4f', wer_value)
    logger.info('CER: %.4f', cer_value)

    metrics_payload = {
        "model_dir": args.model_dir,
        "data_dir": args.data_dir,
        "split": args.split,
        "num_samples": len(preds_preprocessed),
        "decode": {
            "num_beams": args.num_beams,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "repetition_penalty": args.repetition_penalty,
            "length_penalty": args.length_penalty,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "max_length": args.max_length,
        },
        "wer": wer_value,
        "cer": cer_value,
        "normalization": "SerbianTextPreprocessor.preprocess",
    }

    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding='utf-8')
        logger.info(f"Metrics saved to: {metrics_path}")

    if args.predictions_out:
        preds_path = Path(args.predictions_out)
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        with preds_path.open('w', encoding='utf-8') as f:
            for row in prediction_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(f"Predictions saved to: {preds_path}")


if __name__ == '__main__':
    main()
