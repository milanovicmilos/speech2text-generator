#!/usr/bin/env python3
"""End-to-end Kaggle runner for full alignment + train + strict holdout evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def run_cmd(command: List[str], cwd: Path) -> None:
    print("[RUN]", " ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_holdout_lists(
    args: argparse.Namespace,
    holdout_dir: Path,
    repo_root: Path,
) -> Tuple[Path, Path, Dict[str, str]]:
    """Resolve fixed holdout list paths and metadata for reproducible runs."""
    if args.fixed_train_val_list_json and args.fixed_holdout_list_json:
        train_val_list = Path(args.fixed_train_val_list_json)
        holdout_list = Path(args.fixed_holdout_list_json)
        if not train_val_list.exists() or not holdout_list.exists():
            raise FileNotFoundError(
                "Both --fixed_train_val_list_json and --fixed_holdout_list_json must exist when provided."
            )
        return train_val_list, holdout_list, {
            "mode": "external_fixed",
            "train_val_list": str(train_val_list),
            "holdout_list": str(holdout_list),
        }

    train_val_list = holdout_dir / "train_val_list.json"
    holdout_list = holdout_dir / "holdout_list.json"

    if args.freeze_holdout and train_val_list.exists() and holdout_list.exists():
        return train_val_list, holdout_list, {
            "mode": "reused_local",
            "train_val_list": str(train_val_list),
            "holdout_list": str(holdout_list),
        }

    run_cmd(
        [
            sys.executable,
            "tools/create_holdout.py",
            "--raw_dir",
            args.raw_dir,
            "--out_dir",
            str(holdout_dir),
            "--holdout_ratio",
            str(args.holdout_ratio),
            "--seed",
            str(args.seed),
        ],
        cwd=repo_root,
    )
    return train_val_list, holdout_list, {
        "mode": "generated_new",
        "train_val_list": str(train_val_list),
        "holdout_list": str(holdout_list),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full ASR pipeline on Kaggle")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Raw dataset directory")
    parser.add_argument("--work_dir", type=str, default="/kaggle/working/asr_full_run", help="Output working directory")
    parser.add_argument("--model_name", type=str, default="openai/whisper-base", help="Base model")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-6, help="Training learning rate")
    parser.add_argument("--warmup_steps", type=int, default=600, help="Warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation")
    parser.add_argument("--holdout_ratio", type=float, default=0.15, help="Strict holdout ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--freeze_holdout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing holdout lists from work_dir/holdout when present",
    )
    parser.add_argument(
        "--fixed_train_val_list_json",
        type=str,
        default="",
        help="Optional explicit path to train_val_list.json to enforce fixed split",
    )
    parser.add_argument(
        "--fixed_holdout_list_json",
        type=str,
        default="",
        help="Optional explicit path to holdout_list.json to enforce fixed split",
    )
    parser.add_argument("--alignment_model_size", type=str, default="small", help="faster-whisper model size")
    parser.add_argument("--target_chunk_seconds", type=float, default=24.0, help="Aligned chunk target duration")
    parser.add_argument("--min_chunk_seconds", type=float, default=3.0, help="Aligned chunk min duration")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate")
    parser.add_argument("--qa_unmatched_threshold", type=float, default=0.35, help="QA threshold for unmatched word ratio")
    parser.add_argument("--qa_max_chunk_seconds", type=float, default=30.0, help="QA threshold for chunk duration")
    parser.add_argument("--min_aligned_word_ratio", type=float, default=0.55, help="Minimum aligned word ratio for chunk acceptance")
    parser.add_argument("--max_chars_per_second", type=float, default=20.0, help="Maximum allowed transcript chars/sec")
    parser.add_argument(
        "--drop_suspicious",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop suspicious chunks during aligned chunk build",
    )
    parser.add_argument("--num_beams", type=int, default=8, help="Eval decoding beams")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=10, help="Eval no repeat ngram")
    parser.add_argument("--repetition_penalty", type=float, default=5.0, help="Eval repetition penalty")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Eval length penalty")
    parser.add_argument("--temperature", type=float, default=0.0, help="Eval temperature")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Eval max new tokens")
    parser.add_argument("--max_length", type=int, default=256, help="Eval max length")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

    repo_root = Path(__file__).resolve().parents[1]
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    holdout_dir = work_dir / "holdout"
    aligned_train_dir = work_dir / "aligned_train"
    aligned_holdout_dir = work_dir / "aligned_holdout"
    model_out_dir = work_dir / "models" / "raw_aligned_full"
    metrics_dir = work_dir / "metrics"
    logs_dir = work_dir / "logs"

    for path in [holdout_dir, aligned_train_dir, aligned_holdout_dir, model_out_dir, metrics_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    has_cuda = torch.cuda.is_available()
    align_device = "cuda" if has_cuda else "cpu"
    align_compute_type = "float16" if has_cuda else "int8"

    print(f"[INFO] CUDA available: {has_cuda}")
    if has_cuda:
        print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")

    train_val_list_path, holdout_list_path, holdout_lock = _resolve_holdout_lists(
        args=args,
        holdout_dir=holdout_dir,
        repo_root=repo_root,
    )

    run_cmd(
        [
            sys.executable,
            "cli/build_aligned_chunks.py",
            "--raw_dir",
            args.raw_dir,
            "--out_dir",
            str(aligned_train_dir),
            "--model_size",
            args.alignment_model_size,
            "--device",
            align_device,
            "--compute_type",
            align_compute_type,
            "--target_chunk_seconds",
            str(args.target_chunk_seconds),
            "--min_chunk_seconds",
            str(args.min_chunk_seconds),
            "--sample_rate",
            str(args.sample_rate),
            "--qa_unmatched_threshold",
            str(args.qa_unmatched_threshold),
            "--qa_max_chunk_seconds",
            str(args.qa_max_chunk_seconds),
            "--min_aligned_word_ratio",
            str(args.min_aligned_word_ratio),
            "--max_chars_per_second",
            str(args.max_chars_per_second),
            "--include_list_json",
            str(train_val_list_path),
            *( ["--drop_suspicious"] if args.drop_suspicious else [] ),
        ],
        cwd=repo_root,
    )

    run_cmd(
        [
            sys.executable,
            "cli/build_aligned_chunks.py",
            "--raw_dir",
            args.raw_dir,
            "--out_dir",
            str(aligned_holdout_dir),
            "--model_size",
            args.alignment_model_size,
            "--device",
            align_device,
            "--compute_type",
            align_compute_type,
            "--target_chunk_seconds",
            str(args.target_chunk_seconds),
            "--min_chunk_seconds",
            str(args.min_chunk_seconds),
            "--sample_rate",
            str(args.sample_rate),
            "--qa_unmatched_threshold",
            str(args.qa_unmatched_threshold),
            "--qa_max_chunk_seconds",
            str(args.qa_max_chunk_seconds),
            "--min_aligned_word_ratio",
            str(args.min_aligned_word_ratio),
            "--max_chars_per_second",
            str(args.max_chars_per_second),
            "--include_list_json",
            str(holdout_list_path),
            *( ["--drop_suspicious"] if args.drop_suspicious else [] ),
        ],
        cwd=repo_root,
    )

    run_cmd(
        [
            sys.executable,
            "tools/check_holdout_conflicts.py",
            "--holdout_list",
            str(holdout_list_path),
            "--manifest",
            str(aligned_train_dir / "manifest.json"),
            "--out",
            str(metrics_dir / "holdout_conflicts.json"),
        ],
        cwd=repo_root,
    )

    train_command = [
        sys.executable,
        "cli/train.py",
        "--data_dir",
        str(aligned_train_dir),
        "--output_dir",
        str(model_out_dir),
        "--model_name",
        args.model_name,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--warmup_steps",
        str(args.warmup_steps),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--seed",
        str(args.seed),
        "--num_workers",
        "2" if has_cuda else "0",
    ]
    if has_cuda:
        train_command.append("--fp16")

    run_cmd(train_command, cwd=repo_root)

    eval_common = [
        "--data_dir",
        str(aligned_holdout_dir),
        "--split",
        "all",
        "--batch_size",
        "8",
        "--num_beams",
        str(args.num_beams),
        "--no_repeat_ngram_size",
        str(args.no_repeat_ngram_size),
        "--repetition_penalty",
        str(args.repetition_penalty),
        "--length_penalty",
        str(args.length_penalty),
        "--temperature",
        str(args.temperature),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--max_length",
        str(args.max_length),
    ]

    finetuned_metrics = metrics_dir / "finetuned_holdout_metrics.json"
    baseline_metrics = metrics_dir / "baseline_holdout_metrics.json"

    run_cmd(
        [
            sys.executable,
            "cli/eval_model.py",
            "--model_dir",
            str(model_out_dir / "final"),
            *eval_common,
            "--metrics_out",
            str(finetuned_metrics),
            "--predictions_out",
            str(metrics_dir / "finetuned_holdout_predictions.jsonl"),
        ],
        cwd=repo_root,
    )

    run_cmd(
        [
            sys.executable,
            "cli/eval_model.py",
            "--model_dir",
            args.model_name,
            *eval_common,
            "--metrics_out",
            str(baseline_metrics),
            "--predictions_out",
            str(metrics_dir / "baseline_holdout_predictions.jsonl"),
        ],
        cwd=repo_root,
    )

    finetuned_payload = read_json(finetuned_metrics)
    baseline_payload = read_json(baseline_metrics)

    summary = {
        "environment": {
            "cuda": has_cuda,
            "device_name": torch.cuda.get_device_name(0) if has_cuda else "cpu",
        },
        "paths": {
            "work_dir": str(work_dir),
            "holdout_dir": str(holdout_dir),
            "aligned_train_dir": str(aligned_train_dir),
            "aligned_holdout_dir": str(aligned_holdout_dir),
            "model_out_dir": str(model_out_dir),
            "metrics_dir": str(metrics_dir),
        },
        "holdout_lock": holdout_lock,
        "baseline": baseline_payload,
        "finetuned": finetuned_payload,
        "delta": {
            "wer": float(baseline_payload["wer"]) - float(finetuned_payload["wer"]),
            "cer": float(baseline_payload["cer"]) - float(finetuned_payload["cer"]),
        },
    }

    summary_path = metrics_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[DONE] Summary written:", summary_path)
    print(json.dumps(summary["delta"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
