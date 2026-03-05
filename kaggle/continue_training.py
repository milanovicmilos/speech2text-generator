#!/usr/bin/env python3
"""Continue ASR training on Kaggle from a saved model or checkpoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(command: list[str], cwd: Path) -> None:
    print("[RUN]", " ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue training from saved model/checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Aligned training data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="New output directory for continued run")
    parser.add_argument(
        "--base_model_dir",
        type=str,
        required=True,
        help="Path to previously trained final model directory (e.g., .../final)",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="",
        help="Optional checkpoint path for true optimizer-state resume",
    )
    parser.add_argument("--epochs", type=int, default=4, help="Extra epochs to train")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-6, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=300, help="Warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    command = [
        sys.executable,
        "cli/train.py",
        "--data_dir",
        args.data_dir,
        "--output_dir",
        args.output_dir,
        "--model_name",
        args.base_model_dir,
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
        "2",
        "--fp16",
    ]

    if args.resume_from_checkpoint:
        command.extend(["--resume_from_checkpoint", args.resume_from_checkpoint])

    run_cmd(command, cwd=repo_root)


if __name__ == "__main__":
    main()
