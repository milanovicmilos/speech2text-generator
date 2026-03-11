#!/usr/bin/env python3
"""Create strict article-level holdout splits from raw audio/text pairs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a", ".ogg")


def _collect_pairs(raw_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for audio_path in sorted(raw_dir.rglob("*")):
        if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        text_path = audio_path.with_suffix(".txt")
        if not text_path.exists():
            continue
        rows.append(
            {
                "sample_id": audio_path.stem,
                "audio_path": str(audio_path),
                "text_path": str(text_path),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create strict article-level holdout split")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Raw data directory")
    parser.add_argument("--out_dir", type=str, default="data/holdout", help="Output directory")
    parser.add_argument("--holdout_ratio", type=float, default=0.15, help="Holdout ratio in range (0,1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _collect_pairs(raw_dir)
    if not pairs:
        raise ValueError(f"No valid raw pairs found in {raw_dir}")

    rng = random.Random(args.seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)

    holdout_size = max(1, int(len(indices) * args.holdout_ratio))
    holdout_indices = set(indices[:holdout_size])

    train_val = [pairs[index] for index in indices if index not in holdout_indices]
    holdout = [pairs[index] for index in indices if index in holdout_indices]

    summary = {
        "raw_dir": str(raw_dir),
        "total_pairs": len(pairs),
        "train_val_pairs": len(train_val),
        "holdout_pairs": len(holdout),
        "holdout_ratio": args.holdout_ratio,
        "seed": args.seed,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "train_val_list.json").write_text(json.dumps(train_val, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "holdout_list.json").write_text(json.dumps(holdout, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
