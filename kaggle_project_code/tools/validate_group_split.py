#!/usr/bin/env python3
"""Validate that group-aware split has no source overlap across train/val/test."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Set

from transformers import WhisperProcessor

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.data_loader import create_dataloaders  # noqa: E402


def _base_stem(stem: str) -> str:
    match = re.match(r"(?P<base>.+?)_chunk\d+$", stem)
    return match.group("base") if match else stem


def _collect_group_keys(subset) -> Set[str]:
    keys: Set[str] = set()
    base_dataset = subset.dataset
    for ds_index in subset.indices:
        file_idx = base_dataset.valid_indices[ds_index]
        audio_path = base_dataset.audio_files[file_idx]
        keys.add(_base_stem(audio_path.stem))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate no overlap between split groups")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Dataset directory")
    parser.add_argument("--processor", type=str, default="openai/whisper-tiny", help="Processor checkpoint")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for loader creation")
    parser.add_argument("--seed", type=int, default=42, help="Split seed")
    parser.add_argument("--out", type=str, default="", help="Optional output JSON path")

    args = parser.parse_args()

    processor = WhisperProcessor.from_pretrained(args.processor)

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        processor=processor,
        batch_size=args.batch_size,
        seed=args.seed,
        group_split=True,
    )

    train_groups = _collect_group_keys(train_loader.dataset)
    val_groups = _collect_group_keys(val_loader.dataset)
    test_groups = _collect_group_keys(test_loader.dataset)

    overlap_tv = sorted(train_groups & val_groups)
    overlap_tt = sorted(train_groups & test_groups)
    overlap_vt = sorted(val_groups & test_groups)

    report: Dict[str, object] = {
        "data_dir": args.data_dir,
        "seed": args.seed,
        "train_group_count": len(train_groups),
        "val_group_count": len(val_groups),
        "test_group_count": len(test_groups),
        "overlap_train_val": overlap_tv,
        "overlap_train_test": overlap_tt,
        "overlap_val_test": overlap_vt,
        "is_clean": len(overlap_tv) == 0 and len(overlap_tt) == 0 and len(overlap_vt) == 0,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
