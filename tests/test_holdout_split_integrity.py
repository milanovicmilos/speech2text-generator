"""Data split integrity checks used in project delivery validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Set


def _read_stems(path: Path) -> Set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {Path(item).stem for item in data}


def test_holdout_lists_do_not_overlap() -> None:
    base = Path("data/aligned_raw_v1_improved/holdout")
    train_val_list = base / "train_val_list.json"
    holdout_list = base / "holdout_list.json"

    if not train_val_list.exists() or not holdout_list.exists():
        # Test is optional on fresh clones without generated holdout artifacts.
        return

    train_val_stems = _read_stems(train_val_list)
    holdout_stems = _read_stems(holdout_list)
    overlap = train_val_stems.intersection(holdout_stems)
    assert not overlap, f"Detected holdout leakage for stems: {sorted(overlap)[:10]}"
