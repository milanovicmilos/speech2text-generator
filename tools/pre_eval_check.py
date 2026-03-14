#!/usr/bin/env python3
"""Validate baseline/finetuned prediction integrity by strict sample_id merge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-EDA integrity check for Kaggle ASR results")
    parser.add_argument("--baseline", required=True, help="Path to baseline predictions JSONL")
    parser.add_argument("--finetuned", required=True, help="Path to finetuned predictions JSONL")
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    finetuned_path = Path(args.finetuned)

    baseline_df = read_jsonl(baseline_path)
    finetuned_df = read_jsonl(finetuned_path)

    for frame_name, frame in (("baseline", baseline_df), ("finetuned", finetuned_df)):
        if "sample_id" not in frame.columns:
            raise ValueError(f"Missing sample_id column in {frame_name} predictions")
        if frame["sample_id"].isna().any() or (frame["sample_id"].astype(str).str.strip() == "").any():
            raise ValueError(f"Empty sample_id values detected in {frame_name} predictions")
        if frame["sample_id"].duplicated().any():
            duplicates = int(frame["sample_id"].duplicated().sum())
            raise ValueError(f"Duplicate sample_id values in {frame_name}: {duplicates}")

    merged = baseline_df.merge(
        finetuned_df,
        on="sample_id",
        how="outer",
        indicator=True,
        suffixes=("_baseline", "_finetuned"),
    )

    total_baseline = int(len(baseline_df))
    total_finetuned = int(len(finetuned_df))
    matched = int((merged["_merge"] == "both").sum())

    if total_baseline == total_finetuned and matched == total_baseline:
        print("Integritet 100% - Spremno za EDA")
        print(f"matched={matched} baseline={total_baseline} finetuned={total_finetuned}")
        return

    missing_in_finetuned = merged.loc[merged["_merge"] == "left_only", "sample_id"].astype(str).tolist()
    missing_in_baseline = merged.loc[merged["_merge"] == "right_only", "sample_id"].astype(str).tolist()

    print("Integritet NIJE 100% - STOP")
    print(f"matched={matched} baseline={total_baseline} finetuned={total_finetuned}")
    if missing_in_finetuned:
        print(f"missing_in_finetuned={len(missing_in_finetuned)}")
        print("example_missing_in_finetuned:", missing_in_finetuned[:10])
    if missing_in_baseline:
        print(f"missing_in_baseline={len(missing_in_baseline)}")
        print("example_missing_in_baseline:", missing_in_baseline[:10])
    raise SystemExit(1)


if __name__ == "__main__":
    main()
