#!/usr/bin/env python3
"""Analyze Whisper prediction errors from JSONL outputs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

from jiwer import wer


def _contains_digit(text: str) -> bool:
    return bool(re.search(r"\d", text))


def _repetition_score(text: str) -> int:
    tokens = text.split()
    repeats = 0
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            repeats += 1
    return repeats


def classify(reference: str, prediction: str) -> List[str]:
    tags: List[str] = []
    if _contains_digit(reference) or _contains_digit(prediction):
        tags.append("numbers")
    if _repetition_score(prediction) >= 2:
        tags.append("repetition")
    if len(reference.split()) >= 25:
        tags.append("long_utterance")

    pred_words = set(prediction.split())
    ref_words = set(reference.split())
    if len(ref_words) > 0:
        missing_ratio = len(ref_words - pred_words) / len(ref_words)
        if missing_ratio > 0.6:
            tags.append("high_omission")
    if not prediction.strip():
        tags.append("empty_prediction")
    if not tags:
        tags.append("other")
    return tags


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Whisper prediction errors")
    parser.add_argument("--predictions", type=str, required=True, help="Path to JSONL predictions")
    parser.add_argument("--out_json", type=str, default="", help="Optional output JSON path")
    parser.add_argument("--out_md", type=str, default="", help="Optional output Markdown path")
    parser.add_argument("--top_k", type=int, default=20, help="Number of worst samples to include")
    args = parser.parse_args()

    rows: List[Dict[str, object]] = []
    with Path(args.predictions).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            reference = str(item.get("reference", ""))
            prediction = str(item.get("prediction", ""))
            sample_wer = wer(reference, prediction) if reference.strip() else 0.0
            tags = classify(reference, prediction)
            rows.append(
                {
                    "id": item.get("id"),
                    "reference": reference,
                    "prediction": prediction,
                    "sample_wer": sample_wer,
                    "tags": tags,
                }
            )

    rows_sorted = sorted(rows, key=lambda x: float(x["sample_wer"]), reverse=True)
    tag_counter = Counter(tag for row in rows for tag in row["tags"])

    summary = {
        "num_samples": len(rows),
        "tag_distribution": dict(tag_counter),
        "worst_samples": rows_sorted[: args.top_k],
    }

    print(json.dumps({"num_samples": summary["num_samples"], "tag_distribution": summary["tag_distribution"]}, ensure_ascii=False, indent=2))

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.out_md:
        lines = [
            "# Whisper Error Analysis",
            "",
            f"- num_samples: {summary['num_samples']}",
            "- tag_distribution:",
        ]
        for key, value in summary["tag_distribution"].items():
            lines.append(f"  - {key}: {value}")
        lines.append("")
        lines.append("## Worst Samples")
        lines.append("")
        lines.append("| id | sample_wer | tags |")
        lines.append("|---:|---:|---|")
        for row in summary["worst_samples"]:
            tags = ", ".join(row["tags"])
            lines.append(f"| {row['id']} | {row['sample_wer']:.3f} | {tags} |")
        Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
