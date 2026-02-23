#!/usr/bin/env python3
"""Compare multiple Whisper metrics JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_metrics(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_path"] = str(path)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Whisper run metrics JSON files")
    parser.add_argument("--metrics", nargs="+", required=True, help="Paths to metrics JSON files")
    parser.add_argument("--out_md", type=str, default="", help="Optional markdown output")
    args = parser.parse_args()

    rows: List[Dict[str, object]] = [load_metrics(Path(p)) for p in args.metrics]
    rows = sorted(rows, key=lambda r: float(r.get("wer", 999.0)))

    print("Run comparison (sorted by WER):")
    for row in rows:
        print(f"- {row.get('_path')}: WER={row.get('wer'):.4f}, CER={row.get('cer'):.4f}, split={row.get('split')}")

    if args.out_md:
        lines = [
            "# Whisper Run Comparison",
            "",
            "| metrics_file | model_dir | split | num_samples | WER | CER |",
            "|---|---|---|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row.get('_path')} | {row.get('model_dir')} | {row.get('split')} | {row.get('num_samples')} | {float(row.get('wer', 0.0)):.4f} | {float(row.get('cer', 0.0)):.4f} |"
            )
        Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
