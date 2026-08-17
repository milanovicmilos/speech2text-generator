#!/usr/bin/env python3
"""Summarize Whisper training metrics from HuggingFace trainer_state.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _get_eval_rows(log_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in log_history:
        if "eval_wer" in row or "eval_cer" in row or "eval_loss" in row:
            rows.append(row)
    return rows


def _safe_min(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return min(values)


def summarize(trainer_state_path: Path) -> Dict[str, Any]:
    content = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    log_history = content.get("log_history", [])
    eval_rows = _get_eval_rows(log_history)

    eval_wers = [float(r["eval_wer"]) for r in eval_rows if "eval_wer" in r]
    eval_cers = [float(r["eval_cer"]) for r in eval_rows if "eval_cer" in r]
    eval_losses = [float(r["eval_loss"]) for r in eval_rows if "eval_loss" in r]

    summary: Dict[str, Any] = {
        "trainer_state": str(trainer_state_path),
        "best_global_step": content.get("best_global_step"),
        "best_metric": content.get("best_metric"),
        "best_model_checkpoint": content.get("best_model_checkpoint"),
        "global_step": content.get("global_step"),
        "epoch": content.get("epoch"),
        "num_eval_points": len(eval_rows),
        "min_eval_wer": _safe_min(eval_wers),
        "min_eval_cer": _safe_min(eval_cers),
        "min_eval_loss": _safe_min(eval_losses),
    }
    return summary


def write_markdown(summary: Dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Whisper Run Summary",
        "",
        f"- trainer_state: {summary['trainer_state']}",
        f"- best_global_step: {summary['best_global_step']}",
        f"- best_metric: {summary['best_metric']}",
        f"- best_model_checkpoint: {summary['best_model_checkpoint']}",
        f"- global_step: {summary['global_step']}",
        f"- epoch: {summary['epoch']}",
        f"- num_eval_points: {summary['num_eval_points']}",
        f"- min_eval_wer: {summary['min_eval_wer']}",
        f"- min_eval_cer: {summary['min_eval_cer']}",
        f"- min_eval_loss: {summary['min_eval_loss']}",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Whisper trainer_state metrics")
    parser.add_argument(
        "--trainer_state",
        type=str,
        required=True,
        help="Path to trainer_state.json",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="",
        help="Optional path to write JSON summary",
    )
    parser.add_argument(
        "--out_md",
        type=str,
        default="",
        help="Optional path to write Markdown summary",
    )

    args = parser.parse_args()

    trainer_state_path = Path(args.trainer_state)
    if not trainer_state_path.exists():
        raise FileNotFoundError(f"trainer_state.json not found: {trainer_state_path}")

    summary = summarize(trainer_state_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.out_md:
        write_markdown(summary, Path(args.out_md))


if __name__ == "__main__":
    main()
