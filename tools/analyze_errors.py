#!/usr/bin/env python3
"""Category-level ASR error analysis from predictions JSONL."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from jiwer import cer, wer

NAME_RE = re.compile(r"\b[A-ZČĆŽŠĐ][a-zčćžšđ]+\b")
DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}([./-]\d{2,4})?\b")
DIGIT_RE = re.compile(r"\d")
PUNCT_RE = re.compile(r"[.,!?;:()\[\]{}\-—–'\"]")


def _iter_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if not line.strip():
                continue
            yield json.loads(line)


def _token_set(text: str) -> set[str]:
    return {token.lower() for token in text.split() if token.strip()}


def _load_lexicon(path: str) -> set[str]:
    if not path:
        return set()
    lexicon_path = Path(path)
    if not lexicon_path.exists():
        raise FileNotFoundError(f"Lexicon file not found: {lexicon_path}")
    tokens = {
        line.strip().lower()
        for line in lexicon_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    return tokens


def _classify(
    reference_raw: str,
    prediction_raw: str,
    prediction_preprocessed: str,
    domain_lexicon: set[str],
) -> List[str]:
    tags: List[str] = []

    combined_raw = f"{reference_raw} {prediction_raw}"
    combined_preprocessed = prediction_preprocessed.lower()

    if DIGIT_RE.search(combined_raw):
        tags.append("numbers")
    if DATE_RE.search(combined_raw):
        tags.append("dates")
    if PUNCT_RE.search(combined_raw):
        tags.append("punctuation")
    if NAME_RE.search(reference_raw):
        tags.append("names")

    if domain_lexicon and (_token_set(combined_preprocessed) & domain_lexicon):
        tags.append("domain_entities")

    ref_tokens = _token_set(reference_raw)
    pred_tokens = _token_set(prediction_preprocessed)
    oov_tokens = pred_tokens - ref_tokens
    if oov_tokens:
        tags.append("oov_tokens")

    if not tags:
        tags.append("other")
    return tags


def _safe_metric(metric_fn, references: List[str], predictions: List[str]) -> float:
    if not references:
        return 0.0
    try:
        return float(metric_fn(references, predictions))
    except Exception:
        return 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze prediction errors by linguistic category")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions JSONL")
    parser.add_argument("--out_json", type=str, default="", help="Optional JSON output path")
    parser.add_argument(
        "--domain_lexicon",
        type=str,
        default="",
        help="Optional newline-delimited lexicon for custom domain_entities category",
    )
    args = parser.parse_args()

    domain_lexicon = _load_lexicon(args.domain_lexicon)

    rows = list(_iter_rows(Path(args.predictions)))
    if not rows:
        raise ValueError("No rows found in predictions JSONL")

    categories: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {"refs": [], "preds": []})
    category_counts = Counter()

    for row in rows:
        reference_raw = str(row.get("reference_raw", row.get("reference", "")))
        prediction_raw = str(row.get("prediction_raw", row.get("prediction", "")))
        reference_preprocessed = str(row.get("reference_preprocessed", row.get("reference", "")))
        prediction_preprocessed = str(row.get("prediction_preprocessed", row.get("prediction", "")))

        row_tags = _classify(
            reference_raw=reference_raw,
            prediction_raw=prediction_raw,
            prediction_preprocessed=prediction_preprocessed,
            domain_lexicon=domain_lexicon,
        )
        for tag in row_tags:
            category_counts[tag] += 1
            categories[tag]["refs"].append(reference_preprocessed)
            categories[tag]["preds"].append(prediction_preprocessed)

    category_metrics = {}
    for category, payload in categories.items():
        refs = payload["refs"]
        preds = payload["preds"]
        category_metrics[category] = {
            "count": len(refs),
            "wer": _safe_metric(wer, refs, preds),
            "cer": _safe_metric(cer, refs, preds),
        }

    result = {
        "num_samples": len(rows),
        "category_distribution": dict(category_counts),
        "category_metrics": category_metrics,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
