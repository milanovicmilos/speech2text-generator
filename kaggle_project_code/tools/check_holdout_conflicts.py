#!/usr/bin/env python3
"""Check overlap between holdout list and aligned training manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _source_audio_set_from_holdout(rows: List[Dict[str, str]]) -> Set[str]:
    return {row.get("audio_path", "") for row in rows if row.get("audio_path")}


def _source_audio_set_from_manifest(rows: List[Dict[str, object]]) -> Set[str]:
    return {str(row.get("source_audio", "")) for row in rows if row.get("source_audio")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate holdout vs aligned manifest overlap")
    parser.add_argument("--holdout_list", type=str, required=True, help="Path to holdout_list.json")
    parser.add_argument("--manifest", type=str, required=True, help="Path to aligned manifest.json")
    parser.add_argument("--out", type=str, default="", help="Optional output JSON path")
    args = parser.parse_args()

    holdout_rows = _read_json(Path(args.holdout_list))
    manifest_rows = _read_json(Path(args.manifest))

    holdout_sources = _source_audio_set_from_holdout(holdout_rows)
    manifest_sources = _source_audio_set_from_manifest(manifest_rows)

    overlap = sorted(holdout_sources & manifest_sources)

    result = {
        "holdout_count": len(holdout_sources),
        "manifest_source_count": len(manifest_sources),
        "overlap_count": len(overlap),
        "is_clean": len(overlap) == 0,
        "overlap_sources": overlap,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
