from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def read_json(path: Path) -> Dict[str, Any] | List[Dict[str, Any]]:
    """Read a JSON document using UTF-8 encoding."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL rows and skip malformed records safely."""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def list_audio_files(path: Path) -> List[Path]:
    """List audio files recursively for known extensions."""
    if not path.exists():
        return []
    exts = (".wav", ".mp3", ".flac", ".m4a")
    return [file_path for file_path in path.rglob("*") if file_path.suffix.lower() in exts]
