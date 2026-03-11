#!/usr/bin/env python3
"""Run EDA statistics over paired audio/text dataset for ASR project reporting."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Dict, List

import librosa


def _tokenize(text: str) -> List[str]:
    return [token for token in text.lower().split() if token]


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _find_pairs(data_dir: Path) -> List[tuple[Path, Path]]:
    if (data_dir / "audio").exists() and (data_dir / "text").exists():
        audio_root = data_dir / "audio"
        text_root = data_dir / "text"
    else:
        audio_root = data_dir
        text_root = data_dir

    audio_files: List[Path] = []
    for extension in ("*.mp3", "*.wav", "*.flac"):
        audio_files.extend(audio_root.rglob(extension))

    pairs: List[tuple[Path, Path]] = []
    for audio_path in sorted(audio_files):
        rel_path = audio_path.relative_to(audio_root)
        candidate_1 = text_root / rel_path.with_suffix(".txt")
        candidate_2 = text_root / f"{audio_path.stem}.txt"

        text_path = candidate_1 if candidate_1.exists() else candidate_2
        if text_path.exists():
            pairs.append((audio_path, text_path))

    return pairs


def summarize_dataset(data_dir: Path, top_k_tokens: int) -> Dict[str, object]:
    pairs = _find_pairs(data_dir)

    durations: List[float] = []
    word_counts: List[int] = []
    char_counts: List[int] = []
    words_per_second: List[float] = []
    token_counter: Counter[str] = Counter()

    for audio_path, text_path in pairs:
        text = _safe_read_text(text_path)
        tokens = _tokenize(text)

        try:
            duration = float(librosa.get_duration(path=str(audio_path)))
        except Exception:
            duration = 0.0

        durations.append(duration)
        word_counts.append(len(tokens))
        char_counts.append(len(text))

        if duration > 0:
            words_per_second.append(len(tokens) / duration)

        token_counter.update(tokens)

    total_duration = sum(durations)
    summary: Dict[str, object] = {
        "data_dir": str(data_dir),
        "num_pairs": len(pairs),
        "total_audio_seconds": total_duration,
        "total_audio_hours": total_duration / 3600.0,
        "duration_seconds": {
            "min": min(durations) if durations else 0.0,
            "max": max(durations) if durations else 0.0,
            "mean": mean(durations) if durations else 0.0,
            "median": median(durations) if durations else 0.0,
        },
        "text_words": {
            "min": min(word_counts) if word_counts else 0,
            "max": max(word_counts) if word_counts else 0,
            "mean": mean(word_counts) if word_counts else 0.0,
            "median": median(word_counts) if word_counts else 0.0,
        },
        "text_characters": {
            "min": min(char_counts) if char_counts else 0,
            "max": max(char_counts) if char_counts else 0,
            "mean": mean(char_counts) if char_counts else 0.0,
            "median": median(char_counts) if char_counts else 0.0,
        },
        "words_per_second": {
            "min": min(words_per_second) if words_per_second else 0.0,
            "max": max(words_per_second) if words_per_second else 0.0,
            "mean": mean(words_per_second) if words_per_second else 0.0,
            "median": median(words_per_second) if words_per_second else 0.0,
        },
        "top_tokens": token_counter.most_common(top_k_tokens),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ASR dataset statistics")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Dataset directory")
    parser.add_argument("--top_k_tokens", type=int, default=50, help="How many top tokens to report")
    parser.add_argument("--out_json", type=str, default="", help="Optional JSON output path")

    args = parser.parse_args()

    summary = summarize_dataset(Path(args.data_dir), args.top_k_tokens)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
