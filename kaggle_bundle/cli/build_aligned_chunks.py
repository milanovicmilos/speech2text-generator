#!/usr/bin/env python3
"""Build alignment-first training chunks from raw long-form audio/text pairs.

This script uses faster-whisper word timestamps to segment long recordings
into <= `target_chunk_seconds` audio chunks and maps each chunk to a slice of
reference words using token-level sequence matching.

Output structure:
- <out_dir>/audio/*.wav
- <out_dir>/text/*.txt
- <out_dir>/report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
from difflib import SequenceMatcher

import numpy as np
import soundfile as sf
import librosa
from faster_whisper import WhisperModel


logger = logging.getLogger(__name__)


WORD_RE = re.compile(r"\S+")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _normalize_token(token: str) -> str:
    token = token.lower().strip()
    token = re.sub(r"[^a-zA-Z0-9čćžšđČĆŽŠĐ]+", "", token)
    return token


def _split_words(text: str) -> List[str]:
    return WORD_RE.findall(text)


@dataclass
class WordTs:
    word: str
    start: float
    end: float


@dataclass
class PairResult:
    source_audio: str
    source_text: str
    chunks_written: int
    skipped: bool
    reason: str
    chunk_manifest_rows: List[dict]
    suspicious_chunks: int


def _collect_pairs(raw_dir: Path) -> List[Tuple[Path, Path]]:
    audio_files: List[Path] = []
    for ext in ("*.mp3", "*.wav", "*.flac", "*.m4a", "*.ogg"):
        audio_files.extend(raw_dir.rglob(ext))

    pairs: List[Tuple[Path, Path]] = []
    for audio_path in sorted(audio_files):
        text_path = audio_path.with_suffix(".txt")
        if text_path.exists():
            pairs.append((audio_path, text_path))
    return pairs


def _extract_word_timestamps(model: WhisperModel, audio_path: Path, language: str) -> List[WordTs]:
    segments, _info = model.transcribe(
        str(audio_path),
        language=language,
        task="transcribe",
        word_timestamps=True,
        vad_filter=True,
        beam_size=1,
        temperature=0.0,
    )

    words: List[WordTs] = []
    for seg in segments:
        if seg.words is None:
            continue
        for word in seg.words:
            if word.start is None or word.end is None:
                continue
            w = (word.word or "").strip()
            if not w:
                continue
            words.append(WordTs(word=w, start=float(word.start), end=float(word.end)))

    return words


def _build_hyp_to_ref_map(hyp_words: Sequence[str], ref_words: Sequence[str]) -> List[int]:
    """Build monotonic mapping hyp_index -> nearest ref_index via SequenceMatcher blocks."""
    if not hyp_words or not ref_words:
        return []

    matcher = SequenceMatcher(a=hyp_words, b=ref_words, autojunk=False)
    blocks = matcher.get_matching_blocks()

    anchors: List[Tuple[int, int]] = []
    for block in blocks:
        for offset in range(block.size):
            anchors.append((block.a + offset, block.b + offset))

    if not anchors:
        anchors = [(0, 0), (len(hyp_words) - 1, len(ref_words) - 1)]
    else:
        if anchors[0][0] != 0:
            anchors.insert(0, (0, max(0, anchors[0][1] - anchors[0][0])))
        if anchors[-1][0] != len(hyp_words) - 1:
            anchors.append((len(hyp_words) - 1, min(len(ref_words) - 1, anchors[-1][1] + (len(hyp_words) - 1 - anchors[-1][0]))))

    anchors = sorted(anchors, key=lambda x: x[0])

    mapping: List[int] = [0] * len(hyp_words)
    cursor = 0
    for index in range(len(hyp_words)):
        while cursor + 1 < len(anchors) and anchors[cursor + 1][0] <= index:
            cursor += 1

        left_h, left_r = anchors[cursor]
        if cursor + 1 < len(anchors):
            right_h, right_r = anchors[cursor + 1]
            if right_h == left_h:
                mapped = left_r
            else:
                ratio = (index - left_h) / float(right_h - left_h)
                mapped = int(round(left_r + ratio * (right_r - left_r)))
        else:
            mapped = left_r

        mapped = max(0, min(mapped, len(ref_words) - 1))
        mapping[index] = mapped

    return mapping


def _group_word_indices_by_time(words: Sequence[WordTs], target_chunk_seconds: float) -> List[Tuple[int, int]]:
    if not words:
        return []

    groups: List[Tuple[int, int]] = []
    start_idx = 0
    current_start_t = words[0].start

    for idx in range(1, len(words)):
        span = words[idx].end - current_start_t
        if span > target_chunk_seconds:
            groups.append((start_idx, idx - 1))
            start_idx = idx
            current_start_t = words[idx].start

    groups.append((start_idx, len(words) - 1))
    return groups


def _safe_stem(path: Path) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-.]+", "_", path.stem)


def _process_pair(
    audio_path: Path,
    text_path: Path,
    out_audio_dir: Path,
    out_text_dir: Path,
    model: WhisperModel,
    language: str,
    target_chunk_seconds: float,
    min_chunk_seconds: float,
    sample_rate: int,
    qa_unmatched_threshold: float,
    qa_max_chunk_seconds: float,
) -> PairResult:
    try:
        raw_text = text_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        return PairResult(str(audio_path), str(text_path), 0, True, f"read_text_failed:{exc}", [], 0)

    if not raw_text:
        return PairResult(str(audio_path), str(text_path), 0, True, "empty_text", [], 0)

    ref_words = _split_words(raw_text)
    if len(ref_words) < 5:
        return PairResult(str(audio_path), str(text_path), 0, True, "too_few_ref_words", [], 0)

    try:
        audio, sr = sf.read(str(audio_path), always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != sample_rate:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=sample_rate)
            sr = sample_rate
    except Exception as exc:
        return PairResult(str(audio_path), str(text_path), 0, True, f"read_audio_failed:{exc}", [], 0)

    words_ts = _extract_word_timestamps(model, audio_path, language)
    if len(words_ts) < 6:
        return PairResult(str(audio_path), str(text_path), 0, True, "too_few_asr_words", [], 0)

    hyp_words_norm = [_normalize_token(w.word) for w in words_ts]
    ref_words_norm = [_normalize_token(w) for w in ref_words]

    hyp_to_ref = _build_hyp_to_ref_map(hyp_words_norm, ref_words_norm)
    if not hyp_to_ref:
        return PairResult(str(audio_path), str(text_path), 0, True, "empty_alignment_map", [], 0)

    groups = _group_word_indices_by_time(words_ts, target_chunk_seconds=target_chunk_seconds)
    stem = _safe_stem(audio_path)

    written = 0
    suspicious_chunks = 0
    chunk_manifest_rows: List[dict] = []
    for chunk_idx, (left, right) in enumerate(groups):
        start_t = words_ts[left].start
        end_t = words_ts[right].end
        duration = end_t - start_t
        if duration < min_chunk_seconds:
            continue

        start_sample = int(max(0.0, start_t) * sample_rate)
        end_sample = int(min(len(audio) / sample_rate, end_t) * sample_rate)
        if end_sample <= start_sample:
            continue

        ref_left = hyp_to_ref[left]
        ref_right = hyp_to_ref[right]
        if ref_right < ref_left:
            ref_left, ref_right = ref_right, ref_left

        ref_left = max(0, min(ref_left, len(ref_words) - 1))
        ref_right = max(ref_left, min(ref_right, len(ref_words) - 1))
        chunk_text_words = ref_words[ref_left:ref_right + 1]
        if len(chunk_text_words) < 3:
            continue

        out_stem = f"{stem}_aligned_{chunk_idx:04d}"
        out_audio = out_audio_dir / f"{out_stem}.wav"
        out_text = out_text_dir / f"{out_stem}.txt"

        total_hyp_words = max(1, (right - left + 1))
        matched = 0
        for hyp_index in range(left, right + 1):
            ref_index = hyp_to_ref[hyp_index]
            hyp_token = hyp_words_norm[hyp_index]
            ref_token = ref_words_norm[ref_index] if 0 <= ref_index < len(ref_words_norm) else ""
            if hyp_token and ref_token and hyp_token == ref_token:
                matched += 1
        aligned_word_ratio = matched / float(total_hyp_words)

        suspicious_reasons: List[str] = []
        if duration > qa_max_chunk_seconds:
            suspicious_reasons.append("duration_exceeds_threshold")
        if (1.0 - aligned_word_ratio) > qa_unmatched_threshold:
            suspicious_reasons.append("high_unmatched_word_ratio")
        if suspicious_reasons:
            suspicious_chunks += 1

        try:
            sf.write(str(out_audio), audio[start_sample:end_sample], sample_rate)
            out_text.write_text(" ".join(chunk_text_words), encoding="utf-8")
            written += 1
            chunk_manifest_rows.append(
                {
                    "chunk_id": out_stem,
                    "chunk_audio": str(out_audio),
                    "chunk_text": str(out_text),
                    "source_audio": str(audio_path),
                    "source_text": str(text_path),
                    "start_sec": round(float(start_t), 3),
                    "end_sec": round(float(end_t), 3),
                    "duration_sec": round(float(duration), 3),
                    "ref_word_left": int(ref_left),
                    "ref_word_right": int(ref_right),
                    "hyp_word_left": int(left),
                    "hyp_word_right": int(right),
                    "aligned_word_ratio": round(float(aligned_word_ratio), 4),
                    "suspicious_reasons": suspicious_reasons,
                }
            )
        except Exception:
            continue

    if written == 0:
        return PairResult(str(audio_path), str(text_path), 0, True, "no_valid_chunks", [], 0)

    return PairResult(
        str(audio_path),
        str(text_path),
        written,
        False,
        "ok",
        chunk_manifest_rows,
        suspicious_chunks,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build aligned chunks from raw data")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Raw dataset root")
    parser.add_argument("--out_dir", type=str, default="data/aligned_raw_v1", help="Output dataset root")
    parser.add_argument("--language", type=str, default="sr", help="ASR language code")
    parser.add_argument("--model_size", type=str, default="small", help="faster-whisper model size/path")
    parser.add_argument("--target_chunk_seconds", type=float, default=24.0, help="Target max chunk length")
    parser.add_argument("--min_chunk_seconds", type=float, default=3.0, help="Minimum chunk length")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Required sample rate")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--compute_type", type=str, default="int8", help="faster-whisper compute type")
    parser.add_argument("--max_files", type=int, default=0, help="Optional limit for quick experiments")
    parser.add_argument(
        "--include_list_json",
        type=str,
        default="",
        help="Optional JSON list (from tools/create_holdout.py) used to include only selected raw audio paths",
    )
    parser.add_argument(
        "--qa_unmatched_threshold",
        type=float,
        default=0.4,
        help="Flag chunk when unmatched word ratio exceeds this threshold",
    )
    parser.add_argument(
        "--qa_max_chunk_seconds",
        type=float,
        default=30.0,
        help="Flag chunk when duration exceeds this threshold",
    )
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_audio_dir = out_dir / "audio"
    out_text_dir = out_dir / "text"
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    out_text_dir.mkdir(parents=True, exist_ok=True)

    pairs = _collect_pairs(raw_dir)

    if args.include_list_json:
        include_rows = json.loads(Path(args.include_list_json).read_text(encoding="utf-8"))
        include_audio_paths = {
            str(Path(row["audio_path"]).resolve())
            for row in include_rows
            if isinstance(row, dict) and row.get("audio_path")
        }
        pairs = [
            (audio_path, text_path)
            for (audio_path, text_path) in pairs
            if str(audio_path.resolve()) in include_audio_paths
        ]

    if args.max_files > 0:
        pairs = pairs[: args.max_files]

    logger.info("Found %s raw pairs to process", len(pairs))
    model = WhisperModel(args.model_size, device=args.device, compute_type=args.compute_type)

    results: List[PairResult] = []
    for index, (audio_path, text_path) in enumerate(pairs, start=1):
        logger.info("[%s/%s] %s", index, len(pairs), audio_path.name)
        result = _process_pair(
            audio_path=audio_path,
            text_path=text_path,
            out_audio_dir=out_audio_dir,
            out_text_dir=out_text_dir,
            model=model,
            language=args.language,
            target_chunk_seconds=args.target_chunk_seconds,
            min_chunk_seconds=args.min_chunk_seconds,
            sample_rate=args.sample_rate,
            qa_unmatched_threshold=args.qa_unmatched_threshold,
            qa_max_chunk_seconds=args.qa_max_chunk_seconds,
        )
        results.append(result)

    kept_pairs = sum(1 for r in results if not r.skipped)
    skipped_pairs = sum(1 for r in results if r.skipped)
    total_chunks = sum(r.chunks_written for r in results)
    all_chunk_rows = [row for result in results for row in result.chunk_manifest_rows]
    suspicious_chunks = sum(result.suspicious_chunks for result in results)

    report = {
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "pairs_total": len(results),
        "pairs_kept": kept_pairs,
        "pairs_skipped": skipped_pairs,
        "total_chunks_written": total_chunks,
        "avg_chunks_per_kept_pair": (total_chunks / kept_pairs) if kept_pairs else 0.0,
        "qa": {
            "unmatched_threshold": args.qa_unmatched_threshold,
            "max_chunk_seconds": args.qa_max_chunk_seconds,
            "suspicious_chunks": suspicious_chunks,
            "suspicious_chunk_ratio": (suspicious_chunks / total_chunks) if total_chunks else 0.0,
        },
        "skipped_examples": [
            {
                "audio": r.source_audio,
                "reason": r.reason,
            }
            for r in results
            if r.skipped
        ][:200],
    }

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(all_chunk_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Done. Report saved: %s", report_path)
    logger.info("Chunk manifest saved: %s", manifest_path)


if __name__ == "__main__":
    main()
