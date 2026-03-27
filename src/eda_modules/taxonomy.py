from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import pandas as pd
from IPython.display import display


def run_taxonomy_analysis(ctx: Dict[str, Any], pred_map: Dict[str, pd.DataFrame], quality_df: pd.DataFrame) -> Dict[str, Any]:
    """Create taxonomy dataframe by combining structural, acoustic and linguistic indicators."""

    def tok(value: str) -> List[str]:
        return re.findall(r"\w+", (value or "").lower(), flags=re.UNICODE)

    def normalize_sample_id(value: object) -> str:
        text = str(value or "").strip()
        if not text:
            return text
        if any(sep in text for sep in ["\\", "/"]) or text.lower().endswith((".wav", ".mp3", ".flac")):
            return Path(text).stem
        return text

    def edit_counts(ref_tokens: List[str], hyp_tokens: List[str]) -> Dict[str, int]:
        n, m = len(ref_tokens), len(hyp_tokens)
        dp = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            dist, sub, ins, dele = dp[i - 1][0]
            dp[i][0] = (dist + 1, sub, ins, dele + 1)
        for j in range(1, m + 1):
            dist, sub, ins, dele = dp[0][j - 1]
            dp[0][j] = (dist + 1, sub, ins + 1, dele)

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    sub = dp[i - 1][j - 1]
                    ins = dp[i][j - 1]
                    dele = dp[i - 1][j]
                    candidates = [
                        (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                        (ins[0] + 1, ins[1], ins[2] + 1, ins[3]),
                        (dele[0] + 1, dele[1], dele[2], dele[3] + 1),
                    ]
                    dp[i][j] = min(candidates, key=lambda x: x[0])

        _, sub, ins, dele = dp[n][m]
        return {"substitutions": sub, "insertions": ins, "deletions": dele}

    def sample_wer_from_tokens(ref_tokens: List[str], hyp_tokens: List[str]) -> float:
        counts = edit_counts(ref_tokens, hyp_tokens)
        return (counts["substitutions"] + counts["insertions"] + counts["deletions"]) / max(1, len(ref_tokens))

    def resolve_audio_path_for_id(sample_id: str) -> Optional[Path]:
        candidates = list((ctx["RUN_V1_DIR"] / "aligned_holdout" / "audio").glob(f"{sample_id}*"))
        if candidates:
            return candidates[0]
        fallback = ctx["ROOT"] / "data" / "aligned_raw_v1_improved" / "audio" / f"{sample_id}.wav"
        return fallback if fallback.exists() else None

    def snr_proxy_for_audio_path(audio_path: Optional[Path]) -> float:
        if audio_path is None or not audio_path.exists():
            return np.nan
        try:
            y, _ = librosa.load(audio_path, sr=16000, mono=True)
            if y.size == 0:
                return np.nan
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512).flatten()
            if len(rms) == 0:
                return np.nan
            floor = max(float(np.percentile(rms, 10)), 1e-8)
            mean_rms = max(float(np.mean(rms)), 1e-8)
            return 20.0 * np.log10(mean_rms / floor)
        except Exception:
            return np.nan

    def classify_error(
        wer_value: float,
        edits: Dict[str, int],
        n_ref_tokens: int,
        aligned_ratio: float,
        cps_value: float,
        snr_proxy_db: float,
        snr_threshold: float,
    ) -> str:
        sub_ratio = edits["substitutions"] / max(1, n_ref_tokens)
        del_ratio = edits["deletions"] / max(1, n_ref_tokens)
        ins_ratio = edits["insertions"] / max(1, n_ref_tokens)

        structural_flag = (pd.notna(aligned_ratio) and aligned_ratio < 0.60) or (pd.notna(cps_value) and cps_value > 20.0)
        acoustic_flag = pd.notna(snr_proxy_db) and pd.notna(snr_threshold) and snr_proxy_db < snr_threshold and (del_ratio + ins_ratio) >= 0.25
        linguistic_flag = sub_ratio >= max(del_ratio, ins_ratio)

        if structural_flag:
            return "Strukturne greške (tajming/alignment)"
        if acoustic_flag:
            return "Akustičke greške (buka/prekid govora)"
        if linguistic_flag or wer_value >= 0.60:
            return "Lingvističke greške (morfologija/ortografija)"
        return "Lingvističke greške (morfologija/ortografija)"

    taxonomy_df = pd.DataFrame()
    source_key = "v1_finetuned" if "v1_finetuned" in pred_map else ("v2" if "v2" in pred_map else None)
    if source_key is not None and {"ref", "pred"}.issubset(pred_map[source_key].columns):
        base_df = pred_map[source_key].dropna(subset=["ref", "pred"]).copy()

        if "audio_path" in base_df.columns:
            base_df["sample_key"] = base_df["audio_path"].map(normalize_sample_id)
        elif "sample_id" in base_df.columns:
            base_df["sample_key"] = base_df["sample_id"].map(normalize_sample_id)
        elif "id" in base_df.columns:
            base_df["sample_key"] = base_df["id"].map(normalize_sample_id)
        else:
            base_df["sample_key"] = base_df.index.map(str)

        quality_flags = pd.DataFrame(columns=["sample_key", "aligned_word_ratio", "cps"])
        if not quality_df.empty and "sample_id" in quality_df.columns:
            q = quality_df[["sample_id"] + [c for c in ["aligned_word_ratio", "cps"] if c in quality_df.columns]].copy()
            q["sample_key"] = q["sample_id"].map(normalize_sample_id)
            for col in ["aligned_word_ratio", "cps"]:
                if col not in q.columns:
                    q[col] = np.nan
            quality_flags = q[["sample_key", "aligned_word_ratio", "cps"]].drop_duplicates("sample_key")

        path_map: Dict[str, Path] = {}
        if "audio_path" in base_df.columns:
            for _, row in base_df[["sample_key", "audio_path"]].dropna().drop_duplicates("sample_key").iterrows():
                audio_path = Path(str(row["audio_path"]))
                if audio_path.exists():
                    path_map[str(row["sample_key"])] = audio_path

        snr_rows = []
        for sample_id in base_df["sample_key"].astype(str).unique().tolist():
            path = path_map.get(sample_id) or resolve_audio_path_for_id(sample_id)
            snr_rows.append({"sample_key": sample_id, "snr_proxy_db": snr_proxy_for_audio_path(path)})

        snr_map = pd.DataFrame(snr_rows, columns=["sample_key", "snr_proxy_db"])
        snr_threshold = snr_map["snr_proxy_db"].median(skipna=True) if not snr_map.empty else np.nan

        merged = base_df.merge(quality_flags, on="sample_key", how="left").merge(snr_map, on="sample_key", how="left")
        out_rows = []
        for _, row in merged.iterrows():
            ref_tokens = tok(str(row["ref"]))
            pred_tokens = tok(str(row["pred"]))
            edits = edit_counts(ref_tokens, pred_tokens)
            sample_wer = sample_wer_from_tokens(ref_tokens, pred_tokens)
            out_rows.append(
                {
                    "sample_id": row["sample_key"],
                    "sample_wer": sample_wer,
                    "aligned_word_ratio": row.get("aligned_word_ratio", np.nan),
                    "cps": row.get("cps", np.nan),
                    "snr_proxy_db": row.get("snr_proxy_db", np.nan),
                    "error_category": classify_error(
                        wer_value=sample_wer,
                        edits=edits,
                        n_ref_tokens=len(ref_tokens),
                        aligned_ratio=float(row["aligned_word_ratio"]) if pd.notna(row.get("aligned_word_ratio", np.nan)) else np.nan,
                        cps_value=float(row["cps"]) if pd.notna(row.get("cps", np.nan)) else np.nan,
                        snr_proxy_db=float(row["snr_proxy_db"]) if pd.notna(row.get("snr_proxy_db", np.nan)) else np.nan,
                        snr_threshold=float(snr_threshold) if pd.notna(snr_threshold) else np.nan,
                    ),
                    "substitutions": edits["substitutions"],
                    "insertions": edits["insertions"],
                    "deletions": edits["deletions"],
                }
            )

        taxonomy_df = pd.DataFrame(out_rows)
        display(taxonomy_df.head())

    if not taxonomy_df.empty:
        summary = (
            taxonomy_df.groupby("error_category", as_index=False)
            .agg(count=("sample_id", "count"), mean_wer=("sample_wer", "mean"))
            .sort_values("count", ascending=False)
        )
        summary["share_percent"] = 100.0 * summary["count"] / summary["count"].sum()
        display(summary)

    return {"taxonomy_df": taxonomy_df}
