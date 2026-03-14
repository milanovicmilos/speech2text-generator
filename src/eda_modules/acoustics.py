from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from IPython.display import Markdown, display
from scipy import stats as scipy_stats

from .io_utils import list_audio_files, read_jsonl_records
from .stats import bonferroni_correction
from .visualization import style_plotly_figure


def _normalize_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize prediction schema for strict sample_id-based joins."""
    if df.empty:
        return df

    out = df.copy()
    if "sample_id" not in out.columns:
        raise ValueError("Missing required 'sample_id' column in prediction file.")

    if "wer" not in out.columns and "sample_wer" in out.columns:
        out = out.rename(columns={"sample_wer": "wer"})

    if "wer" not in out.columns:
        ref_col = None
        pred_col = None
        for candidate in ["reference", "ref", "reference_raw"]:
            if candidate in out.columns:
                ref_col = candidate
                break
        for candidate in ["prediction", "pred", "prediction_raw"]:
            if candidate in out.columns:
                pred_col = candidate
                break
        if ref_col is not None and pred_col is not None:
            def _sentence_wer_local(reference: str, hypothesis: str) -> float:
                ref = str(reference).split()
                hyp = str(hypothesis).split()
                m, n = len(ref), len(hyp)
                if m == 0:
                    return float(0 if n == 0 else n)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                for i in range(m + 1):
                    dp[i][0] = i
                for j in range(n + 1):
                    dp[0][j] = j
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        cost = 0 if ref[i - 1] == hyp[j - 1] else 1
                        dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
                return float(dp[m][n] / max(1, m))

            out["wer"] = [
                _sentence_wer_local(ref_val, pred_val)
                for ref_val, pred_val in zip(out[ref_col], out[pred_col])
            ]

    if "audio_path" not in out.columns:
        for candidate in ["path", "audio", "audio_file"]:
            if candidate in out.columns:
                out = out.rename(columns={candidate: "audio_path"})
                break

    out["sample_id"] = out["sample_id"].astype(str).str.strip()
    if out["sample_id"].eq("").any():
        raise ValueError("Empty sample_id values detected in prediction file.")
    if out["sample_id"].duplicated().any():
        duplicates = int(out["sample_id"].duplicated().sum())
        raise ValueError(f"Duplicate sample_id values detected in prediction file: {duplicates}")

    if "wer" in out.columns:
        out["wer"] = pd.to_numeric(out["wer"], errors="coerce")

    return out


def load_audio_stats(audio_paths: List[Path], max_files: int = 800) -> pd.DataFrame:
    """Compute robust per-file audio descriptors for a bounded sample."""
    rows: List[Dict[str, Any]] = []
    for audio_path in audio_paths[:max_files]:
        try:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
        except Exception:
            continue
        if sr <= 0 or y.size == 0:
            continue

        duration = len(y) / sr
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512).flatten()
        rms_mean = float(np.nanmean(rms)) if len(rms) else np.nan
        db = float(20 * np.log10(max(rms_mean, 1e-12))) if pd.notna(rms_mean) else np.nan
        if len(rms):
            p20 = np.percentile(rms, 20)
            silence_ratio = float(np.mean(rms < p20))
        else:
            silence_ratio = np.nan
        rows.append(
            {
                "path": str(audio_path),
                "file_name": audio_path.name,
                "duration_s": duration,
                "sr": sr,
                "rms_mean": rms_mean,
                "db_mean": db,
                "silence_ratio": silence_ratio,
            }
        )
    return pd.DataFrame(rows)


def _align_token_ops(ref_tokens: List[str], hyp_tokens: List[str]) -> List[Tuple[str, str, str]]:
    """Align token lists and return operation sequence (M,S,I,D)."""
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back: List[List[Optional[str]]] = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = "M"
            else:
                sub = dp[i - 1][j - 1] + 1
                ins = dp[i][j - 1] + 1
                dele = dp[i - 1][j] + 1
                best = min(sub, ins, dele)
                dp[i][j] = best
                back[i][j] = "S" if best == sub else ("I" if best == ins else "D")

    ops: List[Tuple[str, str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if op in ("M", "S"):
            ops.append((op, ref_tokens[i - 1], hyp_tokens[j - 1]))
            i -= 1
            j -= 1
        elif op == "I":
            ops.append(("I", "", hyp_tokens[j - 1]))
            j -= 1
        else:
            ops.append(("D", ref_tokens[i - 1], ""))
            i -= 1

    return list(reversed(ops))


def _snr_proxy_from_waveform(y: np.ndarray) -> float:
    """Estimate signal quality from waveform RMS distribution."""
    if y.size == 0:
        return np.nan
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512).flatten()
    if len(rms) == 0:
        return np.nan
    floor = max(float(np.percentile(rms, 10)), 1e-8)
    mean_rms = max(float(np.nanmean(rms)), 1e-8)
    return 20.0 * np.log10(mean_rms / floor)


def _resolve_audio_path(sample_id: str, raw_audio_path: str, ctx: Dict[str, Any]) -> Optional[Path]:
    """Resolve audio path robustly, with local sample_id fallbacks for Kaggle-origin metadata."""
    candidate = Path(str(raw_audio_path)) if raw_audio_path else None
    if candidate is not None and candidate.exists():
        return candidate

    fallback_dirs = [
        Path(ctx.get("RUN_V1_DIR", Path())) / "aligned_holdout" / "audio",
        Path(ctx.get("RUN_V1_DIR", Path())) / "aligned_train" / "audio",
        Path(ctx.get("RUN_V2_DIR", Path())) / "aligned_holdout" / "audio",
        Path(ctx.get("RUN_V2_DIR", Path())) / "aligned_train" / "audio",
        ctx["ROOT"] / "data" / "aligned_raw_v1_rerun" / "audio",
        ctx["ROOT"] / "data" / "aligned_raw_v1_improved" / "audio",
    ]

    for audio_dir in fallback_dirs:
        if not audio_dir.exists():
            continue
        direct = audio_dir / f"{sample_id}.wav"
        if direct.exists():
            return direct
        matches = sorted(audio_dir.glob(f"{sample_id}*"))
        if matches:
            return matches[0]

    return None


def run_audio_analysis(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Run full acoustic analytics, including ZCR/SNR bridge diagnostics."""
    aligned_audio_paths = list_audio_files(ctx["RUN_V2_DIR"] / "aligned_train" / "audio")
    if not aligned_audio_paths:
        aligned_audio_paths = list_audio_files(ctx["ROOT"] / "data" / "aligned_raw_v1_improved" / "audio")

    audio_df = load_audio_stats(aligned_audio_paths, max_files=800)
    print("Analyzed audio files:", len(audio_df))
    display(audio_df.head())

    zcr_error_bridge_stats: Dict[str, float] = {}
    if audio_df.empty:
        return {"audio_df": audio_df, "zcr_error_bridge_stats": zcr_error_bridge_stats}

    q25 = audio_df["duration_s"].quantile(0.25)
    med = audio_df["duration_s"].median()
    q75 = audio_df["duration_s"].quantile(0.75)

    plt.figure(figsize=(14, 6))
    sns.histplot(audio_df["duration_s"], bins=40, kde=True, color="#2D7FF9")
    plt.axvline(q25, color="orange", linestyle="--", label="P25")
    plt.axvline(med, color="red", linestyle="-", label="Median")
    plt.axvline(q75, color="green", linestyle="--", label="P75")
    plt.title("Distribucija trajanja chunkova sa KDE")
    plt.xlabel("Trajanje (sekunde)")
    plt.ylabel("Broj chunkova")
    plt.legend()
    plt.show()
    print({"P25": round(q25, 2), "Median": round(med, 2), "P75": round(q75, 2)})

    rms_floor = np.maximum(audio_df["rms_mean"].quantile(0.1), 1e-8)
    audio_df["snr_proxy_db"] = 20.0 * np.log10(np.maximum(audio_df["rms_mean"], 1e-8) / rms_floor)
    audio_df["snr_class"] = np.where(
        audio_df["snr_proxy_db"] >= audio_df["snr_proxy_db"].median(),
        "High SNR (studio)",
        "Low SNR / Field Report",
    )

    snr_summary = (
        audio_df.groupby("snr_class")
        .agg(
            count=("path", "count"),
            duration_median=("duration_s", "median"),
            db_mean_median=("db_mean", "median"),
            zcr_proxy=("silence_ratio", "median"),
        )
        .reset_index()
    )
    display(snr_summary)

    fig = px.box(
        audio_df,
        y="db_mean",
        points="all",
        color="snr_class",
        title="Distribucija prosečne jačine signala po SNR klasi",
    )
    style_plotly_figure(fig, x_title="SNR klasa", y_title="Srednji nivo signala [dB]")
    fig.show()

    fig = px.histogram(
        audio_df,
        x="silence_ratio",
        color="snr_class",
        nbins=40,
        barmode="overlay",
        title="Udeo tišine po SNR klasi",
    )
    style_plotly_figure(fig, x_title="Udeo tišine [0-1]", y_title="Broj segmenta [count]")
    fig.show()

    zcr_rows = []
    sample_for_zcr = audio_df.sample(min(250, len(audio_df)), random_state=42)
    for _, row in sample_for_zcr.iterrows():
        try:
            y, _ = librosa.load(row["path"], sr=16000, mono=True)
            if y.size == 0:
                continue
            zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512).flatten()
            zcr_rows.append({"path": row["path"], "snr_class": row["snr_class"], "zcr_mean": float(np.nanmean(zcr))})
        except Exception:
            continue

    zcr_df = pd.DataFrame(zcr_rows)
    if not zcr_df.empty:
        fig = px.violin(
            zcr_df,
            x="snr_class",
            y="zcr_mean",
            box=True,
            points="all",
            title="ZCR profil po SNR klasi",
        )
        style_plotly_figure(fig, x_title="SNR klasa", y_title="Srednji ZCR [0-1]")
        fig.show()

    pred_path = ctx.get(
        "V1_FINETUNED_PRED",
        ctx["ROOT"] / "logs" / "verification" / "dataset_strategy" / "raw_aligned_v2026_run3_improved_eval_aligned_test_predictions.jsonl",
    )

    if pred_path.exists():
        rows = read_jsonl_records(pred_path)
        fricatives = set("sšzž")
        bridge_rows = []

        for row in rows:
            sample_id_value = str(row.get("sample_id", "")).strip()
            if not sample_id_value:
                raise ValueError(
                    "Missing required 'sample_id' in acoustic bridge input rows. "
                    "Strict sample_id-based analysis is required."
                )
            audio_path = _resolve_audio_path(sample_id_value, str(row.get("audio_path", "")), ctx)
            if audio_path is None or not audio_path.exists():
                continue
            try:
                y, _ = librosa.load(audio_path, sr=16000, mono=True)
            except Exception:
                continue
            if y.size == 0:
                continue

            zcr_mean = float(np.nanmean(librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512)))
            snr_proxy_db = _snr_proxy_from_waveform(y)
            ref_tokens = re.findall(r"\w+", str(row.get("reference_raw", "")).lower(), flags=re.UNICODE)
            hyp_tokens = re.findall(r"\w+", str(row.get("prediction_raw", "")).lower(), flags=re.UNICODE)
            ops = _align_token_ops(ref_tokens, hyp_tokens)

            fricative_substitutions = 0
            total_substitutions = 0
            reference_fricative_tokens = 0
            for op, ref_tok, hyp_tok in ops:
                if set(ref_tok) & fricatives:
                    reference_fricative_tokens += 1
                if op == "S":
                    total_substitutions += 1
                    if set(ref_tok) & fricatives and ref_tok != hyp_tok:
                        fricative_substitutions += 1

            bridge_rows.append(
                {
                    "sample_id": sample_id_value,
                    "zcr_mean": zcr_mean,
                    "snr_proxy_db": snr_proxy_db,
                    "fricative_substitutions": fricative_substitutions,
                    "total_substitutions": total_substitutions,
                    "reference_fricative_tokens": max(1, reference_fricative_tokens),
                    "fricative_sub_share": fricative_substitutions / max(1, total_substitutions),
                    "fricative_ref_error_rate": fricative_substitutions / max(1, reference_fricative_tokens),
                }
            )

        bridge_df = pd.DataFrame(bridge_rows)
        bridge_df = bridge_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["zcr_mean", "snr_proxy_db", "fricative_sub_share"])
        if not bridge_df.empty:
            zcr_threshold = bridge_df["zcr_mean"].median()
            snr_threshold = bridge_df["snr_proxy_db"].median()
            bridge_df["high_zcr"] = bridge_df["zcr_mean"] >= zcr_threshold
            bridge_df["low_snr"] = bridge_df["snr_proxy_db"] < snr_threshold
            bridge_df["acoustic_regime"] = np.select(
                [
                    bridge_df["high_zcr"] & bridge_df["low_snr"],
                    bridge_df["high_zcr"] & ~bridge_df["low_snr"],
                    ~bridge_df["high_zcr"] & bridge_df["low_snr"],
                ],
                ["High ZCR + Low SNR", "High ZCR + High SNR", "Low ZCR + Low SNR"],
                default="Low ZCR + High SNR",
            )

            summary = (
                bridge_df.groupby("acoustic_regime", as_index=False)
                .agg(
                    samples=("sample_id", "count"),
                    mean_zcr=("zcr_mean", "mean"),
                    mean_snr_proxy_db=("snr_proxy_db", "mean"),
                    mean_fricative_sub_share=("fricative_sub_share", "mean"),
                    mean_fricative_ref_error_rate=("fricative_ref_error_rate", "mean"),
                )
                .sort_values("mean_fricative_sub_share", ascending=False)
            )
            display(summary)

            spearman = scipy_stats.spearmanr(bridge_df["zcr_mean"], bridge_df["fricative_sub_share"], nan_policy="omit")
            high_zcr_low_snr = bridge_df[bridge_df["high_zcr"] & bridge_df["low_snr"]]["fricative_sub_share"].dropna()
            others = bridge_df[~(bridge_df["high_zcr"] & bridge_df["low_snr"])]["fricative_sub_share"].dropna()
            mann_whitney = (
                scipy_stats.mannwhitneyu(high_zcr_low_snr, others, alternative="greater")
                if len(high_zcr_low_snr) > 0 and len(others) > 0
                else None
            )

            n1 = int(len(high_zcr_low_snr))
            n2 = int(len(others))
            rank_biserial = np.nan
            if mann_whitney is not None and n1 > 0 and n2 > 0:
                u_stat = float(mann_whitney.statistic)
                rank_biserial = (2.0 * u_stat / (n1 * n2)) - 1.0

            raw_p = {
                "spearman_zcr_vs_fricative_sub_share": float(spearman.pvalue),
                "mann_whitney_highz_lowsnr_vs_others": float(mann_whitney.pvalue) if mann_whitney is not None else np.nan,
            }
            adjusted_p = bonferroni_correction(raw_p)

            zcr_error_bridge_stats = {
                "zcr_threshold": float(zcr_threshold),
                "snr_threshold": float(snr_threshold),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "spearman_p_bonferroni": float(adjusted_p["spearman_zcr_vs_fricative_sub_share"]),
                "high_zcr_low_snr_mean_fricative_sub_share": float(high_zcr_low_snr.mean()) if len(high_zcr_low_snr) else np.nan,
                "other_regimes_mean_fricative_sub_share": float(others.mean()) if len(others) else np.nan,
                "mann_whitney_p": float(mann_whitney.pvalue) if mann_whitney is not None else np.nan,
                "mann_whitney_p_bonferroni": float(adjusted_p["mann_whitney_highz_lowsnr_vs_others"]),
                "mann_whitney_rank_biserial": float(rank_biserial),
                "high_zcr_low_snr_n": n1,
                "other_regimes_n": n2,
            }
            print("ZCR-error bridge stats:", zcr_error_bridge_stats)

            fig = px.box(
                bridge_df,
                x="acoustic_regime",
                y="fricative_sub_share",
                color="acoustic_regime",
                points="all",
                title="Frikativne supstitucije po akustičkom režimu",
            )
            style_plotly_figure(
                fig,
                x_title="Akustički režim",
                y_title="Udeo frikativnih supstitucija među svim supstitucijama [0-1]",
            )
            fig.show()

            if (
                pd.notna(zcr_error_bridge_stats.get("mann_whitney_p_bonferroni", np.nan))
                and zcr_error_bridge_stats["mann_whitney_p_bonferroni"] < 0.05
            ):
                display(
                    Markdown(
                        "Uočene su statističke indikacije obogaćenja frikativnih supstitucija u High-ZCR/Low-SNR režimu. "
                        "Nalaz ukazuje na moguću povezanost akustičke nestabilnosti i fonetske distorzije, "
                        "ali se ne interpretira kao stroga kauzalnost bez dodatnih kontrolisanih eksperimenata."
                    )
                )

    baseline_pred_path = ctx.get("V1_BASELINE_PRED")
    finetuned_pred_path = ctx.get("V1_FINETUNED_PRED")
    if baseline_pred_path and finetuned_pred_path and Path(baseline_pred_path).exists() and Path(finetuned_pred_path).exists():
        baseline_df = _normalize_prediction_frame(pd.DataFrame(read_jsonl_records(Path(baseline_pred_path))))
        finetuned_df = _normalize_prediction_frame(pd.DataFrame(read_jsonl_records(Path(finetuned_pred_path))))

        required_cols = {"sample_id", "wer"}
        if required_cols.issubset(baseline_df.columns) and required_cols.issubset(finetuned_df.columns):
            compare_df = pd.merge(
                baseline_df[["sample_id", "wer", "audio_path"]].rename(columns={"wer": "wer_baseline"}),
                finetuned_df[["sample_id", "wer", "audio_path"]].rename(columns={"wer": "wer_finetuned", "audio_path": "audio_path_ft"}),
                on="sample_id",
                how="inner",
            )
            if len(compare_df) > 0:
                compare_df["audio_path"] = compare_df["audio_path_ft"].where(
                    compare_df["audio_path_ft"].notna(),
                    compare_df["audio_path"],
                )
                compare_df["wer_delta"] = compare_df["wer_baseline"] - compare_df["wer_finetuned"]

                acoustic_rows: List[Dict[str, Any]] = []
                for _, row in compare_df.iterrows():
                    audio_path = _resolve_audio_path(str(row["sample_id"]), str(row.get("audio_path", "")), ctx)
                    if audio_path is None or not audio_path.exists():
                        continue
                    try:
                        y, _ = librosa.load(audio_path, sr=16000, mono=True)
                    except Exception:
                        continue
                    if y.size == 0:
                        continue

                    zcr_mean = float(np.nanmean(librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512)))
                    snr_proxy_db = _snr_proxy_from_waveform(y)
                    acoustic_rows.append(
                        {
                            "sample_id": str(row["sample_id"]),
                            "zcr_mean": zcr_mean,
                            "snr_proxy_db": snr_proxy_db,
                        }
                    )

                acoustic_df = pd.DataFrame(acoustic_rows)
                if not acoustic_df.empty:
                    compare_df = compare_df.merge(acoustic_df, on="sample_id", how="inner")
                    compare_df = compare_df.replace([np.inf, -np.inf], np.nan).dropna(
                        subset=["wer_baseline", "wer_finetuned", "wer_delta", "zcr_mean", "snr_proxy_db"]
                    )

                if not compare_df.empty:
                    snr_threshold = float(compare_df["snr_proxy_db"].median())
                    zcr_threshold = float(compare_df["zcr_mean"].median())
                    compare_df["snr_class"] = np.where(compare_df["snr_proxy_db"] < snr_threshold, "Low SNR", "High SNR")
                    compare_df["zcr_class"] = np.where(compare_df["zcr_mean"] >= zcr_threshold, "High ZCR", "Low ZCR")

                    by_snr = (
                        compare_df.groupby("snr_class", as_index=False)
                        .agg(
                            n=("sample_id", "count"),
                            mean_wer_baseline=("wer_baseline", "mean"),
                            mean_wer_finetuned=("wer_finetuned", "mean"),
                            mean_wer_delta=("wer_delta", "mean"),
                            median_wer_delta=("wer_delta", "median"),
                        )
                        .sort_values("snr_class")
                    )
                    display(by_snr)

                    long_df = compare_df.melt(
                        id_vars=["sample_id", "snr_class", "zcr_class"],
                        value_vars=["wer_baseline", "wer_finetuned"],
                        var_name="model_variant",
                        value_name="wer",
                    )
                    fig = px.box(
                        long_df,
                        x="snr_class",
                        y="wer",
                        color="model_variant",
                        points="all",
                        title="WER pre/post fine-tuninga po SNR klasi",
                    )
                    style_plotly_figure(fig, x_title="SNR klasa", y_title="WER [0-1]", show_target_wer=True, y_is_wer=True)
                    fig.show()

                    fig = px.scatter(
                        compare_df,
                        x="snr_proxy_db",
                        y="wer_delta",
                        color="zcr_class",
                        trendline="ols",
                        title="Dobitak fine-tuninga (baseline - finetuned WER) vs SNR",
                    )
                    style_plotly_figure(fig, x_title="SNR proxy [dB]", y_title="WER delta (pozitivno = poboljšanje)")
                    fig.show()

                    low_snr_delta = compare_df.loc[compare_df["snr_class"] == "Low SNR", "wer_delta"].dropna()
                    high_snr_delta = compare_df.loc[compare_df["snr_class"] == "High SNR", "wer_delta"].dropna()

                    p_raw: Dict[str, float] = {}
                    if len(low_snr_delta) > 0 and len(high_snr_delta) > 0:
                        mann = scipy_stats.mannwhitneyu(low_snr_delta, high_snr_delta, alternative="two-sided")
                        p_raw["mann_whitney_delta_low_vs_high_snr"] = float(mann.pvalue)
                    else:
                        mann = None

                    snr_spearman = scipy_stats.spearmanr(compare_df["snr_proxy_db"], compare_df["wer_delta"], nan_policy="omit")
                    zcr_spearman = scipy_stats.spearmanr(compare_df["zcr_mean"], compare_df["wer_delta"], nan_policy="omit")
                    p_raw["spearman_snr_vs_delta"] = float(snr_spearman.pvalue)
                    p_raw["spearman_zcr_vs_delta"] = float(zcr_spearman.pvalue)

                    adjusted = bonferroni_correction(p_raw)
                    zcr_error_bridge_stats.update(
                        {
                            "comparison_samples": int(len(compare_df)),
                            "comparison_snr_threshold": snr_threshold,
                            "comparison_zcr_threshold": zcr_threshold,
                            "mean_delta_low_snr": float(low_snr_delta.mean()) if len(low_snr_delta) else np.nan,
                            "mean_delta_high_snr": float(high_snr_delta.mean()) if len(high_snr_delta) else np.nan,
                            "mann_whitney_delta_low_vs_high_snr_p": float(mann.pvalue) if mann is not None else np.nan,
                            "mann_whitney_delta_low_vs_high_snr_p_bonferroni": float(adjusted.get("mann_whitney_delta_low_vs_high_snr", np.nan)),
                            "spearman_snr_vs_delta_rho": float(snr_spearman.statistic),
                            "spearman_snr_vs_delta_p": float(snr_spearman.pvalue),
                            "spearman_snr_vs_delta_p_bonferroni": float(adjusted["spearman_snr_vs_delta"]),
                            "spearman_zcr_vs_delta_rho": float(zcr_spearman.statistic),
                            "spearman_zcr_vs_delta_p": float(zcr_spearman.pvalue),
                            "spearman_zcr_vs_delta_p_bonferroni": float(adjusted["spearman_zcr_vs_delta"]),
                        }
                    )
                    print("Acoustic delta stats (baseline vs finetuned):", zcr_error_bridge_stats)

    sr_counts = audio_df["sr"].value_counts().rename_axis("sample_rate").reset_index(name="count")
    fig = px.bar(sr_counts, x="sample_rate", y="count", title="Provera sample-rate integriteta")
    style_plotly_figure(fig, x_title="Sample rate [Hz]", y_title="Broj fajlova [count]")
    fig.show()

    non_16k = audio_df[audio_df["sr"] != 16000]
    print("Files with non-16kHz SR:", len(non_16k))

    return {"audio_df": audio_df, "zcr_error_bridge_stats": zcr_error_bridge_stats}
