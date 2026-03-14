from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
from IPython.display import Markdown, display
from scipy import stats

from .io_utils import read_json, read_jsonl_records
from .stats import bonferroni_correction
from .visualization import style_plotly_figure


def read_metrics_file(path: Path) -> Dict[str, float]:
    """Read standard metric file and coerce fields to float."""
    data = read_json(path)
    if not isinstance(data, dict):
        return {"wer": np.nan, "cer": np.nan, "num_samples": np.nan}
    return {
        "wer": float(data.get("wer", np.nan)),
        "cer": float(data.get("cer", np.nan)),
        "num_samples": float(data.get("num_samples", np.nan)),
    }


def read_predictions_jsonl(path: Path) -> pd.DataFrame:
    """Read predictions JSONL as dataframe."""
    return pd.DataFrame(read_jsonl_records(path))


def _sentence_wer(reference: str, hypothesis: str) -> Tuple[int, int]:
    """Return edit distance and reference length at sentence level."""
    ref = str(reference).split()
    hyp = str(hypothesis).split()
    m, n = len(ref), len(hyp)
    if m == 0:
        return (0 if n == 0 else n), max(1, m)

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[m][n], m


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize prediction schema to sample_id/ref/pred/wer/audio_path."""
    if df.empty:
        return df

    out = df.copy()
    ref_candidates = [c for c in out.columns if c.lower() in ["reference", "ref", "target", "label", "text", "ground_truth"]]
    pred_candidates = [c for c in out.columns if c.lower() in ["prediction", "pred", "hypothesis", "generated_text", "transcription"]]
    wer_candidates = [c for c in out.columns if c.lower() in ["wer", "sample_wer"]]

    if ref_candidates and "ref" not in out.columns:
        out = out.rename(columns={ref_candidates[0]: "ref"})
    if pred_candidates and "pred" not in out.columns:
        out = out.rename(columns={pred_candidates[0]: "pred"})
    if wer_candidates and "wer" not in out.columns:
        out = out.rename(columns={wer_candidates[0]: "wer"})

    if "wer" not in out.columns and {"ref", "pred"}.issubset(out.columns):
        computed_wer: List[float] = []
        for ref_val, pred_val in zip(out["ref"], out["pred"]):
            errors, ref_len = _sentence_wer(str(ref_val), str(pred_val))
            computed_wer.append(errors / max(1, ref_len))
        out["wer"] = computed_wer

    if "audio_path" not in out.columns:
        audio_path_candidates = [c for c in out.columns if c.lower() in ["path", "audio_path"]]
        if audio_path_candidates:
            out = out.rename(columns={audio_path_candidates[0]: "audio_path"})

    if "sample_id" not in out.columns:
        sample_id_candidates = [c for c in out.columns if c.lower() in ["sample_id", "audio_id"]]
        if sample_id_candidates:
            out = out.rename(columns={sample_id_candidates[0]: "sample_id"})

    if "sample_id" not in out.columns:
        raise ValueError("Missing required 'sample_id' column. Row-order comparison is forbidden.")

    out["sample_id"] = out["sample_id"].astype(str).str.strip()
    if out["sample_id"].eq("").any():
        raise ValueError("Empty sample_id values detected. sample_id is required for strict merge.")
    if out["sample_id"].duplicated().any():
        duplicates = int(out["sample_id"].duplicated().sum())
        raise ValueError(f"Duplicate sample_id values detected: {duplicates}")

    if "wer" in out.columns:
        out["wer"] = pd.to_numeric(out["wer"], errors="coerce")

    return out


def run_ablation_analysis(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Run metric aggregation, normalization-only estimate and ablation summary."""
    metric_files = {
        "processed_baseline": ctx["V1_BASELINE_METRICS"],
        "processed_finetuned": ctx["V1_FINETUNED_METRICS"],
        "legacy_v2_baseline": ctx["V2_BASELINE_METRICS"],
        "legacy_v2_finetuned": ctx["V2_FINETUNED_METRICS"],
        "raw_finetuned": ctx["ROOT"] / "logs" / "verification" / "dataset_strategy" / "raw_v2026_run1_eval_raw_test.json",
        "raw_baseline": ctx["ROOT"] / "logs" / "verification" / "dataset_strategy" / "baseline_raw_openai_whisper_base.json",
        "unfiltered_baseline": ctx["ROOT"] / "logs" / "verification" / "dataset_strategy" / "baseline_openai_whisper_base_eval_aligned_test.json",
        "unfiltered_finetuned": ctx["ROOT"] / "logs" / "verification" / "dataset_strategy" / "raw_aligned_v2026_run2_eval_aligned_test.json",
        "aligned_baseline": ctx["ROOT"] / "logs" / "verification" / "dataset_strategy" / "baseline_openai_whisper_base_eval_aligned_test.json",
    }

    rows = []
    for label, path in metric_files.items():
        if not path.exists():
            continue
        metric = read_metrics_file(path)
        rows.append(
            {
                "scenario": label,
                "wer": metric["wer"],
                "cer": metric["cer"],
                "num_samples": metric["num_samples"],
                "file": str(path),
            }
        )

    data_first_metrics = pd.DataFrame(rows)
    display(data_first_metrics)

    if not data_first_metrics.empty:
        fig = px.bar(
            data_first_metrics.sort_values("wer"),
            x="scenario",
            y="wer",
            color="scenario",
            title="WER po scenariju: sirovi i obrađeni podaci",
        )
        style_plotly_figure(fig, x_title="Scenarijo evaluacije", y_title="WER [0-1]", show_target_wer=True, y_is_wer=True)
        fig.show()

    dual_reporting_df = pd.DataFrame()
    if not data_first_metrics.empty:
        by_scenario = data_first_metrics.set_index("scenario")

        def metric_value(key: str, col: str) -> float:
            if key not in by_scenario.index:
                return np.nan
            return float(by_scenario.loc[key, col])

        dual_reporting_df = pd.DataFrame(
            [
                {
                    "model_family": "Baseline",
                    "unfiltered_wer": metric_value("unfiltered_baseline", "wer"),
                    "filtered_wer": metric_value("processed_baseline", "wer"),
                    "unfiltered_num_samples": metric_value("unfiltered_baseline", "num_samples"),
                    "filtered_num_samples": metric_value("processed_baseline", "num_samples"),
                },
                {
                    "model_family": "Finetuned",
                    "unfiltered_wer": metric_value("unfiltered_finetuned", "wer"),
                    "filtered_wer": metric_value("processed_finetuned", "wer"),
                    "unfiltered_num_samples": metric_value("unfiltered_finetuned", "num_samples"),
                    "filtered_num_samples": metric_value("processed_finetuned", "num_samples"),
                },
            ]
        )
        dual_reporting_df["absolute_drop_wer"] = dual_reporting_df["unfiltered_wer"] - dual_reporting_df["filtered_wer"]
        dual_reporting_df["relative_drop_percent"] = (
            100.0 * dual_reporting_df["absolute_drop_wer"] / dual_reporting_df["unfiltered_wer"]
        )
        dual_reporting_df["relative_drop_percent"] = dual_reporting_df["relative_drop_percent"].replace([np.inf, -np.inf], np.nan)

        display(Markdown("Obavezno dualno izveštavanje (unfiltered vs filtered) radi eliminacije cherry-picking sumnje:"))
        display(dual_reporting_df)

    normalization_effect: Dict[str, Any] = {}
    pred_path = (
        ctx["ROOT"] / "logs" / "verification" / "dataset_strategy" / "raw_aligned_v2026_run3_improved_eval_aligned_test_predictions.jsonl"
    )
    if pred_path.exists():
        pred_df = read_predictions_jsonl(pred_path)
        required = {"reference_raw", "prediction_raw", "reference_preprocessed", "prediction_preprocessed"}
        if required.issubset(pred_df.columns):
            raw_err = raw_words = prep_err = prep_words = 0
            for _, row in pred_df.iterrows():
                e_raw, w_raw = _sentence_wer(str(row.get("reference_raw", "")), str(row.get("prediction_raw", "")))
                e_prep, w_prep = _sentence_wer(str(row.get("reference_preprocessed", "")), str(row.get("prediction_preprocessed", "")))
                raw_err += e_raw
                raw_words += w_raw
                prep_err += e_prep
                prep_words += w_prep

            raw_wer = raw_err / max(1, raw_words)
            prep_wer = prep_err / max(1, prep_words)
            normalization_effect = {
                "samples": int(len(pred_df)),
                "raw_text_wer": float(raw_wer),
                "preprocessed_text_wer": float(prep_wer),
                "absolute_drop": float(raw_wer - prep_wer),
                "relative_drop_percent": float(((raw_wer - prep_wer) / raw_wer) * 100 if raw_wer > 0 else np.nan),
            }
            print("Normalization-only effect:", normalization_effect)

    report_files = {
        "aligned_raw_v1": ctx["ROOT"] / "data" / "aligned_raw_v1" / "report.json",
        "aligned_raw_v1_improved": ctx["ROOT"] / "data" / "aligned_raw_v1_improved" / "report.json",
        "aligned_raw_v1_rerun": ctx["ROOT"] / "data" / "aligned_raw_v1_rerun" / "report.json",
    }

    evolution_rows = []
    for dataset_name, report_path in report_files.items():
        if not report_path.exists():
            continue
        report = read_json(report_path)
        if not isinstance(report, dict):
            continue
        evolution_rows.append(
            {
                "dataset": dataset_name,
                "pairs_total": float(report.get("pairs_total", np.nan)),
                "pairs_kept": float(report.get("pairs_kept", np.nan)),
                "pairs_skipped": float(report.get("pairs_skipped", 0)),
                "total_chunks_written": float(report.get("total_chunks_written", np.nan)),
                "dropped_suspicious_chunks": float(report.get("qa", {}).get("dropped_suspicious_chunks", 0)),
                "suspicious_chunks": float(report.get("qa", {}).get("suspicious_chunks", 0)),
                "suspicious_chunk_ratio": float(report.get("qa", {}).get("suspicious_chunk_ratio", np.nan)),
            }
        )

    data_quality_evolution = pd.DataFrame(evolution_rows)
    display(data_quality_evolution)

    quality_cost_summary = pd.DataFrame()
    improved_report_path = ctx["ROOT"] / "data" / "aligned_raw_v1_improved" / "report.json"
    improved_manifest_path = ctx["ROOT"] / "data" / "aligned_raw_v1_improved" / "manifest.json"
    rerun_manifest_path = ctx["ROOT"] / "data" / "aligned_raw_v1_rerun" / "manifest.json"

    if improved_report_path.exists() and improved_manifest_path.exists():
        improved_report = read_json(improved_report_path)
        improved_manifest = read_json(improved_manifest_path)
        if isinstance(improved_report, dict) and isinstance(improved_manifest, list):
            retained_audio_sec = float(sum(float(row.get("duration_sec", 0.0) or 0.0) for row in improved_manifest if isinstance(row, dict)))
            reported_dropped = int(improved_report.get("qa", {}).get("dropped_suspicious_chunks", 0))
            candidate_total = int(improved_report.get("total_chunks_written", 0)) + reported_dropped
            reported_ratio = (reported_dropped / candidate_total) * 100.0 if candidate_total > 0 else np.nan

            reconstructed_drop_count = np.nan
            reconstructed_drop_audio_sec = np.nan
            if rerun_manifest_path.exists():
                rerun_manifest = read_json(rerun_manifest_path)
                if isinstance(rerun_manifest, list):
                    reconstructed_rows = []
                    unmatched_threshold = float(improved_report.get("qa", {}).get("unmatched_threshold", 0.35))
                    max_chunk_seconds = float(improved_report.get("qa", {}).get("max_chunk_seconds", 30.0))
                    min_aligned_word_ratio = float(improved_report.get("qa", {}).get("min_aligned_word_ratio", 0.55))
                    max_chars_per_second = float(improved_report.get("qa", {}).get("max_chars_per_second", 20.0))

                    for row in rerun_manifest:
                        if not isinstance(row, dict):
                            continue
                        duration_sec = float(row.get("duration_sec", 0.0) or 0.0)
                        aligned_word_ratio = float(row.get("aligned_word_ratio", 0.0) or 0.0)
                        chars_per_second = float(row.get("chars_per_second", 0.0) or 0.0)
                        suspicious = (
                            (duration_sec > max_chunk_seconds)
                            or ((1.0 - aligned_word_ratio) > unmatched_threshold)
                            or (min_aligned_word_ratio > 0 and aligned_word_ratio < min_aligned_word_ratio)
                            or (max_chars_per_second > 0 and chars_per_second > max_chars_per_second)
                        )
                        if suspicious:
                            reconstructed_rows.append(row)

                    reconstructed_drop_count = int(len(reconstructed_rows))
                    reconstructed_drop_audio_sec = float(sum(float(row.get("duration_sec", 0.0) or 0.0) for row in reconstructed_rows))

            quality_cost_summary = pd.DataFrame(
                [
                    {
                        "reported_dropped_chunks": reported_dropped,
                        "reported_drop_ratio_percent": reported_ratio,
                        "retained_audio_sec": retained_audio_sec,
                        "retained_audio_hours": retained_audio_sec / 3600.0,
                        "reconstructed_drop_count": reconstructed_drop_count,
                        "reconstructed_drop_audio_sec": reconstructed_drop_audio_sec,
                        "reconstructed_drop_audio_ms": reconstructed_drop_audio_sec * 1000.0 if pd.notna(reconstructed_drop_audio_sec) else np.nan,
                    }
                ]
            )
            display(quality_cost_summary)

    ablation_df = pd.DataFrame()
    improvement_attribution = pd.DataFrame()
    if not data_first_metrics.empty:
        metrics = data_first_metrics.set_index("scenario")
        raw_baseline_wer = float(metrics.loc["raw_baseline", "wer"]) if "raw_baseline" in metrics.index else np.nan
        aligned_baseline_wer = float(metrics.loc["aligned_baseline", "wer"]) if "aligned_baseline" in metrics.index else np.nan
        processed_baseline_wer = float(metrics.loc["processed_baseline", "wer"]) if "processed_baseline" in metrics.index else np.nan

        norm_only_wer = np.nan
        rel_drop = normalization_effect.get("relative_drop_percent", np.nan)
        if not np.isnan(raw_baseline_wer) and pd.notna(rel_drop):
            norm_only_wer = raw_baseline_wer * (1.0 - rel_drop / 100.0)

        ablation_df = pd.DataFrame(
            [
                {"step_order": 1, "step": "Raw Data", "wer": raw_baseline_wer, "evidence_type": "measured"},
                {"step_order": 2, "step": "Text Normalization Only", "wer": norm_only_wer, "evidence_type": "estimated_from_same-prediction_delta"},
                {"step_order": 3, "step": "Text Norm + Audio Alignment", "wer": aligned_baseline_wer, "evidence_type": "measured"},
                {"step_order": 4, "step": "Text Norm + Alignment + CPS Filtering", "wer": processed_baseline_wer, "evidence_type": "measured"},
            ]
        ).sort_values("step_order")
        display(ablation_df)

        if {"raw_baseline", "raw_finetuned", "processed_baseline", "processed_finetuned"}.issubset(metrics.index):
            raw_b = float(metrics.loc["raw_baseline", "wer"])
            raw_ft = float(metrics.loc["raw_finetuned", "wer"])
            proc_b = float(metrics.loc["processed_baseline", "wer"])
            proc_ft = float(metrics.loc["processed_finetuned", "wer"])
            total_gain = raw_b - proc_ft
            preprocessing_gain = raw_b - proc_b
            finetuning_gain_processed = proc_b - proc_ft

            improvement_attribution = pd.DataFrame(
                [
                    {
                        "component": "Baseline vs Finetuned (Raw)",
                        "start_wer": raw_b,
                        "end_wer": raw_ft,
                        "abs_delta": raw_b - raw_ft,
                        "rel_delta_percent": ((raw_b - raw_ft) / raw_b) * 100.0 if raw_b > 0 else np.nan,
                        "share_of_total_gain_percent": ((raw_b - raw_ft) / total_gain) * 100.0 if total_gain != 0 else np.nan,
                    },
                    {
                        "component": "Baseline vs Finetuned (Processed)",
                        "start_wer": proc_b,
                        "end_wer": proc_ft,
                        "abs_delta": finetuning_gain_processed,
                        "rel_delta_percent": ((proc_b - proc_ft) / proc_b) * 100.0 if proc_b > 0 else np.nan,
                        "share_of_total_gain_percent": (finetuning_gain_processed / total_gain) * 100.0 if total_gain != 0 else np.nan,
                    },
                    {
                        "component": "Poboljšanje pripisano isključivo preprocessingu",
                        "start_wer": raw_b,
                        "end_wer": proc_b,
                        "abs_delta": preprocessing_gain,
                        "rel_delta_percent": (preprocessing_gain / raw_b) * 100.0 if raw_b > 0 else np.nan,
                        "share_of_total_gain_percent": (preprocessing_gain / total_gain) * 100.0 if total_gain != 0 else np.nan,
                    },
                ]
            )
            display(improvement_attribution)

    return {
        "data_first_metrics": data_first_metrics,
        "dual_reporting_df": dual_reporting_df,
        "normalization_effect": normalization_effect,
        "data_quality_evolution": data_quality_evolution,
        "quality_cost_summary": quality_cost_summary,
        "ablation_df": ablation_df,
        "improvement_attribution": improvement_attribution,
    }


def run_fair_comparison(ctx: Dict[str, Any], quality_df: pd.DataFrame) -> Dict[str, Any]:
    """Build fair paired comparison dataframe across v1/v2 predictions."""
    preferred_pred_files = {
        "v1_baseline": ctx["V1_BASELINE_PRED"],
        "v1_finetuned": ctx["V1_FINETUNED_PRED"],
        "v2_baseline": ctx["V2_BASELINE_PRED"],
        "v2_finetuned": ctx["V2_FINETUNED_PRED"],
    }

    pred_map: Dict[str, pd.DataFrame] = {}
    for key, path in preferred_pred_files.items():
        if path.exists():
            pred_map[key] = normalize_cols(read_predictions_jsonl(path))

    if not {"v1_baseline", "v1_finetuned", "v2_finetuned"}.issubset(pred_map.keys()):
        pred_files = [p for p in ctx["ROOT"].rglob("*.jsonl") if any(k in p.name.lower() for k in ["pred", "holdout"])]
        for path in pred_files:
            lower = str(path).lower()
            if "baseline_holdout_predictions" in lower:
                key = "v1_baseline"
            elif "finetuned_holdout_predictions" in lower:
                key = "v1_finetuned"
            elif "run2_rerun" in lower and "raw_aligned" in lower:
                key = "v2_finetuned"
            elif "run2_rerun" in lower and "baseline" in lower:
                key = "v2_baseline"
            else:
                key = "other"
            if key in pred_map:
                continue
            try:
                pred_map[key] = normalize_cols(read_predictions_jsonl(path))
            except Exception:
                continue

    eval_alignment_df = pd.DataFrame()
    v1_paired_df = pd.DataFrame()

    if "v1_baseline" in pred_map and "v1_finetuned" in pred_map:
        baseline_df = pred_map["v1_baseline"]
        finetuned_df = pred_map["v1_finetuned"]
        needed = {"sample_id", "wer"}
        if needed.issubset(baseline_df.columns) and needed.issubset(finetuned_df.columns):
            v1_paired_df = pd.merge(
                baseline_df[["sample_id", "wer"]].rename(columns={"wer": "wer_baseline_v1"}),
                finetuned_df[["sample_id", "wer"]].rename(columns={"wer": "wer_finetuned_v1"}),
                on="sample_id",
                how="inner",
            )
            if len(v1_paired_df) > 0:
                v1_paired_df["wer_delta_baseline_minus_finetuned"] = (
                    v1_paired_df["wer_baseline_v1"] - v1_paired_df["wer_finetuned_v1"]
                )
                fig = px.box(
                    v1_paired_df.melt(
                        id_vars=["sample_id"],
                        value_vars=["wer_baseline_v1", "wer_finetuned_v1"],
                        var_name="model_variant",
                        value_name="wer",
                    ),
                    x="model_variant",
                    y="wer",
                    points="all",
                    color="model_variant",
                    title="Fiksni holdout: baseline vs finetuned (strict sample_id)"
                )
                style_plotly_figure(fig, x_title="Model", y_title="WER [0-1]", show_target_wer=True, y_is_wer=True)
                fig.show()

    if "v1_finetuned" in pred_map and "v2_finetuned" in pred_map:
        p1 = pred_map["v1_finetuned"]
        p2 = pred_map["v2_finetuned"]
        needed = {"sample_id", "wer"}
        if needed.issubset(p1.columns) and needed.issubset(p2.columns):
            common = pd.merge(
                p1[["sample_id", "wer"]].rename(columns={"wer": "wer_v1"}),
                p2[["sample_id", "wer"]].rename(columns={"wer": "wer_v2"}),
                on="sample_id",
                how="inner",
            )
            print("Common samples found:", len(common))
            if len(common) > 0:
                fig = px.scatter(common, x="wer_v1", y="wer_v2", trendline="ols", title="Pošteno poređenje: WER v1 vs WER v2")
                fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="red", dash="dash"))
                style_plotly_figure(fig, x_title="WER v1 [0-1]", y_title="WER v2 [0-1]", show_target_wer=True, y_is_wer=True)
                fig.show()

                if not quality_df.empty and {"sample_id", "aligned_word_ratio"}.issubset(quality_df.columns):
                    awr_by_sample = quality_df[["sample_id", "aligned_word_ratio"]].dropna().drop_duplicates("sample_id")
                    eval_alignment_df = common.merge(awr_by_sample, on="sample_id", how="left")
                else:
                    eval_alignment_df = common
                if not v1_paired_df.empty:
                    eval_alignment_df = eval_alignment_df.merge(
                        v1_paired_df[["sample_id", "wer_baseline_v1", "wer_finetuned_v1", "wer_delta_baseline_minus_finetuned"]],
                        on="sample_id",
                        how="left",
                    )
                display(eval_alignment_df.head())

    return {"pred_map": pred_map, "eval_alignment_df": eval_alignment_df, "v1_paired_df": v1_paired_df}


def run_statistical_tests(eval_alignment_df: pd.DataFrame, v1_paired_df: pd.DataFrame | None = None) -> None:
    """Run robust normality and paired tests for v1/v2 comparison."""
    if eval_alignment_df.empty and (v1_paired_df is None or v1_paired_df.empty):
        return

    normality_rejected = False
    p_values: Dict[str, float] = {}
    for run_col in ["wer_v1", "wer_v2"]:
        if run_col not in eval_alignment_df.columns:
            continue
        series = eval_alignment_df[run_col].dropna()
        if len(series) > 3:
            sample = series.sample(min(5000, len(series)), random_state=42)
            sh_w, sh_p = stats.shapiro(sample)
            normality_rejected = normality_rejected or (sh_p < 0.05)
            p_values[f"shapiro_{run_col}"] = float(sh_p)
            print(f"Shapiro-Wilk {run_col}: W={sh_w:.4f}, p={sh_p:.4e}")

    if normality_rejected:
        display(Markdown("Distribucije WER nisu normalne, poređenje verzija je vođeno neparametrijskim testovima."))

    corr_df = eval_alignment_df.dropna(subset=["aligned_word_ratio", "wer_v2"]) if {"aligned_word_ratio", "wer_v2"}.issubset(eval_alignment_df.columns) else pd.DataFrame()
    if len(corr_df) > 3:
        pearson_r, pearson_p = stats.pearsonr(corr_df["aligned_word_ratio"], corr_df["wer_v2"])
        spearman_r, spearman_p = stats.spearmanr(corr_df["aligned_word_ratio"], corr_df["wer_v2"], nan_policy="omit")
        p_values["pearson_aligned_vs_wer_v2"] = float(pearson_p)
        p_values["spearman_aligned_vs_wer_v2"] = float(spearman_p)
        print(f"Pearson(aligned_word_ratio, WER_v2): r={pearson_r:.4f}, p={pearson_p:.4e}")
        print(f"Spearman(aligned_word_ratio, WER_v2): rho={spearman_r:.4f}, p={spearman_p:.4e}")

    paired = eval_alignment_df.dropna(subset=["wer_v1", "wer_v2"]) if {"wer_v1", "wer_v2"}.issubset(eval_alignment_df.columns) else pd.DataFrame()
    if len(paired) > 3:
        diff = paired["wer_v1"] - paired["wer_v2"]
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(paired["wer_v1"], paired["wer_v2"], alternative="two-sided")
        p_values["wilcoxon_wer_v1_vs_v2"] = float(wilcoxon_p)
        print(f"Wilcoxon(wer_v1 vs wer_v2): W={wilcoxon_stat:.4f}, p={wilcoxon_p:.4e}")

        std_diff = float(np.nanstd(diff, ddof=1)) if len(diff) > 1 else np.nan
        cohen_dz = float(np.nanmean(diff) / std_diff) if pd.notna(std_diff) and std_diff > 0 else np.nan
        effect_size_df = pd.DataFrame(
            [
                {
                    "comparison": "wer_v1_vs_wer_v2",
                    "n": int(len(diff)),
                    "median_delta_wer": float(np.nanmedian(diff)),
                    "mean_delta_wer": float(np.nanmean(diff)),
                    "cohen_dz": cohen_dz,
                }
            ]
        )
        display(Markdown("Effect size za upareno poređenje (pored p-vrednosti):"))
        display(effect_size_df)

    paired_v1 = v1_paired_df if v1_paired_df is not None else pd.DataFrame()
    if len(paired_v1) > 3 and {"wer_baseline_v1", "wer_finetuned_v1"}.issubset(paired_v1.columns):
        diff_v1 = paired_v1["wer_baseline_v1"] - paired_v1["wer_finetuned_v1"]
        wilcoxon_stat_v1, wilcoxon_p_v1 = stats.wilcoxon(
            paired_v1["wer_baseline_v1"],
            paired_v1["wer_finetuned_v1"],
            alternative="greater",
        )
        p_values["wilcoxon_v1_baseline_vs_finetuned"] = float(wilcoxon_p_v1)
        print(
            "Wilcoxon(v1 baseline vs finetuned, alternative=baseline>finetuned): "
            f"W={wilcoxon_stat_v1:.4f}, p={wilcoxon_p_v1:.4e}"
        )

        std_diff_v1 = float(np.nanstd(diff_v1, ddof=1)) if len(diff_v1) > 1 else np.nan
        cohen_dz_v1 = float(np.nanmean(diff_v1) / std_diff_v1) if pd.notna(std_diff_v1) and std_diff_v1 > 0 else np.nan
        effect_size_v1_df = pd.DataFrame(
            [
                {
                    "comparison": "v1_baseline_vs_finetuned",
                    "n": int(len(diff_v1)),
                    "median_delta_wer": float(np.nanmedian(diff_v1)),
                    "mean_delta_wer": float(np.nanmean(diff_v1)),
                    "cohen_dz": cohen_dz_v1,
                }
            ]
        )
        display(Markdown("Effect size za ključni test poboljšanja (v1 baseline vs finetuned):"))
        display(effect_size_v1_df)

    if p_values:
        adjusted = bonferroni_correction(p_values)
        bonf_df = pd.DataFrame(
            {
                "test": list(p_values.keys()),
                "p_raw": [p_values[key] for key in p_values],
                "p_bonferroni": [adjusted[key] for key in p_values],
            }
        )
        bonf_df["significant_005_after_bonferroni"] = bonf_df["p_bonferroni"] < 0.05
        display(Markdown("Bonferroni korekcija za višestruka poređenja:"))
        display(bonf_df)


def run_oov_analysis(pred_map: Dict[str, pd.DataFrame]) -> None:
    """Estimate missing-reference-token reduction from v1 to v2."""
    if "v1" not in pred_map or "v2" not in pred_map:
        return

    p1 = pred_map["v1"]
    p2 = pred_map["v2"]
    needed = {"sample_id", "ref", "pred"}
    if not needed.issubset(p1.columns) or not needed.issubset(p2.columns):
        return

    joined = pd.merge(
        p1[["sample_id", "ref", "pred"]].rename(columns={"pred": "pred_v1"}),
        p2[["sample_id", "pred"]].rename(columns={"pred": "pred_v2"}),
        on="sample_id",
        how="inner",
    )

    def token_set(value: str) -> set[str]:
        return set(re.findall(r"\w+", (value or "").lower(), flags=re.UNICODE))

    oov_v1: Counter[str] = Counter()
    oov_v2: Counter[str] = Counter()

    for _, row in joined.iterrows():
        ref_tokens = token_set(str(row["ref"]))
        v1_tokens = token_set(str(row["pred_v1"]))
        v2_tokens = token_set(str(row["pred_v2"]))
        oov_v1.update(ref_tokens - v1_tokens)
        oov_v2.update(ref_tokens - v2_tokens)

    top_v1 = pd.DataFrame(oov_v1.most_common(30), columns=["token", "missing_count_v1"])
    top_v2 = pd.DataFrame(oov_v2.most_common(30), columns=["token", "missing_count_v2"])
    learned = top_v1.merge(top_v2, on="token", how="left").fillna(0)
    learned["gain_v2"] = learned["missing_count_v1"] - learned["missing_count_v2"]
    display(learned.sort_values("gain_v2", ascending=False).head(25))


def run_error_breakdown(pred_map: Dict[str, pd.DataFrame]) -> None:
    """Compute substitution/insertion/deletion pie chart from v2 predictions."""
    def tokenize(value: str) -> List[str]:
        return re.findall(r"\w+", (value or "").lower(), flags=re.UNICODE)

    def edit_breakdown(ref: List[str], hyp: List[str]) -> Dict[str, int]:
        n, m = len(ref), len(hyp)
        dp = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            dist, sub, ins, dele = dp[i - 1][0]
            dp[i][0] = (dist + 1, sub, ins, dele + 1)
        for j in range(1, m + 1):
            dist, sub, ins, dele = dp[0][j - 1]
            dp[0][j] = (dist + 1, sub, ins + 1, dele)

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref[i - 1] == hyp[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    sub = dp[i - 1][j - 1]
                    ins = dp[i][j - 1]
                    dele = dp[i - 1][j]
                    cands = [
                        (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                        (ins[0] + 1, ins[1], ins[2] + 1, ins[3]),
                        (dele[0] + 1, dele[1], dele[2], dele[3] + 1),
                    ]
                    dp[i][j] = min(cands, key=lambda x: x[0])

        _, sub, ins, dele = dp[n][m]
        return {"substitutions": sub, "insertions": ins, "deletions": dele}

    if "v2" in pred_map and {"ref", "pred"}.issubset(pred_map["v2"].columns):
        sample_df = pred_map["v2"].dropna(subset=["ref", "pred"]).head(500)
        agg: Counter[str] = Counter()
        for _, row in sample_df.iterrows():
            agg.update(edit_breakdown(tokenize(str(row["ref"])), tokenize(str(row["pred"]))))
        err_df = pd.DataFrame({"type": list(agg.keys()), "count": list(agg.values())})
        if not err_df.empty:
            fig = px.pie(err_df, names="type", values="count", title="Error categorization (Sub/Del/Ins)")
            fig.show()


def run_character_levenshtein(pred_map: Dict[str, pd.DataFrame]) -> None:
    """Analyze normalized character-Levenshtein profile and Serbian grapheme confusions."""
    def levenshtein_chars(a: str, b: str) -> int:
        n, m = len(a), len(b)
        if n == 0:
            return m
        if m == 0:
            return n
        dp = np.zeros((n + 1, m + 1), dtype=int)
        dp[:, 0] = np.arange(n + 1)
        dp[0, :] = np.arange(m + 1)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
        return int(dp[n, m])

    if "v2" in pred_map and {"ref", "pred"}.issubset(pred_map["v2"].columns):
        pairs = [("č", "ć"), ("ć", "č"), ("š", "ž"), ("ž", "š"), ("đ", "dj")]
        rows = []
        confusion: Counter[str] = Counter()

        for _, row in pred_map["v2"].dropna(subset=["ref", "pred"]).head(1200).iterrows():
            ref = str(row["ref"]).lower()
            pred = str(row["pred"]).lower()
            dist = levenshtein_chars(ref, pred)
            rows.append({"sample_id": str(row.get("sample_id", "")), "char_lev": dist, "norm_char_lev": dist / max(len(ref), 1)})
            for src, dst in pairs:
                if src in ref and dst in pred:
                    confusion[f"{src}->{dst}"] += 1

        lev_df = pd.DataFrame(rows)
        if not lev_df.empty:
            display(lev_df.describe().T)
        if confusion:
            conf_df = pd.DataFrame(confusion.items(), columns=["confusion", "count"]).sort_values("count", ascending=False)
            display(conf_df)
