from __future__ import annotations

import re
from textwrap import dedent
from typing import Any, Dict

import numpy as np
import pandas as pd
from IPython.display import display


def build_results_digest(
    data_first_metrics: pd.DataFrame,
    ablation_df: pd.DataFrame,
    data_quality_evolution: pd.DataFrame,
    quality_cost_summary: pd.DataFrame,
    normalization_effect: Dict[str, Any],
    zcr_error_bridge_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Build final digest dictionary, summary table and LaTeX snippet."""
    digest: Dict[str, Any] = {}

    if not data_first_metrics.empty:
        metrics = data_first_metrics.set_index("scenario")
        if {"processed_baseline", "processed_finetuned"}.issubset(metrics.index):
            proc_b = float(metrics.loc["processed_baseline", "wer"])
            proc_ft = float(metrics.loc["processed_finetuned", "wer"])
            digest["processed_baseline_wer"] = proc_b
            digest["processed_finetuned_wer"] = proc_ft
            digest["processed_rel_improvement_percent"] = ((proc_b - proc_ft) / proc_b * 100.0) if proc_b > 0 else np.nan

    if not ablation_df.empty:
        for _, row in ablation_df.iterrows():
            key = re.sub(r"[^a-z0-9]+", "_", str(row["step"]).lower()).strip("_")
            digest[f"ablation_{key}_wer"] = float(row["wer"]) if pd.notna(row["wer"]) else np.nan

    if not data_quality_evolution.empty:
        quality_indexed = data_quality_evolution.set_index("dataset")
        if "aligned_raw_v1_improved" in quality_indexed.index:
            digest["improved_dropped_suspicious_chunks"] = float(quality_indexed.loc["aligned_raw_v1_improved", "dropped_suspicious_chunks"])

    if not quality_cost_summary.empty:
        first = quality_cost_summary.iloc[0]
        for key in quality_cost_summary.columns:
            digest[key] = float(first[key]) if pd.notna(first[key]) else np.nan

    if normalization_effect:
        digest.update(normalization_effect)

    if zcr_error_bridge_stats:
        for key, value in zcr_error_bridge_stats.items():
            digest[f"zcr_bridge_{key}"] = float(value) if pd.notna(value) else np.nan

    summary_rows = [
        {
            "scenario": "Processed Baseline -> Processed Finetuned",
            "start_wer": digest.get("processed_baseline_wer", np.nan),
            "end_wer": digest.get("processed_finetuned_wer", np.nan),
            "rel_delta_percent": digest.get("processed_rel_improvement_percent", np.nan),
        }
    ]
    summary_table_df = pd.DataFrame(summary_rows)
    display(summary_table_df)

    latex_rows = "\n".join(
        f"{row['scenario']} & {row['start_wer']:.4f} & {row['end_wer']:.4f} & {row['rel_delta_percent']:.2f}\\% \\\\" 
        for _, row in summary_table_df.iterrows()
        if pd.notna(row["start_wer"]) and pd.notna(row["end_wer"]) and pd.notna(row["rel_delta_percent"])
    )

    latex_table = dedent(
        f"""
        \\begin{{table*}}[t]
        \\centering
        \\begin{{tabular}}{{l c c c}}
        \\hline
        Scenario & Start WER & End WER & Rel. poboljsanje \\\\
        \\hline
        {latex_rows}
        \\hline
        \\end{{tabular}}
        \\caption{{Sazetak poboljsanja na velikom Kaggle holdout skupu (new_res).}}
        \\end{{table*}}
        """
    )

    return {"digest": digest, "summary_table_df": summary_table_df, "latex_table": latex_table}
