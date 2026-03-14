from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display
from scipy import stats

from .io_utils import read_json
from .visualization import style_plotly_figure


def parse_manifest_items(manifest_path: Path) -> List[dict]:
    """Parse manifest in either list or dict-with-items shape."""
    data = read_json(manifest_path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "items" in data:
        items = data.get("items", [])
        return items if isinstance(items, list) else []
    return []


def get_first(item: dict, keys: List[str], default: Any = np.nan) -> Any:
    """Return first non-null item value from candidate keys."""
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return default


def run_quality_analysis(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze alignment quality and CPS outliers with robust statistics."""
    quality_rows: List[Dict[str, Any]] = []
    quality_sources = ctx["ALIGN_DIRS"] + [
        ctx["RUN_V1_DIR"] / "aligned_holdout",
        ctx["RUN_V2_DIR"] / "aligned_holdout",
    ]
    for align_dir in quality_sources:
        manifest_path = align_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        for item in parse_manifest_items(manifest_path):
            text = str(get_first(item, ["text", "transcript", "label", "chunk_text"], default=""))
            duration = float(get_first(item, ["duration", "duration_s", "duration_sec", "audio_duration"], default=np.nan))
            awr = float(get_first(item, ["aligned_word_ratio", "alignment_word_ratio"], default=np.nan))
            sample_id = str(get_first(item, ["id", "sample_id", "chunk_id", "audio_path", "path", "chunk_audio"], default=""))
            n_chars = len(text)
            n_words = len(re.findall(r"\w+", text, flags=re.UNICODE))
            cps = (n_chars / duration) if pd.notna(duration) and duration > 0 else np.nan
            quality_rows.append(
                {
                    "dataset": align_dir.name,
                    "sample_id": Path(sample_id).stem if sample_id else "",
                    "duration_s": duration,
                    "n_chars": n_chars,
                    "n_words": n_words,
                    "cps": cps,
                    "aligned_word_ratio": awr,
                }
            )

    quality_df = pd.DataFrame(quality_rows)
    quality_df = quality_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["duration_s"])

    if quality_df.empty:
        return {"quality_df": quality_df}

    quality_df["is_cps_outlier"] = (quality_df["cps"] > 18) | (quality_df["cps"] < 2)
    quality_df["is_awr_below_060"] = quality_df["aligned_word_ratio"] < 0.60

    # Toxic sample: likely corrupted supervision (alignment/CPS failure) and should be excluded from training/eval.
    quality_df["is_toxic_sample"] = (
        quality_df["is_awr_below_060"]
        | quality_df["is_cps_outlier"]
        | (quality_df["n_words"] <= 0)
        | (quality_df["duration_s"] <= 0)
    )

    # Hard sample: challenging but valid supervision signal; keep for realistic difficulty estimation.
    hard_cps = quality_df["cps"].between(16.0, 18.0, inclusive="both") | quality_df["cps"].between(2.0, 3.0, inclusive="both")
    hard_alignment = quality_df["aligned_word_ratio"].between(0.60, 0.75, inclusive="left")
    word_heavy = quality_df["n_words"] >= quality_df["n_words"].quantile(0.90)
    quality_df["is_hard_sample"] = (~quality_df["is_toxic_sample"]) & (hard_cps | hard_alignment | word_heavy)

    display(
        Markdown(
            "Definicije QA statusa: "
            "toksičan uzorak = strukturno nevalidan supervision signal (AWR < 0.60, CPS van [2, 18], prazan tekst ili nevalidno trajanje); "
            "težak uzorak = validan ali izazovan signal (granični AWR/CPS ili visok leksički obim)."
        )
    )

    qa_status = pd.DataFrame(
        [
            {
                "status": "toxic",
                "count": int(quality_df["is_toxic_sample"].sum()),
            },
            {
                "status": "hard_non_toxic",
                "count": int(quality_df["is_hard_sample"].sum()),
            },
            {
                "status": "regular",
                "count": int((~quality_df["is_toxic_sample"] & ~quality_df["is_hard_sample"]).sum()),
            },
        ]
    )
    qa_status["share_percent"] = 100.0 * qa_status["count"] / max(1, len(quality_df))
    display(qa_status)

    cps_clean = quality_df["cps"].dropna()
    if not cps_clean.empty:
        plt.figure(figsize=(14, 6))
        sns.histplot(cps_clean, bins=60, kde=True, color="#1F77B4")
        plt.axvline(2, color="orange", linestyle="--", label="CPS = 2")
        plt.axvline(18, color="red", linestyle="--", label="CPS = 18")
        plt.title("CPS distribucija (Characters Per Second)")
        plt.legend()
        plt.show()

        sample = cps_clean.sample(min(5000, len(cps_clean)), random_state=42) if len(cps_clean) > 3 else cps_clean
        if len(sample) > 3:
            sh_stat, sh_p = stats.shapiro(sample)
            if sh_p < 0.05:
                display(Markdown("Distribucija CPS nije normalna, pa je Spearman korelacija metodološki opravdana."))

    fig = px.box(
        quality_df.dropna(subset=["aligned_word_ratio"]),
        x="dataset",
        y="aligned_word_ratio",
        points="all",
        title="Distribucija aligned_word_ratio po dataset-u",
    )
    fig.add_hline(y=0.60, line_dash="dash", line_color="red")
    style_plotly_figure(fig, x_title="Dataset", y_title="Aligned word ratio [0-1]")
    fig.show()

    corr_cols = ["duration_s", "n_words", "n_chars", "cps", "aligned_word_ratio"]
    corr_df = quality_df[corr_cols].dropna()
    if not corr_df.empty:
        corr = corr_df.corr(method="spearman", numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
        plt.title("Spearman korelacije između trajanja, obima teksta i alignment metrika")
        plt.show()

    return {"quality_df": quality_df}
