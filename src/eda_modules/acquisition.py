from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
from IPython.display import display

from .io_utils import list_audio_files, read_json
from .visualization import style_plotly_figure


def infer_source(stem: str) -> str:
    """Infer coarse source category from filename stem."""
    lower = stem.lower()
    if any(key in lower for key in ["podkast", "podcast"]):
        return "podkast"
    if any(key in lower for key in ["dnevnik", "jutarnji", "vesti"]):
        return "dnevnik"
    if any(key in lower for key in ["reportaza", "reporta", "emisija", "dokumentarne"]):
        return "reportaže"
    return "ostalo"


def parse_date_from_name(name: str) -> Optional[datetime]:
    """Extract date from filename in YYYY-MM-DD or DD-MM-YYYY styles."""
    patterns = [
        r"(20\d{2})[-_](\d{2})[-_](\d{2})",
        r"(\d{2})[-_](\d{2})[-_](20\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if not match:
            continue
        groups = match.groups()
        try:
            if len(groups[0]) == 4:
                return datetime(int(groups[0]), int(groups[1]), int(groups[2]))
            return datetime(int(groups[2]), int(groups[1]), int(groups[0]))
        except ValueError:
            continue
    return None


def run_data_acquisition(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate raw-data inventory and alignment volume metrics."""
    raw_records: List[Dict[str, Any]] = []
    for audio_path in list_audio_files(ctx["DATA_RAW_DIR"]):
        parsed_date = parse_date_from_name(audio_path.stem)
        fs_date = datetime.fromtimestamp(audio_path.stat().st_mtime)
        raw_records.append(
            {
                "path": str(audio_path),
                "stem": audio_path.stem,
                "suffix": audio_path.suffix.lower(),
                "source": infer_source(audio_path.stem),
                "parsed_date": parsed_date,
                "effective_date": parsed_date if parsed_date is not None else fs_date,
            }
        )

    raw_df = pd.DataFrame(raw_records)

    alignment_stats: List[Dict[str, Any]] = []
    for align_dir in ctx["ALIGN_DIRS"]:
        manifest_path = align_dir / "manifest.json"
        report_path = align_dir / "report.json"

        chunk_count = 0
        suspicious_count = np.nan

        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if isinstance(manifest, list):
                chunk_count = len(manifest)
            elif isinstance(manifest, dict):
                chunk_count = len(manifest.get("items", []))

        if report_path.exists():
            report = read_json(report_path)
            if isinstance(report, dict):
                suspicious_count = report.get("suspicious_count", report.get("num_suspicious", np.nan))

        alignment_stats.append(
            {
                "dataset": align_dir.name,
                "chunk_count": chunk_count,
                "suspicious_count": suspicious_count,
            }
        )

    alignment_df = pd.DataFrame(alignment_stats)
    if not alignment_df.empty:
        alignment_df["suspicious_count"] = alignment_df["suspicious_count"].fillna(0).astype(int)
    raw_total = len(raw_df)
    best_aligned = int(alignment_df["chunk_count"].max()) if not alignment_df.empty else 0

    display(alignment_df)

    return {
        "raw_df": raw_df,
        "alignment_df": alignment_df,
        "raw_total": raw_total,
        "best_aligned": best_aligned,
    }


def plot_data_acquisition(raw_df: pd.DataFrame, raw_total: int, best_aligned: int) -> None:
    """Render source, timeline and raw-vs-aligned volume charts."""
    if raw_df.empty:
        return

    source_counts = raw_df["source"].value_counts().rename_axis("source").reset_index(name="count")
    fig = px.bar(source_counts, x="source", y="count", color="source", title="Broj fajlova po izvoru")
    style_plotly_figure(fig, x_title="Tip izvora", y_title="Broj fajlova [count]")
    fig.show()

    timeline_df = raw_df.copy()
    timeline_df["month"] = timeline_df["effective_date"].dt.to_period("M").astype(str)
    timeline_counts = timeline_df.groupby("month", as_index=False).size().rename(columns={"size": "count"})
    fig = px.line(
        timeline_counts,
        x="month",
        y="count",
        markers=True,
        title="Vremenska distribucija prikupljenih vesti",
    )
    style_plotly_figure(fig, x_title="Mesec", y_title="Broj fajlova [count]")
    fig.show()

    raw_vs_aligned = pd.DataFrame(
        {
            "stage": ["raw_audio", "aligned_chunks_best"],
            "count": [raw_total, best_aligned],
        }
    )
    fig = px.bar(
        raw_vs_aligned,
        x="stage",
        y="count",
        color="stage",
        title="Raw i aligned volumen podataka",
    )
    style_plotly_figure(fig, x_title="Faza pipeline-a", y_title="Broj jedinica [count]")
    fig.show()
