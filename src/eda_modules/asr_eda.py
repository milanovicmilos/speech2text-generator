from __future__ import annotations

from .acoustics import run_audio_analysis
from .acquisition import plot_data_acquisition, run_data_acquisition
from .constants import COLORWAY, TARGET_WER, configure_plotting
from .context import build_context
from .evaluation import (
    normalize_cols,
    read_metrics_file,
    read_predictions_jsonl,
    run_ablation_analysis,
    run_character_levenshtein,
    run_error_breakdown,
    run_fair_comparison,
    run_oov_analysis,
    run_statistical_tests,
)
from .linguistics import run_text_analysis
from .quality import run_quality_analysis
from .reporting import build_results_digest
from .taxonomy import run_taxonomy_analysis
from .visualization import style_plotly_figure

configure_plotting()

__all__ = [
    "TARGET_WER",
    "COLORWAY",
    "style_plotly_figure",
    "build_context",
    "run_data_acquisition",
    "plot_data_acquisition",
    "run_audio_analysis",
    "run_text_analysis",
    "run_quality_analysis",
    "run_ablation_analysis",
    "read_metrics_file",
    "read_predictions_jsonl",
    "normalize_cols",
    "run_fair_comparison",
    "run_statistical_tests",
    "run_oov_analysis",
    "run_error_breakdown",
    "run_character_levenshtein",
    "run_taxonomy_analysis",
    "build_results_digest",
]
