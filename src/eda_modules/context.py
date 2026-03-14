from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def build_context(root: Optional[Path] = None) -> Dict[str, Any]:
    """Build and print canonical paths used by EDA modules."""
    current_root = root or Path.cwd()
    if current_root.name == "notebooks":
        current_root = current_root.parent

    new_res_dir = current_root / "new_res"
    default_v1_dir = current_root / "res" / "asr_full_run_v1"
    preferred_v1_dir = new_res_dir / "asr_full_run_v1"

    run_v1_dir = preferred_v1_dir if preferred_v1_dir.exists() else default_v1_dir
    run_v2_dir = current_root / "res" / "asr_full_run_v2"

    v1_metrics_dir = run_v1_dir / "metrics"
    legacy_dir = current_root / "logs" / "verification" / "dataset_strategy"

    ctx = {
        "ROOT": current_root,
        "DATA_RAW_DIR": current_root / "data" / "raw",
        "ALIGN_DIRS": [
            current_root / "data" / "aligned_raw_v1",
            current_root / "data" / "aligned_raw_v1_improved",
            current_root / "data" / "aligned_raw_v1_rerun",
        ],
        "RES_DIR": current_root / "res",
        "NEW_RES_DIR": new_res_dir,
        "RUN_V1_DIR": run_v1_dir,
        "RUN_V2_DIR": run_v2_dir,
        "V1_BASELINE_METRICS": v1_metrics_dir / "baseline_holdout_metrics.json",
        "V1_FINETUNED_METRICS": v1_metrics_dir / "finetuned_holdout_metrics.json",
        "V1_BASELINE_PRED": v1_metrics_dir / "baseline_holdout_predictions.jsonl",
        "V1_FINETUNED_PRED": v1_metrics_dir / "finetuned_holdout_predictions.jsonl",
        "V2_BASELINE_METRICS": legacy_dir / "baseline_openai_whisper_base_eval_aligned_test_rerun.json",
        "V2_FINETUNED_METRICS": legacy_dir / "raw_aligned_v2026_run2_rerun_eval_aligned_test.json",
        "V2_BASELINE_PRED": legacy_dir / "baseline_openai_whisper_base_eval_aligned_test_rerun_predictions.jsonl",
        "V2_FINETUNED_PRED": legacy_dir / "raw_aligned_v2026_run2_rerun_eval_aligned_test_predictions.jsonl",
    }

    print(f"ROOT: {ctx['ROOT']}")
    print(f"RAW dir exists: {ctx['DATA_RAW_DIR'].exists()}")
    print(
        f"v1 run exists: {ctx['RUN_V1_DIR'].exists()} | "
        f"v2 run exists: {ctx['RUN_V2_DIR'].exists()}"
    )
    return ctx
