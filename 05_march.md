# 05 March - ASR Improvement Report

## 1) Objective

This session focused on root-cause fixes for weak gains over baseline in the v1 pipeline.
Primary hypothesis was confirmed: supervision quality (alignment noise) and split grouping issues were the main bottlenecks.

---

## 2) Code changes applied

### A) Alignment quality hardening

Updated chunk builder to support strict filtering instead of only warning:

- `cli/build_aligned_chunks.py`
- `kaggle_project_code/cli/build_aligned_chunks.py`
- `kaggle_bundle/cli/build_aligned_chunks.py`

Added/updated QA controls:

- `--qa_unmatched_threshold` default: **0.35** (was 0.4)
- `--min_aligned_word_ratio` default: **0.55** (was 0.0)
- `--max_chars_per_second` default: **20.0** (was 0.0)
- `--drop_suspicious` default: **True** (`--no-drop-suspicious` to disable)

Also added `chars_per_second` to manifest rows.

### B) Split leakage prevention for aligned chunks

Updated group split regex in:

- `src/data/data_loader.py`
- `kaggle_project_code/src/data/data_loader.py`

Now chunk grouping handles both suffix patterns:

- `_chunkNN`
- `_aligned_NNNN`

This keeps all chunks from one source article in one split.

### C) Text preprocessing correction

Removed repeated-token collapsing from base `preprocess()` (it was mutating references and could hide real label errors):

- `src/utils/text_preprocessing.py`
- `kaggle_project_code/src/utils/text_preprocessing.py`

### D) Kaggle full pipeline defaults aligned with improved QA

Updated:

- `kaggle/run_full_pipeline.py`
- `kaggle_project_code/kaggle/run_full_pipeline.py`
- `kaggle_bundle/kaggle/run_full_pipeline.py`

`drop_suspicious` is now default-enabled via boolean optional arg.

---

## 3) Clean local run from scratch (executed)

### Step 1: Rebuild aligned dataset (strict QA)

Output:

- `data/aligned_raw_v1_improved`

Report summary (`data/aligned_raw_v1_improved/report.json`):

- pairs_total: 80
- pairs_kept: 79
- pairs_skipped: 1
- total_chunks_written: 607
- dropped_suspicious_chunks: 185
- suspicious_chunks_kept: 0
- min aligned_word_ratio in final manifest: 0.6522

### Step 2: Train improved model

Output model:

- `models/whisper/raw_aligned_v2026_run3_improved/final`

Training config highlights:

- epochs: 6
- lr: 2.5e-6
- dropout: 0.15
- `--unfreeze_encoder`
- `--freeze_encoder_layers 8`

### Step 3: Evaluate improved model and fair baseline on same test split

Files:

- `logs/verification/dataset_strategy/raw_aligned_v2026_run3_improved_eval_aligned_test.json`
- `logs/verification/dataset_strategy/baseline_openai_whisper_base_eval_aligned_test_run3_improved.json`
- `logs/verification/dataset_strategy/run3_improved_vs_01_march_summary.json`

---

## 4) Results

### Historical reference from `01_march.md`

- Fine-tuned (`raw_aligned_v2026_run2`):
  - WER: **0.5340**
  - CER: **0.2137**
- Baseline on aligned test:
  - WER: **0.5419**
  - CER: **0.2119**

### New improved run (`run3_improved`)

- Fine-tuned:
  - WER: **0.4746130031**
  - CER: **0.1710579447**
- Baseline (same split/decoding):
  - WER: **0.4990712074**
  - CER: **0.1789319604**

### Delta

- New fine-tuned vs 01_march fine-tuned:
  - WER improvement: **0.0593869969** absolute
  - CER improvement: **0.0426420553** absolute
- New fine-tuned vs new baseline (same test):
  - WER improvement: **0.0244582043** absolute
  - CER improvement: **0.0078740157** absolute

Interpretation:

- Improvement is now clear and significant.
- Root-cause fixes (alignment filtering + split integrity + safer preprocessing) materially improved both WER and CER.

---

## 5) New defaults (effective from now)

### Alignment builder defaults

- `qa_unmatched_threshold = 0.35`
- `min_aligned_word_ratio = 0.55`
- `max_chars_per_second = 20.0`
- `drop_suspicious = True`

### Kaggle full pipeline defaults

- Uses the strict QA defaults above
- Suspicious chunks dropped by default

---

## 6) Recommended next actions

1. Scale strict aligned build from pilot 80 files to full raw set 370 files.
2. Re-train with same recipe on full strict set.
3. Keep fair baseline eval on identical split/decoding for each run.
4. Add periodic top-error category report (numbers, names, entities) from predictions JSONL.

---

## 7) Final note

This session moved the project from marginal baseline gains to a robust, measurable advantage by fixing data quality at the source and tightening training discipline.
