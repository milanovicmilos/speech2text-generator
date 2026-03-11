## NEXT TODO — Project health checklist

This file converts the analysis report into a prioritized, actionable TODO checklist. Each item includes: what to do, why it matters, and references to the code or artifacts.

---

### High Priority (must fix before large retrain)

- [ ] Align training and eval preprocessing: apply the same text normalization in training metrics and evaluation
  - What: Ensure `compute_metrics` used by Trainer and `cli/eval_model.py` use the same `SerbianTextPreprocessor().preprocess()` for both predictions and references. Update `cli/train.py` -> `compute_metrics` to preprocess `pred_str` and `label_str` before computing WER/CER.
  - Why: Trainer's `load_best_model_at_end=True` selects checkpoints using its metrics. If preprocessing differs, the checkpoint chosen may not be best under final eval normalization, invalidating comparisons.
  - Files: `cli/train.py`, `cli/eval_model.py`, `src/utils/text_preprocessing.py`

- [ ] Stop or gate auto-population of chunk text in loader
  - What: Disable or require explicit opt-in for the `auto-populate` logic in `create_dataloaders` that writes chunked `.txt` by evenly splitting raw transcripts when `text/` is empty. Replace with an explicit flag `allow_auto_populate` (default `False`) or fail loudly and list missing transcripts.
  - Why: Even-word splitting does not use timestamps — it produces incorrectly aligned supervision which harms model training and masks data-quality problems.
  - Files: `src/data/data_loader.py`

- [ ] Make DataLoader deterministic across `num_workers` > 0
  - What: Add `worker_init_fn` and/or use `torch.Generator` seed for DataLoader shuffling. Seed numpy per worker (`np.random.seed(seed + worker_id)`) and propagate deterministic behavior for augmentation in `__getitem__`.
  - Why: Current augmentation uses `np.random` and DataLoader shuffle uses default RNG; with multiple workers this can cause non-reproducible runs even with `set_seed`.
  - Files: `src/data/data_loader.py`, `src/utils/config.py`

### Medium Priority (training & evaluation consistency)

- [ ] Ensure `create_dataloaders` never silently modifies transcripts
  - What: Log clearly when chunk text population occurs and write a short CSV/JSON with mapping of which chunks were auto-populated. Prefer raising a warning requiring user confirmation.
  - Why: Prevent accidental training on low-quality auto-generated supervision.
  - Files: `src/data/data_loader.py`

- [ ] Use consistent tokenization/truncation policy and log affected examples
  - What: Audit `processor.tokenizer(..., max_length=448, truncation=True)` usage. Log count of examples truncated and consider raising `max_length` or splitting long references earlier in pipeline.
  - Why: Truncation of labels can remove training signal or bias model behavior.
  - Files: `src/data/data_loader.py`, training `run_config.yaml` saved under each run

- [ ] Add evaluation holdout creation script (strict article-level split)
  - What: Create a script that selects and freezes a strict holdout at raw-file granularity (article-level), saves file list and ensures no overlap with training/val when building `data/aligned_*`.
  - Why: Ensures final comparisons (fine-tuned vs baseline) are done on an identical, leakage-free holdout.
  - Files: new `tools/create_holdout.py`, document in `NEXT_TODO.md`

### Model & training suggestions

- [ ] Address hallucination / underfitting with controlled experiments
  - What: Run ablation experiments: (A) lower LR (e.g., 2e-6), (B) stronger dropout (0.2–0.3) in decoder/encoder heads, (C) more epochs (e.g., 20–30) with early stopping, (D) LoRA + progressive unfreeze of encoder layers.
  - Why: Repo memory and diagnostic logs indicate hallucination likely from underfitting or mis-regularization; these are standard mitigations.
  - Files: `cli/train.py`, model adapter config in `src/models/*` (where dropout is configured)

- [ ] Record and log full run deterministic metadata
  - What: Save `git commit`, python env (`pip freeze`), `run_config.yaml`, random seeds, and dataset manifest (list of files used) in the run output directory.
  - Why: Required for reproducibility and debugging of future regressions.
  - Files: training wrapper in `cli/train.py` — add code to capture these artifacts at start.

### Evaluation & metrics

- [ ] Unify evaluation code path and offline metric scripts
  - What: Factor out a shared `evaluate_predictions(preds, refs, preprocessor)` helper used by both `compute_metrics` and `cli/eval_model.py` so the same exact cleaning/normalization is applied.
  - Why: Prevent subtle differences between training-time metrics and post-hoc evaluation.
  - Files: `cli/train.py`, `cli/eval_model.py`, new `src/utils/metrics.py` helper.

- [ ] Store prediction JSONL with both raw and preprocessed fields
  - What: When writing predictions (in `cli/eval_model.py`), include `prediction_raw`, `prediction_preprocessed`, `reference_raw`, `reference_preprocessed`, and the `id` and `audio_path` if available.
  - Why: Facilitates error analysis and debugging of normalization mismatch.
  - Files: `cli/eval_model.py`, evaluation output folder `logs/verification/...`

### Data & alignment pipeline improvements

- [ ] Verify `cli/build_aligned_chunks.py` outputs and generate QA report
  - What: After running alignment, create a QA report that samples chunks and compares timestamp spans vs reference word indices and flags suspicious cases (e.g., chunks where more than X% of words are unmatched or spans > 30s).
  - Why: Prevent poor alignment silently entering training set.
  - Files: `cli/build_aligned_chunks.py`, create `data/aligned_*/report.json` (already exists) — extend with additional checks.

- [ ] Add option to preserve mapping from chunk -> source raw file
  - What: When writing aligned chunks, include metadata file mapping `out_stem -> source_audio, start_sec, end_sec, ref_word_range`.
  - Why: Useful for error analysis and to build strict holdout sets.
  - Files: `cli/build_aligned_chunks.py`, write `data/aligned_raw_v*/manifest.json`

### Diagnostics & error analysis

- [ ] Implement category-level error analysis pipeline
  - What: Post-process `predictions.jsonl` to compute WER/CER and error breakdown for categories: numbers, names, sports entities, dates, punctuation, out-of-vocabulary tokens.
  - Why: Guides targeted data augmentation or targeted loss weighting.
  - Files: new `tools/analyze_errors.py`, use `logs/verification/..._predictions.jsonl`

### Repro/ops & small safety items

- [ ] Default `num_workers=0` in production reproducible runs and document behavior
  - What: Recommend `num_workers=0` for exact reproducibility on CI or when re-running experiments; allow >0 for speed in exploratory runs but require worker seeding.
  - Why: Avoid hidden non-determinism in offline comparisons.
  - Files: `run_config.yaml` template and `cli/train.py` docs

- [ ] Add a CI check for `data/aligned_*` manifest vs holdout lists
  - What: Create a lightweight script that ensures holdout file lists are not present in training manifests and run it as a pre-merge or pre-release check.
  - Why: Prevent accidental leakage when updating data or scripts.
  - Files: new `tools/check_holdout_conflicts.py`

---

## How to verify each item (short)

- For preprocessing consistency: run a quick local script that loads a small validation set, runs both `compute_metrics` (simulated) and `cli/eval_model.py` pipeline and asserts identical WER/CER and identical preprocessed text for a sample.
- For DataLoader determinism: run two short training epochs with `seed=42`, `num_workers=4` (after worker init fixes) and assert that initial minibatch label sequences are identical across runs.
- For alignment QA: run `cli/build_aligned_chunks.py --max_files 5` and inspect `data/aligned_.../manifest.json` and `report.json` and sample WAV+TXT pairs.

---

If you want, I can: 
- implement the `compute_metrics` fix and the unified evaluation helper now (small safe code change),
- or produce the `tools/create_holdout.py` and `tools/analyze_errors.py` scripts next.

Pick one of the two and I'll apply the changes.
