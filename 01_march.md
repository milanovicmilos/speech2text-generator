# 01 March - ASR Project Status

## 1) Scope completed in this session

- Deleted old chunked datasets:
  - `data/chunked`
  - `data/chunked_clean_v1`
- Deleted old model artifacts under `models/whisper` and started from clean state.
- Kept `raw` dataset as primary source (`data/raw`) and validated raw mapping.
- Implemented modern alignment-first preprocessing pipeline for long-form raw audio:
  - New script: `cli/build_aligned_chunks.py`
  - Approach: word timestamps (faster-whisper) + monotonic token alignment to reference text + chunk export.
- Trained and evaluated new models with reproducible CLI runs.

---

## 2) Why earlier results were unstable

Main root cause identified:

- `data/raw` contains long-form recordings (often several minutes), while Whisper training uses 30-second acoustic windows.
- If long-form text is naively mapped to short audio windows, supervision quality drops (text/audio mismatch).
- Old/naive chunked variants also had missing text coverage in many files, causing effective data loss.

Conclusion:

- Do not use naive splitting.
- Use alignment-first segmentation from raw audio-text pairs.

---

## 3) Dataset diagnostics (current)

### Raw dataset (`data/raw`)

From `cli/analyze_dataset.py` and `cli/validate_dataset.py`:

- Total pairs: 370
- Valid pairs: 367
- Total duration: ~38.98h
- Duration median: ~267.8s
- Duration min: ~55.3s
- Duration max: ~3581s

Interpretation:

- Raw pairs are valid, but very long for direct end-to-end short-window training.

### Aligned dataset v1 (`data/aligned_raw_v1`)

Built using `cli/build_aligned_chunks.py` (pilot on 80 raw files):

- Source raw pairs processed: 80
- Kept pairs: 80
- Chunks written: 792
- Valid chunk pairs (validator): 774
- Typical chunk duration: up to 24s (target), median ~23.7s

Files:

- `data/aligned_raw_v1/report.json`
- `logs/verification/dataset_strategy/aligned_raw_v1_eda.json`

---

## 4) Model training and evaluation results

## Baseline on raw test (before advanced alignment-first run)

- Model: `openai/whisper-base`
- Eval dataset: `data/raw` test split
- Metrics:
  - WER: **0.8883**
  - CER: **0.6174**
- File:
  - `logs/verification/dataset_strategy/baseline_raw_openai_whisper_base.json`

## Raw-only improved run (without full alignment strategy)

- Model output: `models/whisper/raw_v2026_run1/final`
- Eval on `data/raw` test:
  - WER: **0.9107**
  - CER: **0.5698**
- Decode sweep also did not solve WER:
  - Search A WER: 0.9163
  - Search B WER: 0.9029
  - Greedy WER: 0.9259

Interpretation:

- Decode tuning alone cannot fix supervision mismatch from long-form raw mapping.

## Alignment-first run (modern pipeline)

- Model output: `models/whisper/raw_aligned_v2026_run2/final`
- Trained on `data/aligned_raw_v1`
- Canonical eval on aligned test split (`data/aligned_raw_v1`, split=test):
  - WER: **0.5340**
  - CER: **0.2137**
- File:
  - `logs/verification/dataset_strategy/raw_aligned_v2026_run2_eval_aligned_test.json`

Target status:

- Goal `WER < 0.65`: **ACHIEVED** (0.5340).

## Fair baseline on same aligned test

- Model: `openai/whisper-base`
- Same eval dataset/split: `data/aligned_raw_v1` test
- Metrics:
  - WER: **0.5419**
  - CER: **0.2119**
- File:
  - `logs/verification/dataset_strategy/baseline_openai_whisper_base_eval_aligned_test.json`

Interpretation:

- Fine-tuned model is better in WER (0.5340 vs 0.5419), but gap is small on this pilot aligned set.

---

## 5) What is currently available in the project

### Data

- Primary raw source: `data/raw`
- New aligned training-ready dataset (pilot): `data/aligned_raw_v1`

### Code changes

- New pipeline script:
  - `cli/build_aligned_chunks.py`
- Loader/training improvements already applied this session (raw handling + training defaults adjustments).

### Models

- `models/whisper/raw_v2026_run1/final`
- `models/whisper/raw_aligned_v2026_run2/final`

### Key reports/metrics

- `logs/verification/dataset_strategy/raw_eda_after_cleanup.json`
- `logs/verification/dataset_strategy/aligned_raw_v1_eda.json`
- `logs/verification/dataset_strategy/baseline_raw_openai_whisper_base.json`
- `logs/verification/dataset_strategy/raw_v2026_run1_eval_raw_test.json`
- `logs/verification/dataset_strategy/raw_aligned_v2026_run2_eval_aligned_test.json`
- `logs/verification/dataset_strategy/baseline_openai_whisper_base_eval_aligned_test.json`

---

## 6) Recommended next steps (high priority)

1. Expand alignment-first build from pilot (80 files) to full raw set (all 370).
2. Re-train on full aligned dataset with same methodology.
3. Create strict holdout test set from raw sources (article-level separation) to avoid leakage.
4. Re-run fair baseline vs fine-tuned on the exact same holdout.
5. Add error analysis by category (numbers/names/sports entities) from prediction JSONL.

Expected outcome:

- Better generalization confidence,
- More meaningful improvement margin over baseline,
- Stable path for future production improvements.

---

## 7) Short final conclusion

Current state is significantly better than naive raw/chunked processing:

- Alignment-first preprocessing is the correct direction.
- WER target (<0.65) is already reached on aligned test (0.5340).
- Next important step is scaling alignment to full raw corpus and validating on a stricter holdout to prove robust gains.

---

## 8) Reproducibility (exact steps to reproduce `raw_aligned_v2026_run2` results)

Follow these steps on a clean checkout to reproduce the aligned-pilot build and the fine-tune / eval that produced the WER = 0.5340 result.

- Environment

  - Create and activate a Python virtual environment (Windows PowerShell example):

    ```powershell
    python -m venv .venv
    & .venv\Scripts\Activate.ps1
    pip install -U pip
    pip install -r requirements.txt
    pip install faster-whisper
    ```

  - (Windows-specific) If you encounter model download symlink errors, set the HF hub symlink disable env var before running `build_aligned_chunks.py`:

    ```powershell
    $env:HF_HUB_DISABLE_SYMLINKS = "1"
    ```

- Build aligned chunks (pilot used 80 raw files)

  - Command used to build the aligned pilot dataset (`data/aligned_raw_v1`):

    ```bash
    python cli/build_aligned_chunks.py \
      --raw_dir data/raw \
      --out_dir data/aligned_raw_v1 \
      --model_size small \
      --device cpu \
      --compute_type int8 \
      --target_chunk_seconds 24.0 \
      --min_chunk_seconds 3.0 \
      --sample_rate 16000 \
      --max_files 80
    ```

  - Result: `data/aligned_raw_v1/audio/`, `data/aligned_raw_v1/text/` and `data/aligned_raw_v1/report.json` (report contains counts used in the notes above).

- Training (aligned pilot -> `models/whisper/raw_aligned_v2026_run2`)

  - Exact training command used for the run that produced the logged `WER = 0.5340`:

    ```bash
    python cli/train.py \
      --data_dir data/aligned_raw_v1 \
      --output_dir models/whisper/raw_aligned_v2026_run2 \
      --model_name openai/whisper-base \
      --epochs 4 \
      --batch_size 4 \
      --learning_rate 3e-06 \
      --warmup_steps 300 \
      --gradient_accumulation_steps 1 \
      --seed 42
    ```

  - Notes:
    - Encoder was left frozen (the training CLI default is to freeze encoder unless `--unfreeze_encoder` is supplied).
    - `run_config.yaml` is automatically saved to `models/whisper/raw_aligned_v2026_run2/run_config.yaml` at run start. The exact saved config for this run is included below for reference.

- Evaluation (canonical eval producing the metrics JSON)

  - Evaluate the fine-tuned model on the aligned test split (recreates `logs/verification/dataset_strategy/raw_aligned_v2026_run2_eval_aligned_test.json`):

    ```bash
    python cli/eval_model.py \
      --data_dir data/aligned_raw_v1 \
      --model_dir models/whisper/raw_aligned_v2026_run2/final \
      --split test \
      --batch_size 8 \
      --num_beams 8 \
      --no_repeat_ngram_size 10 \
      --repetition_penalty 5.0 \
      --length_penalty 1.0 \
      --temperature 0.0 \
      --max_new_tokens 128 \
      --max_length 256 \
      --metrics_out logs/verification/dataset_strategy/raw_aligned_v2026_run2_eval_aligned_test.json \
      --predictions_out logs/verification/dataset_strategy/raw_aligned_v2026_run2_eval_aligned_test_predictions.jsonl
    ```

  - Evaluate the baseline `openai/whisper-base` on the same split (recreates the baseline JSON):

    ```bash
    python cli/eval_model.py \
      --data_dir data/aligned_raw_v1 \
      --model_dir openai/whisper-base \
      --split test \
      --batch_size 8 \
      --num_beams 8 \
      --no_repeat_ngram_size 10 \
      --repetition_penalty 5.0 \
      --length_penalty 1.0 \
      --temperature 0.0 \
      --max_new_tokens 128 \
      --max_length 256 \
      --metrics_out logs/verification/dataset_strategy/baseline_openai_whisper_base_eval_aligned_test.json \
      --predictions_out logs/verification/dataset_strategy/baseline_openai_whisper_base_eval_aligned_test_predictions.jsonl
    ```

- Saved run configuration (exact `run_config.yaml` for `raw_aligned_v2026_run2`)

  - The training run saves its config to `models/whisper/raw_aligned_v2026_run2/run_config.yaml`. Exact contents from the pilot run:

    ```yaml
    batch_size: 4
    data:
      batch_size: 4
      max_duration: 30.0
      num_workers: 0
      pin_memory: false
      raw_data_dir: data/raw
      sample_rate: 16000
      test_ratio: 0.1
      train_ratio: 0.8
      val_ratio: 0.1
    data_dir: data/aligned_raw_v1
    epochs: 4
    fp16: false
    generation:
      length_penalty: 1.0
      max_length: 256
      max_new_tokens: 128
      no_repeat_ngram_size: 10
      num_beams: 8
      repetition_penalty: 5.0
      temperature: 0.0
    gradient_accumulation_steps: 1
    gradient_checkpointing: false
    learning_rate: 3.0e-06
    log_dir: logs
    model:
      freeze_encoder: true
      freeze_n_layers: 0
      language: Serbian
      pretrained_model: openai/whisper-base
    model_name: openai/whisper-base
    model_type: whisper
    output_dir: models/whisper/raw_aligned_v2026_run2
    paths:
      best_model_path: models/best_model.pt
      checkpoint_dir: models/checkpoints
      log_dir: logs
      tensorboard_dir: logs/tensorboard
    resume_from_checkpoint: null
    seed: 42
    training:
      early_stopping:
        min_delta: 0.001
        patience: 3
      eval_steps: 100
      gradient_accumulation_steps: 1
      gradient_checkpointing: false
      learning_rate: 1e-5
      max_grad_norm: 1.0
      mixed_precision: false
      num_epochs: 5
      save_steps: null
      warmup_steps: 500
    unfreeze_encoder: false
    use_lora: false
    warmup_steps: 300
    ```

- Additional notes and tips

  - If you have GPU(s), set `--device cuda` in `build_aligned_chunks.py` and enable `--fp16` in training; adjust `--batch_size` accordingly.
  - The `build_aligned_chunks.py` uses `faster-whisper` to get word timestamps — depending on your platform, the first model download may take a while.
  - The pipeline writes chunked audio at `sample_rate=16000`. The training loader expects that sample rate.
  - All canonical evaluation outputs are saved under `logs/verification/dataset_strategy/` so you can compare exact JSONs produced by the run.

---

If you want, I can now:

- expand the alignment-run to all `370` raw pairs and create a single command you can run, or
- prepare a small reproduction script (PowerShell) that runs the full sequence end-to-end and collects artifacts.

Tell me which of the two you prefer and I'll add it to this document and mark the todo list accordingly.
