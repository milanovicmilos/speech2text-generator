# Serbian ASR Project

Production-oriented speech recognition project for Serbian language.

The codebase is organized around a model-agnostic ASR architecture:
- model contracts (`ASRModelAdapter`, generation config)
- adapter implementations (currently Whisper)
- registry/factory selection (`--model_type`)

This keeps training, evaluation, and inference entrypoints stable while enabling future model additions with minimal CLI changes.

## Project Structure

- `src/`
	- `models/` contracts, registry, and model adapters
	- `data/` dataset loading, pairing, splitting, collator
	- `inference/` transcriber service
	- `utils/` config, metrics, text preprocessing
- `cli/` main command-line workflows
- `tools/` helper scripts (analysis, scraping, diagnostics)
- `configs/` runtime config (`config.yaml`)
- `notebooks/` final notebook artifacts (Kaggle/EDA)
- `docs/` course requirements and submission documents
- `tests/` integrity checks (e.g., holdout leakage)
- `data/`, `models/`, `logs/`, `res/`, `dist/` runtime artifacts

## Source Of Truth And Kaggle Packaging

- Canonical implementation lives only in `src/`, `cli/`, `tools/`, `configs/`, `kaggle/`.
- Kaggle mirror folders were removed to avoid drift and inconsistent behavior.
- Build a clean Kaggle package from canonical folders:

```bash
python build_kaggle.py
```

- Output is generated under `dist/kaggle_bundle/` and `dist/kaggle_bundle.zip`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Main Workflows

### Train

```bash
python cli/train.py --data_dir data/raw --output_dir models/whisper --model_name openai/whisper-base --model_type whisper
```

### Evaluate

```bash
python cli/eval_model.py --data_dir data/raw --model_dir models/whisper/final --model_type whisper --split test
```

### Transcribe

```bash
python cli/transcribe.py --audio path/to/audio.wav --model models/whisper/final --model_type whisper
```

### Compare baseline vs fine-tuned

```bash
python cli/compare_baseline.py --data_dir data/raw --baseline_model openai/whisper-base --finetuned_model models/whisper/final --model_type whisper --split test
```

## Secondary Utilities

- Audio conversion: `python cli/convert_audio.py --input_dir data/raw --output_dir data/processed`
- Dataset validation: `python cli/validate_dataset.py --audio_dir data/chunked/sport/audio --text_dir data/chunked/sport/text`
- Chunk text recovery: `python cli/fill_empty_chunks.py --base_dir data/chunked/sport`
- Chunk text repopulation: `python cli/repopulate_chunks.py --base_dir data/chunked/sport`
- Dataset EDA: `python cli/analyze_dataset.py --data_dir data/raw`

## Notes

- `--model_type` is the extension point for new ASR backends.
- Current default and supported value is `whisper`.
- Keep generated artifacts (`models/`, `logs/`, `data/problematic/`) out of manual source edits.
