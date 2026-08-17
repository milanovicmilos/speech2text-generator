#!/usr/bin/env python3
"""
CLI script for training ASR model.

Usage:
    python cli/train.py --data_dir data/raw --epochs 5 --batch_size 4
"""

import sys
import logging
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# Add src to path so we can import the library
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    WhisperDataCollator,
    create_dataloaders,
    load_config,
    save_config,
    setup_logging,
    set_seed,
    get_device,
    count_parameters,
    get_model_registry,
)
from src.utils.metrics import evaluate_predictions
from src.utils.text_preprocessing import SerbianTextPreprocessor

logger = logging.getLogger(__name__)


def compute_metrics(pred, tokenizer, preprocessor: SerbianTextPreprocessor):
    """Compute WER/CER with the same normalization pipeline as offline evaluation."""
    pred_logits = pred.predictions
    label_ids = pred.label_ids

    # Helper: convert a single element (tensor/ndarray/list) to numpy
    def as_numpy(x):
        try:
            import torch as _torch
            if isinstance(x, _torch.Tensor):
                return x.cpu().numpy()
        except Exception:
            pass
        try:
            return np.array(x)
        except Exception:
            return np.array(list(x))

    # Normalize label ids to list of int lists and replace -100
    labels_list = []
    if label_ids is None:
        labels_list = []
    elif isinstance(label_ids, (list, tuple)):
        for l in label_ids:
            arr = as_numpy(l)
            arr = arr.astype(int)
            arr[arr == -100] = tokenizer.pad_token_id
            labels_list.append(arr.tolist())
    else:
        arr = as_numpy(label_ids)
        if arr.ndim == 2:
            for row in arr:
                row = row.astype(int)
                row[row == -100] = tokenizer.pad_token_id
                labels_list.append(row.tolist())
        else:
            arr = arr.astype(int)
            arr[arr == -100] = tokenizer.pad_token_id
            labels_list.append(arr.tolist())

    # Normalize predictions to list of int lists
    preds_list = []
    if pred_logits is None:
        preds_list = []
    elif isinstance(pred_logits, (list, tuple)):
        for p in pred_logits:
            arr = as_numpy(p)
            if arr.ndim == 3:
                ids = np.argmax(arr, axis=-1)
                # ids may be (1, seq) or (seq_len,)
                for row in ids:
                    preds_list.append(row.astype(int).tolist())
            elif arr.ndim == 2:
                for row in arr:
                    preds_list.append(row.astype(int).tolist())
            elif arr.ndim == 1:
                preds_list.append(arr.astype(int).tolist())
            else:
                preds_list.append(arr.flatten().astype(int).tolist())
    else:
        arr = as_numpy(pred_logits)
        if arr.ndim == 3:
            ids = np.argmax(arr, axis=-1)
            for row in ids:
                preds_list.append(row.astype(int).tolist())
        elif arr.ndim == 2:
            for row in arr:
                preds_list.append(row.astype(int).tolist())
        elif arr.ndim == 1:
            preds_list.append(arr.astype(int).tolist())
        else:
            preds_list.append(arr.flatten().astype(int).tolist())

    # Ensure same length
    n_preds = len(preds_list)
    n_labels = len(labels_list)
    n = min(n_preds, n_labels) if n_preds and n_labels else max(n_preds, n_labels)

    # Decode in small chunks to avoid high memory usage
    pred_str = []
    label_str = []
    chunk = 16
    for i in range(0, n, chunk):
        block_preds = preds_list[i : i + chunk]
        block_labels = labels_list[i : i + chunk]
        # If lengths mismatch, pad with empty lists
        if len(block_preds) < len(block_labels):
            block_preds.extend([[]] * (len(block_labels) - len(block_preds)))
        if len(block_labels) < len(block_preds):
            block_labels.extend([[]] * (len(block_preds) - len(block_labels)))
        pred_str.extend(tokenizer.batch_decode(block_preds, skip_special_tokens=True))
        label_str.extend(tokenizer.batch_decode(block_labels, skip_special_tokens=True))
    
    metrics = evaluate_predictions(
        predictions=pred_str,
        references=label_str,
        preprocessor=preprocessor,
    )
    return {"wer": metrics["wer"], "cer": metrics["cer"]}


def _build_subset_manifest(subset) -> list[dict[str, str]]:
    """Create a lightweight sample manifest from a torch Subset over WhisperSpeechDataset."""
    rows: list[dict[str, str]] = []
    base_dataset = subset.dataset
    for ds_index in subset.indices:
        file_index = base_dataset.valid_indices[ds_index]
        audio_path = base_dataset.audio_files[file_index]
        text_path = base_dataset.text_dir / f"{audio_path.stem}.txt"
        rows.append(
            {
                "sample_id": audio_path.stem,
                "audio_path": str(audio_path),
                "text_path": str(text_path),
            }
        )
    return rows


def _capture_run_metadata(
    output_dir: Path,
    config: dict,
    train_loader,
    val_loader,
    test_loader,
) -> None:
    """Save reproducibility metadata (git commit, pip freeze, dataset manifests)."""
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": config.get("seed"),
        "data_dir": config.get("data_dir"),
        "model_name": config.get("model_name"),
    }

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        metadata_payload["git_commit"] = git_commit
    except Exception:
        metadata_payload["git_commit"] = "unknown"

    (metadata_dir / "run_metadata.json").write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    try:
        pip_freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        (metadata_dir / "pip_freeze.txt").write_text(pip_freeze, encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not capture pip freeze: %s", exc)

    split_manifests = {
        "train_manifest.json": _build_subset_manifest(train_loader.dataset),
        "val_manifest.json": _build_subset_manifest(val_loader.dataset),
        "test_manifest.json": _build_subset_manifest(test_loader.dataset),
    }
    for file_name, rows in split_manifests.items():
        (metadata_dir / file_name).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def setup_arg_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Train ASR model for Serbian",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Data directory with audio files and transcriptions",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/whisper",
        help="Output directory for checkpoints",
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/whisper-base",
        help="Pretrained model name",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="whisper",
        help="Model type from registry",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size",
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Warmup steps",
    )
    
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps to simulate larger batch sizes",
    )
    
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 training",
    )
    
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Log directory",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (default 0 for reproducibility)",
    )

    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Enable DataLoader pin_memory",
    )

    parser.add_argument(
        "--allow_auto_populate",
        action="store_true",
        help="Allow fallback chunk transcript auto-populate when text dir is empty",
    )

    parser.add_argument(
        "--max_label_length",
        type=int,
        default=448,
        help="Maximum tokenizer label length; logs truncated samples",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Optional dropout override applied to model config (e.g. 0.2)",
    )

    parser.add_argument(
        "--freeze_encoder_layers",
        type=int,
        default=0,
        help="Freeze first N encoder layers after unfreezing (for progressive unfreeze experiments)",
    )
    
    parser.add_argument(
        "--unfreeze_encoder",
        action="store_true",
        help="Unfreeze encoder parameters for full fine-tuning",
    )
    
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for efficient fine-tuning",
    )
    
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    
    return parser


def main():
    """CLI entrypoint for training."""
    parser = setup_arg_parser()
    args = parser.parse_args()

    setup_logging('logs')

    # Load config
    config = load_config()
    
    # Override with CLI args
    config.update(vars(args))

    # Reproducibility
    set_seed(config['seed'])
    
    logger.info('=' * 80)
    logger.info('SERBIAN ASR MODEL TRAINING')
    logger.info('=' * 80)
    logger.info(f'Model type: {config["model_type"]}')
    logger.info(f'Model: {config["model_name"]}')
    logger.info(f'Data directory: {config["data_dir"]}')
    logger.info(f'Output directory: {config["output_dir"]}')
    logger.info(f'Epochs: {config["epochs"]}')
    logger.info(f'Batch size: {config["batch_size"]}')
    logger.info(f'Learning rate: {config["learning_rate"]}')
    logger.info(f'FP16: {config["fp16"]}')
    logger.info('=' * 80)
    
    # Load model and processor
    logger.info(f'Loading model {config["model_name"]}...')
    registry = get_model_registry()
    adapter = registry.create(
        config['model_type'],
        model_name_or_path=config['model_name'],
        language='sr',
        freeze_encoder=not config.get('unfreeze_encoder', False),
    )
    processor = adapter.get_processor()
    model = adapter.unwrap()

    if config.get('dropout') is not None:
        dropout_value = float(config['dropout'])
        logger.info('Applying dropout override: %.3f', dropout_value)
        for attr in ('dropout', 'attention_dropout', 'activation_dropout'):
            if hasattr(model.config, attr):
                setattr(model.config, attr, dropout_value)
    
    # Apply LoRA for efficient tuning if requested (2025 best practice)
    if config.get('use_lora', False):
        from peft import LoraConfig, get_peft_model
        logger.info('Applying LoRA for parameter-efficient fine-tuning')
        lora_config = LoraConfig(
            r=16,  # Low-rank dimension
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],  # Whisper attention and FFN
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Unfreeze encoder if requested (overrides LoRA for full tuning)
    if config.get('unfreeze_encoder', False):
        logger.info('Unfreezing encoder for full fine-tuning')
        # More robust: enable grad for any parameter whose name contains 'encoder'
        found = 0
        for name, param in model.named_parameters():
            if 'encoder' in name:
                param.requires_grad = True
                found += 1

        if found == 0:
            # fallback: try searching for common encoder module names in nested base_model
            base = getattr(model, 'base_model', None) or getattr(model, 'model', None)
            if base is not None:
                for name, param in base.named_parameters():
                    if 'encoder' in name:
                        param.requires_grad = True
                        found += 1

        if found == 0:
            logger.warning('Could not locate encoder parameters by name; leaving default trainable params')
        else:
            logger.info(f'Unfroze {found} encoder parameters by name match')

        freeze_n_layers = int(config.get('freeze_encoder_layers', 0) or 0)
        if freeze_n_layers > 0:
            try:
                encoder_layers = model.model.encoder.layers
                for layer_index, layer in enumerate(encoder_layers):
                    if layer_index < freeze_n_layers:
                        for param in layer.parameters():
                            param.requires_grad = False
                logger.info('Kept first %s encoder layers frozen for progressive unfreeze', freeze_n_layers)
            except Exception as exc:
                logger.warning('Could not apply freeze_encoder_layers=%s: %s', freeze_n_layers, exc)

        params = count_parameters(model)
        logger.info(f'Trainable parameters: {params["trainable"]:,}')
    else:
        logger.info('Training full model (encoder + decoder)')
        params = count_parameters(model)
        logger.info(f'Total parameters: {params["total"]:,}')
        logger.info(f'Trainable parameters: {params["trainable"]:,}')
    
    device = get_device()
    adapter.to(device)

    # Persist run config for reproducibility
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, str(output_dir / 'run_config.yaml'))
    
    # Load dataset
    logger.info('Loading dataset...')
    train_loader, val_loader, test_loader = create_dataloaders(
        config['data_dir'],
        processor,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        seed=config['seed'],
        allow_auto_populate=config['allow_auto_populate'],
        max_label_length=config['max_label_length'],
    )
    _capture_run_metadata(
        output_dir=output_dir,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config['output_dir'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps'],
        max_steps=-1,
        num_train_epochs=config['epochs'],
        eval_strategy='epoch',
        save_strategy='epoch',
        predict_with_generate=True,
        generation_max_length=256,

        logging_steps=10,
        save_total_limit=2,  # Keep best and last checkpoint
        load_best_model_at_end=True,
        metric_for_best_model='wer',
        greater_is_better=False,
        fp16=config['fp16'],
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        label_names=['labels'],
    )
    
    # Data collator
    data_collator = WhisperDataCollator(processor)
    
    # Trainer
    text_preprocessor = SerbianTextPreprocessor()
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor.tokenizer, text_preprocessor),
    )
    
    # Train
    logger.info('Starting training...')
    logger.info('=' * 80)
    trainer.train(resume_from_checkpoint=config.get('resume_from_checkpoint'))
    
    # Save final model
    final_model_dir = Path(config['output_dir']) / 'final'
    final_model_dir.mkdir(exist_ok=True)
    model.save_pretrained(final_model_dir)
    processor.save_pretrained(final_model_dir)
    logger.info(f'Final model saved to: {final_model_dir}')


if __name__ == '__main__':
    main()
