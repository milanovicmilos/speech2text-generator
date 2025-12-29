#!/usr/bin/env python3
"""
CLI script for training Whisper ASR model.

Usage:
    python cli/train.py --data_dir data/raw --epochs 5 --batch_size 4
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Tuple

import torch
import numpy as np
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate
import librosa

# Add src to path so we can import the library
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    WhisperSpeechDataset,
    WhisperDataCollator,
    create_dataloaders,
    WhisperASRModel,
    load_config,
    setup_logging,
    set_seed,
    get_device,
    count_parameters,
)

logger = logging.getLogger(__name__)


def compute_metrics(pred, tokenizer):
    """Compute WER for evaluation."""
    import evaluate
    wer = evaluate.load('wer')
    cer = evaluate.load('cer')
    
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
    
    # Compute metrics
    wer_score = wer.compute(predictions=pred_str, references=label_str)
    cer_score = cer.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer_score, "cer": cer_score}


def setup_arg_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Train Whisper for Serbian ASR",
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
    set_seed(42)

    # Load config
    config = load_config()
    
    # Override with CLI args
    config.update(vars(args))
    
    logger.info('=' * 80)
    logger.info('WHISPER ASR MODEL TRAINING - SERBIAN')
    logger.info('=' * 80)
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
    processor = WhisperProcessor.from_pretrained(
        config['model_name'],
        language='sr',
        task='transcribe'
    )
    model = WhisperForConditionalGeneration.from_pretrained(config['model_name'])
    
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

        params = count_parameters(model)
        logger.info(f'Trainable parameters: {params["trainable"]:,}')
    else:
        logger.info('Training full model (encoder + decoder)')
        params = count_parameters(model)
        logger.info(f'Total parameters: {params["total"]:,}')
        logger.info(f'Trainable parameters: {params["trainable"]:,}')
    
    device = get_device()
    model.to(device)
    
    # Load dataset
    logger.info('Loading dataset...')
    train_loader, val_loader, test_loader = create_dataloaders(
        config['data_dir'],
        processor,
        batch_size=config['batch_size']
    )
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config['output_dir'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        gradient_accumulation_steps=1,
        learning_rate=config['learning_rate'],
        warmup_steps=500,
        max_steps=-1,
        num_train_epochs=config['epochs'],
        eval_strategy='epoch',
        save_strategy='epoch',

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
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor.tokenizer),
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
