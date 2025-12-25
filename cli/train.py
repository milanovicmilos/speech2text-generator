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
    
    return parser


def main():
    """Main training function."""
    # Parse arguments
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_dir)
    set_seed(args.seed)
    device = get_device()
    
    logger.info("=" * 80)
    logger.info("WHISPER ASR MODEL TRAINING - SERBIAN")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"FP16: {args.fp16}")
    logger.info("=" * 80)
    
    # Load model and processor
    logger.info(f"Loading model {args.model_name}...")
    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    
    # Freeze encoder
    for param in model.model.encoder.parameters():
        param.requires_grad = False
    logger.info("Froze encoder parameters")
    
    params = count_parameters(model)
    logger.info(f"Total parameters: {params['total']:,}")
    logger.info(f"Trainable parameters: {params['trainable']:,}")
    
    # Load dataset
    logger.info("Loading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        processor=processor,
        batch_size=args.batch_size,
    )
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        gradient_accumulation_steps=1,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        logging_dir=Path(args.log_dir) / "tensorboard",
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        data_collator=WhisperDataCollator(processor=processor),
    )
    
    # Train
    logger.info("Starting training...")
    logger.info("=" * 80)
    trainer.train()
    
    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info(f"Best model saved to: {args.output_dir}")
    
    # Save final model
    final_model_dir = Path(args.output_dir) / "final"
    model.save_pretrained(final_model_dir)
    processor.save_pretrained(final_model_dir)
    logger.info(f"Final model saved to: {final_model_dir}")


if __name__ == "__main__":
    main()
