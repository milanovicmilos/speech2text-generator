#!/usr/bin/env python3
"""
CLI script for evaluating Whisper model.

Usage:
    python cli/evaluate.py --data_dir data/raw --model_dir models/whisper/final --split test
"""

import sys
import logging
import argparse
from pathlib import Path

import torch
import librosa
from evaluate import load
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import setup_logging, get_device, create_dataloaders
from src.utils.text_preprocessing import SerbianTextPreprocessor

logger = logging.getLogger(__name__)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Whisper ASR model")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Data directory")
    parser.add_argument("--model_dir", type=str, default="models/whisper/final", help="Model directory")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Split to evaluate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    
    args = parser.parse_args()
    
    setup_logging('logs')
    device = get_device()
    
    logger.info("=" * 80)
    logger.info("WHISPER ASR - EVALUATION")
    logger.info("=" * 80)
    
    # Load model
    logger.info(f'Loading model from {args.model_dir} for evaluation')
    processor = WhisperProcessor.from_pretrained(args.model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()
    
    # Load data
    _, val_loader, test_loader = create_dataloaders(args.data_dir, processor, batch_size=args.batch_size)
    loader = val_loader if args.split == "val" else test_loader
    
    preproc = SerbianTextPreprocessor()
    wer_metric = load('wer')
    cer_metric = load('cer')
    preds = []
    refs = []
    
    logger.info(f'Running evaluation on {args.split} set')
    for batch in loader:
        input_feats = batch['input_features'].to(device)
        attn = batch.get('attention_mask')
        if attn is not None:
            attn = attn.to(device)
        with torch.no_grad():
            try:
                generated = model.generate(input_feats, attention_mask=attn, language='sr', task='transcribe')
            except TypeError:
                try:
                    forced = processor.get_decoder_prompt_ids(language='sr', task='transcribe')
                    generated = model.generate(input_feats, attention_mask=attn, forced_decoder_ids=forced)
                except Exception:
                    generated = model.generate(input_feats, attention_mask=attn)
        decoded = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
        for i, text in enumerate(decoded):
            preds.append(preproc.preprocess(text))
            refs.append(preproc.preprocess(batch['text'][i]))
    
    logger.info(f'Computed {len(preds)} predictions')
    logger.info('WER: %.4f', wer_metric.compute(predictions=preds, references=refs))
    logger.info('CER: %.4f', cer_metric.compute(predictions=preds, references=refs))


if __name__ == '__main__':
    main()
