#!/usr/bin/env python3
"""
CLI script for evaluating Whisper model.

Usage:
    python cli/evaluate.py --data_dir data/raw --model_dir models/whisper/final --split test
"""

import sys
import logging
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import librosa
from evaluate import load
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import setup_logging, get_device

logger = logging.getLogger(__name__)


def preprocess_text(text: str) -> str:
    """Preprocess text for evaluation."""
    import re
    text = text.lower()
    text = re.sub(r'[\.\'\"!\-—–\(\)\[\]{}<>;:,?\#\*]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_pairs(data_dir: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load audio-text pairs from directory."""
    data = []
    base = Path(data_dir)
    audio_exts = ("mp3", "wav", "flac")
    
    for ext in audio_exts:
        for ap in base.glob(f"**/*.{ext}"):
            tp = ap.with_suffix('.txt')
            if tp.exists():
                try:
                    text = tp.read_text(encoding='utf-8').strip()
                    text = preprocess_text(text)
                    data.append({"audio_path": str(ap), "text": text})
                except Exception as e:
                    logger.warning(f"Skip {ap}: {e}")
    
    if not data:
        raise RuntimeError(f"No audio+txt pairs in {data_dir}")
    
    random.seed(42)
    random.shuffle(data)
    
    n_train = int(0.8 * len(data))
    n_val = int(0.1 * len(data))
    
    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]
    
    return train, val, test


def setup_arg_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate Whisper ASR model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Data directory",
    )
    
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/whisper/final",
        help="Model directory",
    )
    
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Split to evaluate",
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Maximum number of samples to evaluate",
    )
    
    return parser


def main():
    """Main evaluation function."""
    # Parse arguments
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    device = get_device()
    
    logger.info("=" * 80)
    logger.info("WHISPER ASR - EVALUATION")
    logger.info("=" * 80)
    
    # Load model
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        alt = Path('models/whisper')
        logger.info(f"{model_dir} not found, falling back to {alt}")
        model_dir = alt
    
    logger.info(f"Loading model from {model_dir}...")
    processor = WhisperProcessor.from_pretrained(str(model_dir))
    model = WhisperForConditionalGeneration.from_pretrained(str(model_dir))
    model = model.to(device)
    model.eval()
    
    # Language prompt for Serbian transcribe
    try:
        forced_ids = processor.get_decoder_prompt_ids(language='serbian', task='transcribe')
    except Exception:
        forced_ids = None
    
    # Load data
    train, val, test = load_pairs(args.data_dir)
    ds = val if args.split == "val" else test
    
    if args.max_samples:
        ds = ds[:min(len(ds), args.max_samples)]
    
    logger.info(f"Evaluating on {args.split} split, {len(ds)} samples")
    logger.info("=" * 80)
    
    # Evaluate
    refs, preds = [], []
    
    for i, item in enumerate(ds):
        try:
            audio, _ = librosa.load(item['audio_path'], sr=16000, mono=True)
        except Exception as e:
            logger.warning(f"Fail load {item['audio_path']}: {e}")
            continue
        
        if len(audio) > 16000 * 30:
            audio = audio[:16000 * 30]
        
        inputs = processor(audio, sampling_rate=16000, return_tensors='pt').input_features.to(device)
        
        with torch.no_grad():
            gen_kwargs = {}
            if forced_ids is not None:
                gen_kwargs['forced_decoder_ids'] = forced_ids
            predicted_ids = model.generate(inputs, **gen_kwargs)
        
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        text = preprocess_text(text)
        
        refs.append(item['text'])
        preds.append(text)
        
        if (i + 1) % 10 == 0:
            logger.info(f"{i+1}/{len(ds)} done")
    
    # Calculate WER
    wer_metric = load('wer')
    wer = wer_metric.compute(predictions=preds, references=refs)
    
    logger.info("=" * 80)
    logger.info(f"WER ({args.split}, n={len(refs)}): {wer:.4f}")
    logger.info("=" * 80)
    
    # Show examples
    logger.info("Example predictions:")
    for k in range(min(5, len(refs))):
        logger.info(f"\nREF: {refs[k]}")
        logger.info(f"HYP: {preds[k]}")


if __name__ == '__main__':
    main()
