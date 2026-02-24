#!/usr/bin/env python3
"""
CLI script for validating ASR dataset mapping.

Usage:
    python cli/validate_dataset.py --audio_dir data/chunked/sport/audio --text_dir data/chunked/sport/text
"""

import sys
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import setup_logging
from src.data.data_loader import WhisperSpeechDataset
from transformers import WhisperProcessor

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Validate ASR dataset mapping and report issues")
    parser.add_argument('--audio_dir', type=str, default='data/chunked/sport/audio', help='Directory with audio chunks')
    parser.add_argument('--text_dir', type=str, default='data/chunked/sport/text', help='Directory with transcript chunks')
    
    args = parser.parse_args()
    
    setup_logging('logs')
    
    audio_dir = args.audio_dir
    text_dir = args.text_dir
    logger.info('Validating dataset mapping...')
    processor = WhisperProcessor.from_pretrained('openai/whisper-tiny')
    ds = WhisperSpeechDataset(audio_dir, text_dir, processor)
    print('Total audio files:', len(ds.audio_files))
    print('Valid pairs:', len(ds.valid_indices))
    report = Path('data/problematic/problematic_pairs.json')
    if report.exists():
        print('Problematic report:', report)
    else:
        print('No problematic pairs reported')


if __name__ == '__main__':
    main()