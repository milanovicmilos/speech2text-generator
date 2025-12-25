#!/usr/bin/env python3
"""
CLI script for running inference with Whisper model.

Usage:
    python cli/transcribe.py --audio audio.wav --model models/whisper/final
"""

import sys
import logging
import argparse
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import setup_logging, get_device
from src.inference import WhisperTranscriber

logger = logging.getLogger(__name__)


def setup_arg_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio with Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="models/whisper/final",
        help="Path to Whisper model",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device (cpu or cuda)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for transcription",
    )
    
    return parser


def main():
    """Main inference function."""
    # Parse arguments
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    
    logger.info("=" * 80)
    logger.info("WHISPER ASR - INFERENCE")
    logger.info("=" * 80)
    
    # Load transcriber
    logger.info(f"Loading model from {args.model}...")
    transcriber = WhisperTranscriber(
        model_path=args.model,
        device=args.device,
        language="Serbian"
    )
    
    # Transcribe
    logger.info(f"Transcribing {args.audio}...")
    transcription = transcriber.transcribe(args.audio)
    
    logger.info("=" * 80)
    logger.info("TRANSCRIPTION RESULT:")
    logger.info("=" * 80)
    logger.info(transcription)
    logger.info("=" * 80)
    
    # Save output if requested
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(transcription, encoding='utf-8')
        logger.info(f"Transcription saved to: {args.output}")


if __name__ == "__main__":
    main()
