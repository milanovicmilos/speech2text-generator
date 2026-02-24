#!/usr/bin/env python3
"""
CLI script for running ASR inference.

Usage:
    python cli/transcribe.py --audio audio.wav --model models/whisper/final
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Any, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import setup_logging, load_config
from src.inference import ASRTranscriber

logger = logging.getLogger(__name__)


def _load_generation_defaults(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load generation defaults from config with safe fallback values."""
    fallback = {
        "num_beams": 8,
        "no_repeat_ngram_size": 10,
        "repetition_penalty": 5.0,
        "length_penalty": 1.0,
        "temperature": 0.0,
        "max_new_tokens": 128,
        "max_length": 256,
    }
    try:
        config = load_config(config_path)
    except Exception:
        return fallback

    generation = config.get("generation", {}) if isinstance(config, dict) else {}
    if not isinstance(generation, dict):
        return fallback

    defaults = dict(fallback)
    for key in fallback:
        if key in generation:
            defaults[key] = generation[key]
    return defaults


def setup_arg_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    generation_defaults = _load_generation_defaults()

    parser = argparse.ArgumentParser(
        description="Transcribe audio with ASR model",
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
        help="Path to model directory",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="whisper",
        help="Model type registered in model registry",
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
    
    
    parser.add_argument(
        "--num_beams",
        type=int,
        default=int(generation_defaults["num_beams"]),
        help="Number of beams for beam search",
    )
    
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=int(generation_defaults["no_repeat_ngram_size"]),
        help="No-repeat ngram size",
    )
    
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=float(generation_defaults["repetition_penalty"]),
        help="Repetition penalty",
    )
    
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=float(generation_defaults["length_penalty"]),
        help="Length penalty",
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(generation_defaults["temperature"]),
        help="Temperature for sampling",
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
    logger.info("SERBIAN ASR - INFERENCE")
    logger.info("=" * 80)
    
    # Load transcriber
    logger.info(f"Loading model from {args.model}...")
    generation_params = {
        "num_beams": args.num_beams,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
        "temperature": args.temperature,
    }
    transcriber = ASRTranscriber(
        model_path=args.model,
        model_type=args.model_type,
        device=args.device,
        language="Serbian",
        generation_params=generation_params,
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
