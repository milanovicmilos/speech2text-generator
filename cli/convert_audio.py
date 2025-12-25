#!/usr/bin/env python3
"""
Utility script for converting audio files to WAV format.

Usage:
    python tools/convert_audio.py --input_dir data/raw --output_dir data/processed
"""

import sys
import logging
import argparse
from pathlib import Path

from pydub import AudioSegment

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def convert_directory(input_dir: str, output_dir: str = None):
    """Convert all MP3 files in directory to WAV."""
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_path = input_path
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files
    audio_files = (
        list(input_path.glob("*.mp3")) +
        list(input_path.glob("*.m4a")) +
        list(input_path.glob("*.ogg"))
    )
    logger.info(f"Found {len(audio_files)} audio files")
    
    for audio_file in audio_files:
        try:
            wav_file = output_path / f"{audio_file.stem}.wav"
            
            # Skip if WAV already exists
            if wav_file.exists():
                logger.info(f"Skipping {audio_file.name} (WAV already exists)")
                continue
            
            logger.info(f"Converting {audio_file.name}...")
            
            # Load and convert
            audio = AudioSegment.from_file(str(audio_file))
            audio.export(str(wav_file), format="wav", parameters=["-q:a", "9"])
            
            logger.info(f"Saved to {wav_file.name}")
        
        except Exception as e:
            logger.error(f"Error converting {audio_file.name}: {e}")


def setup_arg_parser() -> argparse.ArgumentParser:
    """Setup argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert audio files to WAV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory with audio files",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for WAV files (default: same as input)",
    )
    
    return parser


def main():
    """Main conversion function."""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    convert_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
