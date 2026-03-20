#!/usr/bin/env python3
"""
CLI script for automatically chunking raw audio and text into 30-second segments.

Splits full-length audio files and their transcriptions into aligned 30-second chunks.
Saves chunks to data/chunked/<set>/audio/ and data/chunked/<set>/text/

Usage:
    python cli/chunk_dataset.py --raw_dir data/raw --output_dir data/chunked --chunk_duration 30
"""

import sys
import logging
import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import librosa
import soundfile as sf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import setup_logging
from src.utils.text_preprocessing import SerbianTextPreprocessor

logger = logging.getLogger(__name__)


def get_audio_duration(audio_path: Path, sr: int = 16000) -> float:
    """Get audio duration in seconds."""
    try:
        return float(sf.info(str(audio_path)).duration)
    except Exception:
        return 0.0


def split_text_by_time(text: str, duration_sec: float, chunk_duration: float) -> list:
    """
    Split text into chunks based on time proportions.
    
    Assumes words are evenly distributed across duration.
    """
    words = text.split()
    if not words:
        return []
    
    words_per_sec = len(words) / duration_sec if duration_sec > 0 else 0
    chunks = []
    
    num_chunks = max(1, int(np.ceil(duration_sec / chunk_duration)))
    for i in range(num_chunks):
        start_word = int(i * chunk_duration * words_per_sec)
        end_word = int((i + 1) * chunk_duration * words_per_sec)
        chunk_words = words[start_word:end_word]
        chunks.append(' '.join(chunk_words) if chunk_words else '')
    
    return chunks


def chunk_audio_and_text(
    audio_path: Path,
    text_path: Path,
    output_audio_dir: Path,
    output_text_dir: Path,
    chunk_duration: float = 30.0,
    sr: int = 16000,
):
    """
    Split audio and text into aligned chunks.
    
    Audio is resampled to 16kHz for Wav2Vec2 compatibility.
    
    Args:
        audio_path: Path to audio file
        text_path: Path to text transcription file
        output_audio_dir: Output directory for audio chunks
        output_text_dir: Output directory for text chunks
        chunk_duration: Duration of each chunk in seconds
        sr: Target sample rate (16000 Hz for Wav2Vec2)
    """
    # Load audio and resample to 16kHz
    try:
        audio, loaded_sr = librosa.load(str(audio_path), sr=sr, mono=True)
        logger.info(f"Loaded {audio_path.name} (original sr: {loaded_sr} Hz, resampled to {sr} Hz)")
    except Exception as e:
        logger.error(f"Error loading audio {audio_path}: {e}")
        return
    
    # Verify sample rate is 16kHz
    if sr != 16000:
        logger.warning(f"Sample rate is {sr} Hz, expected 16000 Hz for Wav2Vec2!")
    
    # Get audio duration
    duration_sec = len(audio) / sr
    
    # Load text
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            raw_text = f.read().strip()
    except Exception as e:
        logger.error(f"Error loading text {text_path}: {e}")
        return
    
    # Preprocess text
    preprocessor = SerbianTextPreprocessor()
    text = preprocessor.preprocess(raw_text)
    
    # Split text into chunks
    text_chunks = split_text_by_time(text, duration_sec, chunk_duration)
    
    # Split audio into chunks
    chunk_samples = int(chunk_duration * sr)
    num_chunks = max(1, int(np.ceil(duration_sec / chunk_duration)))
    
    stem = audio_path.stem
    
    for i in range(num_chunks):
        # Extract audio chunk
        start_sample = i * chunk_samples
        end_sample = min((i + 1) * chunk_samples, len(audio))
        audio_chunk = audio[start_sample:end_sample]
        
        # Pad if necessary
        if len(audio_chunk) < chunk_samples:
            audio_chunk = np.pad(audio_chunk, (0, chunk_samples - len(audio_chunk)))
        
        # Save audio chunk
        chunk_audio_name = f"{stem}_chunk{i:03d}.wav"
        chunk_audio_path = output_audio_dir / chunk_audio_name
        try:
            sf.write(str(chunk_audio_path), audio_chunk, sr)
            logger.info(f"Saved audio chunk: {chunk_audio_name}")
        except Exception as e:
            logger.error(f"Error writing audio chunk {chunk_audio_path}: {e}")
        
        # Save text chunk
        chunk_text_name = f"{stem}_chunk{i:03d}.txt"
        chunk_text_path = output_text_dir / chunk_text_name
        text_chunk = text_chunks[i] if i < len(text_chunks) else text_chunks[-1] if text_chunks else text
        try:
            with open(chunk_text_path, 'w', encoding='utf-8') as f:
                f.write(text_chunk)
            logger.info(f"Saved text chunk: {chunk_text_name}")
        except Exception as e:
            logger.error(f"Error writing text chunk {chunk_text_path}: {e}")


def main():
    """Main chunking function."""
    parser = argparse.ArgumentParser(
        description="Chunk raw audio and text files into 30-second segments"
    )
    parser.add_argument(
        '--raw_dir',
        type=str,
        default='data/raw',
        help='Root directory with raw audio and text files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/chunked',
        help='Output directory for chunked data'
    )
    parser.add_argument(
        '--chunk_duration',
        type=float,
        default=30.0,
        help='Duration of each chunk in seconds'
    )
    parser.add_argument(
        '--sr',
        type=int,
        default=16000,
        help='Target sample rate (MUST be 16000 Hz for Wav2Vec2 model). Will resample audio if necessary.'
    )
    
    args = parser.parse_args()
    
    setup_logging('logs')
    
    raw_root = Path(args.raw_dir)
    output_root = Path(args.output_dir)
    
    # Find all subdirectories in raw (e.g., sport, music, etc.)
    subdirs = [d for d in raw_root.iterdir() if d.is_dir()]
    
    if not subdirs:
        logger.error(f"No subdirectories found in {raw_root}")
        return
    
    for subdir in subdirs:
        logger.info(f"Processing {subdir.name}...")
        
        # Create output directories
        output_audio_dir = output_root / subdir.name / 'audio'
        output_text_dir = output_root / subdir.name / 'text'
        output_audio_dir.mkdir(parents=True, exist_ok=True)
        output_text_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all audio files
        audio_files = (
            list(subdir.glob('*.mp3')) +
            list(subdir.glob('*.wav')) +
            list(subdir.glob('*.flac'))
        )
        
        logger.info(f"Found {len(audio_files)} audio files in {subdir.name}")
        
        for audio_path in audio_files:
            # Find corresponding text file
            text_path = subdir / f"{audio_path.stem}.txt"
            if not text_path.exists():
                logger.warning(f"No text file for {audio_path.name}, skipping")
                continue
            
            logger.info(f"Chunking {audio_path.name}...")
            chunk_audio_and_text(
                audio_path,
                text_path,
                output_audio_dir,
                output_text_dir,
                chunk_duration=args.chunk_duration,
                sr=args.sr
            )
    
    logger.info("Chunking complete!")


if __name__ == '__main__':
    main()
