#!/usr/bin/env python3
"""
CLI script for filling empty chunk text files from raw transcripts.

Usage:
    python cli/fill_empty_chunks.py --base_dir data/chunked/sport
"""

import sys
import logging
import argparse
import re
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fill empty chunk text files from raw transcripts")
    parser.add_argument('--base_dir', type=str, default='data/chunked/sport')
    
    args = parser.parse_args()
    
    setup_logging('logs')
    
    base = Path(args.base_dir)
    audio_sub = base / 'audio'
    text_sub = base / 'text'
    raw_dir = base.parents[1] / 'raw' if len(base.parents) > 1 else None
    logger.info('Filling empty texts from raw for %s', base)
    
    for f in sorted([p for p in audio_sub.glob('*.*') if p.suffix.lower() in ('.wav','.mp3','.flac')]):
        stem = f.stem
        txt = text_sub / f'{stem}.txt'
        if txt.exists() and txt.read_text(encoding='utf-8').strip():
            continue
        m_local = re.match(r'(?P<base>.+?)_chunk\d+$', stem)
        base_key = m_local.group('base') if m_local else stem
        raw_candidate = next((p for p in raw_dir.rglob(f'{base_key}.txt')), None) if raw_dir else None
        if raw_candidate and raw_candidate.exists():
            txt.write_text(raw_candidate.read_text(encoding='utf-8'), encoding='utf-8')
        else:
            non_empty = next((p for p in text_sub.glob('*.txt') if p.read_text(encoding='utf-8').strip()), None)
            if non_empty:
                txt.write_text(non_empty.read_text(encoding='utf-8'), encoding='utf-8')
    
    logger.info('Fill-empty complete')


if __name__ == '__main__':
    main()