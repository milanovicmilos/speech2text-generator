#!/usr/bin/env python3
"""
CLI script for repopulating ASR chunked text files (word-based proportional).

Usage:
    python cli/repopulate_chunks.py --base_dir data/chunked
"""

import sys
import logging
import argparse
import re
from pathlib import Path
import librosa

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import setup_logging

logger = logging.getLogger(__name__)


def split_words(text):
    words = re.findall(r"\S+", text)
    return words


def main():
    parser = argparse.ArgumentParser(description="Repopulate ASR chunked text files (word-based proportional)")
    parser.add_argument('--base_dir', type=str, default='data/chunked', help='Chunk root containing audio/ and text/ folders')
    
    args = parser.parse_args()
    
    setup_logging('logs')
    
    base = Path(args.base_dir)
    audio_sub = base / 'audio'
    text_sub = base / 'text'
    logger.info('Repopulating chunked texts (word-based proportional) for %s', base)
    
    files = sorted([p for p in audio_sub.glob('*.*') if p.suffix.lower() in ('.wav','.mp3','.flac')])
    m = re.compile(r'(?P<base>.+?)_chunk\d+$')
    groups = {}
    for f in files:
        mm = m.match(f.stem)
        key = mm.group('base') if mm else f.stem
        groups.setdefault(key, []).append(f)
    
    raw_root = base.parents[1] / 'raw' if len(base.parents) > 1 else None
    raw_index = {p.stem: p for p in (raw_root.rglob('*.txt') if raw_root and raw_root.exists() else [])}
    
    for base_stem, group_files in groups.items():
        group_files = sorted(group_files)
        raw_src = raw_index.get(base_stem)
        if not raw_src:
            continue
        text = raw_src.read_text(encoding='utf-8')
        words = split_words(text)
        durs = [librosa.get_duration(path=str(f)) for f in group_files]
        total_dur = sum(durs) or 1.0
        target_words = [max(1, int(round((d/total_dur)*len(words)))) for d in durs]
        si = 0
        for i, f in enumerate(group_files):
            take = target_words[i]
            part = ' '.join(words[si:si+take]) if si < len(words) else ''
            si += take
            out = text_sub / f'{f.stem}.txt'
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(part or ' '.join(words), encoding='utf-8')
    
    logger.info('Repopulation complete')


if __name__ == '__main__':
    main()