#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from transformers import WhisperProcessor
from src.data.data_loader import WhisperSpeechDataset
import librosa

proc = WhisperProcessor.from_pretrained('models/whisper/final')
wd = 'data/raw'
print('Loading dataset...')
ds = WhisperSpeechDataset(wd, wd, proc)

print('Total audio files:', len(ds.audio_files))
print('Valid indices count:', len(ds.valid_indices))

violations = []
for idx, audio_path in enumerate(ds.audio_files):
    stem = audio_path.stem
    text = ds.transcriptions.get(stem, '')
    words = len(text.split())
    try:
        dur = librosa.get_duration(path=str(audio_path))
    except Exception:
        dur = 0.0
    wps = words / dur if dur>0 else 0.0
    if not (0.3 <= wps <= 5.0) or words < 5 or dur < 5.0:
        violations.append({'audio': str(audio_path), 'words': words, 'dur': dur, 'wps': wps})

print('Violations found:', len(violations))
for v in violations[:30]:
    print(v)

# Check if any violating files are still in valid_indices
still_included = []
for i in ds.valid_indices:
    p = ds.audio_files[i]
    for v in violations:
        if str(p) == v['audio']:
            still_included.append(v)

print('Violations still included in valid_indices:', len(still_included))
for v in still_included[:30]:
    print(v)

print('Done')
