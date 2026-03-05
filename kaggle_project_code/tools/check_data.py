#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from transformers import WhisperProcessor
from src.data.data_loader import create_dataloaders
import librosa

processor = WhisperProcessor.from_pretrained('models/whisper/final')
train_loader, val_loader, test_loader = create_dataloaders('data/raw', processor, batch_size=1)

def inspect_loader(loader, name, n=10):
    print(f"== Inspecting {name} (showing up to {n} samples) ==")
    ds = loader.dataset
    for i in range(min(n, len(ds))):
        item = ds[i]
        audio_idx = ds.dataset.valid_indices[ds.indices[i]] if hasattr(ds, 'indices') else ds.valid_indices[i]
        audio_path = ds.dataset.audio_files[audio_idx] if hasattr(ds, 'dataset') else ds.audio_files[audio_idx]
        try:
            dur = librosa.get_duration(path=str(audio_path))
        except Exception:
            dur = None
        text = item.get('text', '')
        labels = item.get('labels')
        label_len = len(labels) if labels is not None else 0
        decoded = processor.tokenizer.decode(labels.tolist(), skip_special_tokens=True) if labels is not None else ''
        inp = item['input_features']
        print(f"#{i}: path={audio_path.name}, dur={dur}, words={len(text.split())}, label_len={label_len}, decoded_sample='{decoded[:80]}'")
        print(f"     input_features.shape={tuple(inp.shape)}")

inspect_loader(test_loader, 'test', n=10)
inspect_loader(val_loader, 'val', n=3)
inspect_loader(train_loader, 'train', n=3)
print('Done')
