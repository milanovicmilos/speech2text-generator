#!/usr/bin/env python3
"""
Show 10 random test samples and run evaluation with beam search.
"""
from pathlib import Path
import random
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from src.data.data_loader import create_dataloaders
from src.utils.text_preprocessing import SerbianTextPreprocessor
import evaluate


def main():
    print('Loading model and data...')
    processor = WhisperProcessor.from_pretrained('models/whisper/final')
    model = WhisperForConditionalGeneration.from_pretrained('models/whisper/final')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_loader, val_loader, test_loader = create_dataloaders('data/raw', processor, batch_size=1)
    test_dataset = test_loader.dataset
    preproc = SerbianTextPreprocessor()

    # 1) Show 10 random samples
    n = len(test_dataset)
    indices = random.sample(range(n), min(10, n))
    print('\n=== 10 RANDOM TEST SAMPLES ===')
    for i in indices:
        item = test_dataset[i]
        audio_path = getattr(test_dataset, 'audio_files', None)
        valid_idx = getattr(test_dataset, 'valid_indices', None)
        if audio_path is not None and valid_idx is not None:
            idx = valid_idx[i]
            path = test_dataset.audio_files[idx]
        else:
            path = 'N/A'
        inp = item['input_features'].unsqueeze(0).to(device)
        attn = item.get('attention_mask')
        if attn is not None:
            attn = attn.unsqueeze(0).to(device)
        with torch.no_grad():
            # force transcription task and Serbian language to avoid language-detection artifacts
            try:
                gen = model.generate(inp, attention_mask=attn, num_beams=1, task="transcribe", language="sr")
            except TypeError:
                gen = model.generate(inp, task="transcribe", language="sr")
        dec = processor.batch_decode(gen, skip_special_tokens=True)[0]
        ref = item.get('text', '')
        print('---')
        print('audio:', path)
        print('ref  :', preproc.preprocess(ref))
        print('pred :', preproc.preprocess(dec))

    # 2) Full evaluation: try a small sweep of generation params (beams, no_repeat_ngram)
    wer = evaluate.load('wer')
    cer = evaluate.load('cer')

    gen_configs = [
        {'num_beams': 1, 'no_repeat_ngram_size': 0, 'early_stopping': False},
        {'num_beams': 2, 'no_repeat_ngram_size': 2, 'early_stopping': True},
        {'num_beams': 4, 'no_repeat_ngram_size': 3, 'early_stopping': True},
        {'num_beams': 8, 'no_repeat_ngram_size': 3, 'early_stopping': True},
        # Add repetition and length penalties to combat repetitions / garbage
        {'num_beams': 4, 'no_repeat_ngram_size': 3, 'early_stopping': True, 'repetition_penalty': 1.2, 'length_penalty': 1.0},
        {'num_beams': 4, 'no_repeat_ngram_size': 3, 'early_stopping': True, 'repetition_penalty': 1.5, 'length_penalty': 0.8},
        {'num_beams': 6, 'no_repeat_ngram_size': 4, 'early_stopping': True, 'repetition_penalty': 1.3, 'length_penalty': 1.0},
    ]

    results = []
    for cfg in gen_configs:
        print(f"\n=== EVAL: num_beams={cfg['num_beams']}, no_repeat_ngram={cfg['no_repeat_ngram_size']} ===")
        all_preds = []
        all_refs = []
        for batch in test_loader:
            inp = batch['input_features'].to(device)
            attn = batch.get('attention_mask')
            if attn is not None:
                attn = attn.to(device)
            with torch.no_grad():
                try:
                    gen_kwargs = dict(
                        inp=inp,
                        attention_mask=attn,
                        task="transcribe",
                        language="sr",
                        num_beams=cfg['num_beams'],
                        no_repeat_ngram_size=cfg['no_repeat_ngram_size'],
                        early_stopping=cfg['early_stopping'],
                        temperature=0.0,
                        max_new_tokens=128,
                        max_length=256,
                    )
                    # optional keys
                    if 'repetition_penalty' in cfg:
                        gen_kwargs['repetition_penalty'] = cfg['repetition_penalty']
                    if 'length_penalty' in cfg:
                        gen_kwargs['length_penalty'] = cfg['length_penalty']

                    gen = model.generate(**gen_kwargs)
                except Exception:
                    gen = model.generate(inp, attention_mask=attn)
            decs = processor.batch_decode(gen, skip_special_tokens=True)
            for i, d in enumerate(decs):
                pred_pp = preproc.preprocess(d)
                pred_pp = preproc.postprocess_prediction(pred_pp)
                all_preds.append(pred_pp)
                all_refs.append(preproc.preprocess(batch.get('text', [''])[i]))

        wer_val = wer.compute(predictions=all_preds, references=all_refs)
        cer_val = cer.compute(predictions=all_preds, references=all_refs)
        print('Computed', len(all_preds), 'predictions')
        print('WER:', wer_val)
        print('CER:', cer_val)
        results.append((cfg, wer_val, cer_val))

    # print best result
    best = min(results, key=lambda x: x[1] if isinstance(x[1], float) else float(x[1]))
    print('\n=== BEST ===')
    print(f"cfg={best[0]}, WER={best[1]}, CER={best[2]}")


if __name__ == '__main__':
    main()
