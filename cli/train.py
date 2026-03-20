#!/usr/bin/env python3
"""
CLI script for training ASR model.

Usage:
    python cli/train.py --data_dir data/raw --epochs 5 --batch_size 4
"""

import sys
import logging
import argparse
import json
from pathlib import Path

import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

# Add src to path so we can import the library
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    WhisperDataCollator,
    create_dataloaders,
    load_config,
    save_config,
    setup_logging,
    set_seed,
    get_device,
    count_parameters,
    get_model_registry,
)

logger = logging.getLogger(__name__)


def compute_metrics(pred, tokenizer):
    """Compute WER for evaluation."""
    import evaluate
    wer = evaluate.load('wer')
    cer = evaluate.load('cer')
    
    pred_logits = pred.predictions
    label_ids = pred.label_ids

    # Helper: convert a single element (tensor/ndarray/list) to numpy
    def as_numpy(x):
        try:
            import torch as _torch
            if isinstance(x, _torch.Tensor):
                return x.cpu().numpy()
        except Exception:
            pass
        try:
            return np.array(x)
        except Exception:
            return np.array(list(x))

    # Normalize label ids to list of int lists and replace -100
    labels_list = []
    if label_ids is None:
        labels_list = []
    elif isinstance(label_ids, (list, tuple)):
        for l in label_ids:
            arr = as_numpy(l)
            arr = arr.astype(int)
            arr[arr == -100] = tokenizer.pad_token_id
            labels_list.append(arr.tolist())
    else:
        arr = as_numpy(label_ids)
        if arr.ndim == 2:
            for row in arr:
                row = row.astype(int)
                row[row == -100] = tokenizer.pad_token_id
                labels_list.append(row.tolist())
        else:
            arr = arr.astype(int)
            arr[arr == -100] = tokenizer.pad_token_id
            labels_list.append(arr.tolist())

    # Normalize predictions to list of int lists
    preds_list = []
    if pred_logits is None:
        preds_list = []
    elif isinstance(pred_logits, (list, tuple)):
        for p in pred_logits:
            arr = as_numpy(p)
            if arr.ndim == 3:
                ids = np.argmax(arr, axis=-1)
                # ids may be (1, seq) or (seq_len,)
                for row in ids:
                    preds_list.append(row.astype(int).tolist())
            elif arr.ndim == 2:
                for row in arr:
                    preds_list.append(row.astype(int).tolist())
            elif arr.ndim == 1:
                preds_list.append(arr.astype(int).tolist())
            else:
                preds_list.append(arr.flatten().astype(int).tolist())
    else:
        arr = as_numpy(pred_logits)
        if arr.ndim == 3:
            ids = np.argmax(arr, axis=-1)
            for row in ids:
                preds_list.append(row.astype(int).tolist())
        elif arr.ndim == 2:
            for row in arr:
                preds_list.append(row.astype(int).tolist())
        elif arr.ndim == 1:
            preds_list.append(arr.astype(int).tolist())
        else:
            preds_list.append(arr.flatten().astype(int).tolist())

    # Ensure same length
    n_preds = len(preds_list)
    n_labels = len(labels_list)
    n = min(n_preds, n_labels) if n_preds and n_labels else max(n_preds, n_labels)

    # Decode in small chunks to avoid high memory usage
    pred_str = []
    label_str = []
    chunk = 16
    for i in range(0, n, chunk):
        block_preds = preds_list[i : i + chunk]
        block_labels = labels_list[i : i + chunk]
        # If lengths mismatch, pad with empty lists
        if len(block_preds) < len(block_labels):
            block_preds.extend([[]] * (len(block_labels) - len(block_preds)))
        if len(block_labels) < len(block_preds):
            block_labels.extend([[]] * (len(block_preds) - len(block_labels)))
        pred_str.extend(tokenizer.batch_decode(block_preds, skip_special_tokens=True))
        label_str.extend(tokenizer.batch_decode(block_labels, skip_special_tokens=True))
    
    # Compute metrics
    wer_score = wer.compute(predictions=pred_str, references=label_str)
    cer_score = cer.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer_score, "cer": cer_score}


def compute_ctc_metrics(pred, processor):
    """Compute WER/CER for CTC models such as Wav2Vec2.
    
    batch_decode() automatically filters PAD tokens (CTC blank).
    """
    import evaluate

    wer = evaluate.load('wer')
    cer = evaluate.load('cer')

    pred_logits = pred.predictions
    label_ids = pred.label_ids

    if isinstance(pred_logits, tuple):
        pred_logits = pred_logits[0]

    pred_ids = np.argmax(pred_logits, axis=-1)
    label_ids = np.array(label_ids)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # batch_decode() automatically filters pad_token_id (CTC blank)
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(label_ids)

    wer_score = wer.compute(predictions=pred_str, references=label_str)
    cer_score = cer.compute(predictions=pred_str, references=label_str)

    return {"wer": wer_score, "cer": cer_score}


def create_wav2vec2_processor_from_data(data_dir: str, model_name: str, output_dir: str):
    """
    Create Wav2Vec2Processor with vocabulary extracted from training data.

    Builds the character-level vocabulary from *preprocessed* text so that
    the tokenizer matches exactly what the model sees during training.
    The pipe character ``|`` is used as CTC word delimiter (represents spaces).

    Args:
        data_dir: Directory containing text transcription files
        model_name: Pretrained model name (for feature extractor)
        output_dir: Directory to save vocabulary

    Returns:
        Wav2Vec2Processor with custom vocabulary
    """
    from src.utils.text_preprocessing import SerbianTextPreprocessor

    logger.info("Extracting vocabulary from training data (preprocessed)...")
    preproc = SerbianTextPreprocessor()

    # Extract all unique characters from preprocessed text
    chars: set[str] = set()
    data_path = Path(data_dir)

    for txt_file in data_path.rglob("*.txt"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                raw = f.read().strip()
                text = preproc.preprocess(raw)
                chars.update(text)
        except Exception as e:
            logger.warning(f"Failed to read {txt_file}: {e}")

    # Remove whitespace-class characters; space is encoded as "|" word delimiter
    chars.discard('\n')
    chars.discard('\r')
    chars.discard('\t')
    chars.discard(' ')  # space is represented by | in CTC vocabulary

    # Create vocabulary: sorted list of unique characters
    vocab_list = sorted(list(chars))

    # Reserve special tokens: [PAD]=0 (also CTC blank), [UNK]=1, "|"=2 (word delimiter / space)
    vocab_dict: dict[str, int] = {"[PAD]": 0, "[UNK]": 1, "|": 2}
    for idx, ch in enumerate(vocab_list, start=3):
        if ch not in vocab_dict:  # safety guard against duplicates
            vocab_dict[ch] = idx

    logger.info(f"Vocabulary size: {len(vocab_dict)} characters")
    logger.info(f"Sample characters: {list(vocab_dict.keys())[:20]}")

    # Save vocabulary to file
    vocab_path = Path(output_dir) / "vocab.json"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)

    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

    logger.info(f"Vocabulary saved to {vocab_path}")

    # Create tokenizer with custom vocabulary
    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )

    # Load feature extractor from pretrained model
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

    # Create processor
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    logger.info("Wav2Vec2Processor created with custom vocabulary")

    return processor


def setup_arg_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Train ASR model for Serbian",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Data directory with audio files and transcriptions",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/whisper",
        help="Output directory for checkpoints",
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/whisper-base",
        help="Pretrained model name",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="whisper",
        help="Model type from registry",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size",
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Warmup steps",
    )
    
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps to simulate larger batch sizes",
    )
    
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 training",
    )
    
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Log directory",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    parser.add_argument(
        "--unfreeze_encoder",
        action="store_true",
        help="Unfreeze encoder parameters for full fine-tuning",
    )
    
    parser.add_argument(
        "--freeze_n_layers",
        type=int,
        default=0,
        help="Freeze first N transformer encoder layers (wav2vec2). "
             "Recommended: 8 for wav2vec2-base (12 layers), 20 for xls-r-300m (24 layers).",
    )
    
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for efficient fine-tuning",
    )
    
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    
    return parser


def main():
    """CLI entrypoint for training."""
    parser = setup_arg_parser()
    args = parser.parse_args()

    setup_logging('logs')

    # Load config
    config = load_config()
    
    # Override with CLI args
    config.update(vars(args))

    # Reproducibility
    set_seed(config['seed'])
    
    logger.info('=' * 80)
    logger.info('SERBIAN ASR MODEL TRAINING')
    logger.info('=' * 80)
    logger.info(f'Model type: {config["model_type"]}')
    logger.info(f'Model: {config["model_name"]}')
    logger.info(f'Data directory: {config["data_dir"]}')
    logger.info(f'Output directory: {config["output_dir"]}')
    logger.info(f'Epochs: {config["epochs"]}')
    logger.info(f'Batch size: {config["batch_size"]}')
    logger.info(f'Learning rate: {config["learning_rate"]}')
    logger.info(f'FP16: {config["fp16"]}')
    logger.info('=' * 80)
    
    # Load model and processor
    logger.info(f'Loading model {config["model_name"]}...')
    registry = get_model_registry()
    
    # For wav2vec2, create processor with vocabulary from data first
    processor_for_adapter = None
    if config['model_type'].lower() == 'wav2vec2':
        processor_for_adapter = create_wav2vec2_processor_from_data(
            data_dir=config['data_dir'],
            model_name=config['model_name'],
            output_dir=config['output_dir'],
        )
    
    adapter = registry.create(
        config['model_type'],
        model_name_or_path=config['model_name'],
        language='sr',
        freeze_encoder=False,
        freeze_n_layers=config.get('freeze_n_layers', 0),
        processor=processor_for_adapter,
    )
    processor = adapter.get_processor()
    model = adapter.unwrap()
    
    # Apply LoRA for efficient tuning if requested (Whisper path)
    if config.get('use_lora', False) and config['model_type'].lower() == 'whisper':
        from peft import LoraConfig, get_peft_model
        logger.info('Applying LoRA for parameter-efficient fine-tuning')
        lora_config = LoraConfig(
            r=16,  # Low-rank dimension
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],  # Whisper attention and FFN
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Unfreeze encoder if requested (overrides LoRA for full tuning)
    if config.get('unfreeze_encoder', False):
        logger.info('Unfreezing encoder for full fine-tuning')
        # More robust: enable grad for any parameter whose name contains 'encoder'
        found = 0
        for name, param in model.named_parameters():
            if 'encoder' in name:
                param.requires_grad = True
                found += 1

        if found == 0:
            # fallback: try searching for common encoder module names in nested base_model
            base = getattr(model, 'base_model', None) or getattr(model, 'model', None)
            if base is not None:
                for name, param in base.named_parameters():
                    if 'encoder' in name:
                        param.requires_grad = True
                        found += 1

        if found == 0:
            logger.warning('Could not locate encoder parameters by name; leaving default trainable params')
        else:
            logger.info(f'Unfroze {found} encoder parameters by name match')

        params = count_parameters(model)
        logger.info(f'Trainable parameters: {params["trainable"]:,}')
    else:
        logger.info('Training full model (encoder + decoder)')
        params = count_parameters(model)
        logger.info(f'Total parameters: {params["total"]:,}')
        logger.info(f'Trainable parameters: {params["trainable"]:,}')
    
    # Enable gradient checkpointing to reduce memory usage
    if config['model_type'].lower() == 'wav2vec2':
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info('Enabled gradient checkpointing for memory efficiency')
        else:
            logger.warning('Model does not support gradient checkpointing')
    
    device = get_device()
    adapter.to(device)

    # Persist run config for reproducibility
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, str(output_dir / 'run_config.yaml'))
    
    # Load dataset
    logger.info('Loading dataset...')
    train_loader, val_loader, test_loader = create_dataloaders(
        config['data_dir'],
        processor,
        batch_size=config['batch_size']
    )
    
    if config['model_type'].lower() == 'whisper':
        training_args = Seq2SeqTrainingArguments(
            output_dir=config['output_dir'],
            per_device_train_batch_size=config['batch_size'],
            per_device_eval_batch_size=config['batch_size'],
            gradient_accumulation_steps=config['gradient_accumulation_steps'],
            learning_rate=config['learning_rate'],
            warmup_steps=config['warmup_steps'],
            max_steps=-1,
            num_train_epochs=config['epochs'],
            eval_strategy='epoch',
            save_strategy='epoch',
            predict_with_generate=True,
            generation_max_length=256,
            logging_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='wer',
            greater_is_better=False,
            fp16=config['fp16'],
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            label_names=['labels'],
        )

        data_collator = WhisperDataCollator(processor)
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_loader.dataset,
            eval_dataset=val_loader.dataset,
            data_collator=data_collator,
            compute_metrics=lambda pred: compute_metrics(pred, processor.tokenizer),
        )
    else:
        training_args = TrainingArguments(
            output_dir=config['output_dir'],
            per_device_train_batch_size=config['batch_size'],
            per_device_eval_batch_size=config['batch_size'],
            gradient_accumulation_steps=config['gradient_accumulation_steps'],
            learning_rate=config['learning_rate'],
            warmup_steps=config['warmup_steps'],
            num_train_epochs=config['epochs'],
            eval_strategy='epoch',
            save_strategy='epoch',
            logging_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='wer',
            greater_is_better=False,
            fp16=config['fp16'],
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            label_names=['labels'],
        )

        data_collator = WhisperDataCollator(processor)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_loader.dataset,
            eval_dataset=val_loader.dataset,
            data_collator=data_collator,
            compute_metrics=lambda pred: compute_ctc_metrics(pred, processor),
        )
    
    # Train
    logger.info('Starting training...')
    logger.info('=' * 80)
    trainer.train(resume_from_checkpoint=config.get('resume_from_checkpoint'))
    
    # Save final model
    final_model_dir = Path(config['output_dir']) / 'final'
    final_model_dir.mkdir(exist_ok=True)
    model.save_pretrained(final_model_dir)
    processor.save_pretrained(final_model_dir)
    logger.info(f'Final model saved to: {final_model_dir}')


if __name__ == '__main__':
    main()
