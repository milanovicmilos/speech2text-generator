#!/usr/bin/env python3
"""
Quick debug script to see what wav2vec2 model is actually predicting.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    load_config,
    setup_logging,
    set_seed,
    get_device,
    get_model_registry,
)
from src.data.data_loader import create_dataloaders

logger = None

def debug_predictions(model_dir: str, data_dir: str, num_examples: int = 5):
    """Debug: print raw model outputs and references."""
    global logger
    
    setup_logging('logs')
    logger = __import__('logging').getLogger(__name__)
    
    device = get_device()
    
    # Load model and processor directly from disk (Wav2Vec2)
    logger.info(f"Loading model from {model_dir}...")
    from transformers import AutoModelForCTC, Wav2Vec2Processor
    
    model = AutoModelForCTC.from_pretrained(model_dir).to(device)
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model.eval()
    
    # Load data (use validation set, not test - that's what trainer evaluates on)
    logger.info(f"Loading dataset from {data_dir}...")
    _, val_loader, _ = create_dataloaders(data_dir, processor, batch_size=1)
    test_loader = val_loader  # Use validation set for debugging
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Showing {num_examples} predictions:")
    logger.info(f"{'='*80}\n")
    
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if idx >= num_examples:
                break
            
            # Get inputs and labels
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            logits = model(input_values).logits
            
            # CTC decode (argmax)
            pred_ids = torch.argmax(logits, dim=-1)
            
            # Clean up labels (replace -100 with pad_token_id)
            label_clean = labels.clone()
            label_clean[label_clean == -100] = processor.tokenizer.pad_token_id
            
            # Decode
            pred_str = processor.tokenizer.decode(pred_ids[0].cpu().numpy())
            label_str = processor.tokenizer.decode(label_clean[0].cpu().numpy())
            
            # Debug: show token counts
            pred_ids_list = pred_ids[0].cpu().numpy().tolist()
            unique_pred_tokens, counts = np.unique(pred_ids_list, return_counts=True)
            
            logger.info(f"Example {idx + 1}:")
            logger.info(f"  Predicted:  '{pred_str}'")
            logger.info(f"  Reference:  '{label_str}'")
            logger.info(f"  Pred len: {len(pred_str)}, Ref len: {len(label_str)}")
            logger.info(f"  Logits shape: {logits.shape}, Max logit: {logits.max().item():.2f}, Min logit: {logits.min().item():.2f}")
            logger.info(f"  Unique tokens in pred: {list(zip(unique_pred_tokens, counts))}")
            logger.info(f"  Token meanings: 0=[PAD], 1=[UNK], 2=|")
            logger.info("")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='models/wav2vec2/final')
    parser.add_argument('--data_dir', type=str, default='data/raw')
    parser.add_argument('--num_examples', type=int, default=5, help="Number of examples (0 for all)")
    args = parser.parse_args()
    
    num = None if args.num_examples == 0 else args.num_examples
    debug_predictions(args.model_dir, args.data_dir, num)
