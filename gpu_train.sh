#!/bin/bash
# GPU Training Script for Whisper Serbian ASR
# Run this on a machine with CUDA GPU

# Activate virtual environment (adjust path if needed)
source .venv/Scripts/activate  # On Windows, or source venv/bin/activate on Linux

# Install requirements if not done
pip install -r requirements.txt

# Run training with unfrozen encoder for better results, continuing from current model
python cli/train.py train \
  --model_name models/whisper_quick_saved \
  --epochs 3 \
  --output_dir models/whisper_gpu \
  --unfreeze_encoder \
  --batch_size 8 \
  --gradient_accumulation_steps 2 \
  --fp16 \
  --learning_rate 5e-6 \
  --warmup_steps 200

# After training, evaluate
python cli/train.py eval --model_dir models/whisper_gpu/final

echo "Training complete. Check models/whisper_gpu/final for the model."