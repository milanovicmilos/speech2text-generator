#!/bin/bash
# GPU Training Script for Whisper Serbian ASR

# Activate virtual environment if needed
# Linux/macOS example:
# source .venv/bin/activate

pip install -r requirements.txt

# Train
python cli/train.py \
  --data_dir data/raw \
  --model_name openai/whisper-base \
  --epochs 3 \
  --output_dir models/whisper_gpu \
  --unfreeze_encoder \
  --batch_size 8 \
  --gradient_accumulation_steps 2 \
  --fp16 \
  --learning_rate 5e-6 \
  --warmup_steps 200 \
  --seed 42

# Evaluate final checkpoint
python cli/eval_model.py \
  --data_dir data/raw \
  --model_dir models/whisper_gpu/final \
  --split test \
  --batch_size 8

echo "Training complete. Model path: models/whisper_gpu/final"