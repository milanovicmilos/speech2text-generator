# GPU Setup Instructions for Whisper Serbian ASR

## Prerequisites
- CUDA-compatible GPU (NVIDIA)
- CUDA Toolkit installed (version 11.8+ recommended)
- Python 3.8-3.11
- At least 16GB RAM, 32GB preferred

## Setup Steps

1. **Clone and setup environment** (same as CPU setup)
   ```bash
   git clone <repo>
   cd speech_recognation
   python -m venv .venv
   source .venv/Scripts/activate  # Windows
   pip install -r requirements.txt
   ```

2. **Verify GPU availability**
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
   ```

3. **Run GPU training**
   ```bash
   bash gpu_train.sh
   ```

## Expected Results
- Current WER on CPU: 0.967 (after sentence-based chunk splitting)
- With GPU training (3 epochs unfrozen), expect WER <0.5
- Training should complete in ~1-2 hours per epoch
- Model saved to `models/whisper_gpu/final`

## Troubleshooting
- If CUDA out of memory, reduce batch_size or gradient_accumulation_steps
- Ensure PyTorch is installed with CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`