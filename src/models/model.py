"""
Whisper-based ASR model for Serbian language.

This is a much better approach than the previous seq2seq model:
- Uses CTC-like training (Whisper uses attention but with better alignment)
- Automatically learns alignment between audio and text
- Pretrained on 680k hours of multilingual data
- Fine-tuning requires only hours of data, not thousands

Architecture:
    Audio (WAV) → Whisper Encoder → Cross-Attention Decoder → Text

"""

import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperConfig
)

logger = logging.getLogger(__name__)


class WhisperASRModel(nn.Module):
    """
    Whisper-based ASR model for Serbian.
    
    This wraps the HuggingFace Whisper model with additional configuration
    for Serbian speech recognition.
    """
    
    def __init__(
        self,
        model_name: str = "openai/whisper-base",
        language: str = "Serbian",
        freeze_encoder: bool = True,
        freeze_n_layers: int = 0,
    ):
        """
        Initialize Whisper ASR model.
        
        Args:
            model_name: Whisper model name (tiny, base, small, medium, large)
            language: Target language (Serbian)
            freeze_encoder: Whether to freeze encoder parameters
            freeze_n_layers: Number of encoder layers to freeze (0 = freeze all)
        """
        super().__init__()
        
        self.model_name = model_name
        self.language = language
        
        # Default decoding settings (tuned to reduce repetitions)
        self.default_gen_kwargs: Dict[str, Any] = dict(
            num_beams=8,
            no_repeat_ngram_size=10,
            repetition_penalty=5.0,
            length_penalty=1.0,
            early_stopping=True,
            temperature=0.0,
            max_new_tokens=128,
            max_length=256,
            language=self.language,
            suppress_tokens=None,
            begin_suppress_tokens=None,
        )

        # Load pretrained Whisper model
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            attn_implementation="sdpa"  # Scaled dot-product attention
        )

        # Load processor (handles feature extraction and tokenization)
        self.processor = WhisperProcessor.from_pretrained(model_name)

        # Disable forced decoder ids and suppress tokens to let custom generation params drive decoding
        try:
            if hasattr(self.model, "generation_config"):
                self.model.generation_config.forced_decoder_ids = None
            self.model.generation_config.suppress_tokens = None
            self.model.generation_config.begin_suppress_tokens = None
        except Exception:
            pass
        try:
            if hasattr(self.model, "config"):
                self.model.config.forced_decoder_ids = None
        except Exception:
            pass
        
        logger.info(f"Loaded Whisper model: {model_name}")
        logger.info(f"Processor sample rate: {self.processor.feature_extractor.sampling_rate}")
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.model.model.encoder.parameters():
                param.requires_grad = False
            logger.info("Froze all encoder parameters")
        elif freeze_n_layers > 0:
            # Freeze only first N layers
            for layer in self.model.model.encoder.layers[:freeze_n_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
            logger.info(f"Froze first {freeze_n_layers} encoder layers")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def forward(
        self,
        input_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_features: Log-mel spectrogram features [batch, n_mels, seq_len]
            labels: Token IDs for training [batch, max_label_length]
            attention_mask: Attention mask for input features
            
        Returns:
            Dictionary with loss and logits
        """
        outputs = self.model(
            input_features=input_features,
            labels=labels,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }
    
    def generate(
        self,
        input_features: torch.Tensor,
        max_length: int = 256,
        max_new_tokens: int = 128,
        num_beams: int = 8,
        no_repeat_ngram_size: int = 10,
        repetition_penalty: float = 5.0,
        length_penalty: float = 1.0,
        temperature: float = 0.0,
        early_stopping: bool = True,
        language: str = "Serbian",
        suppress_tokens: Optional[list] = None,
        begin_suppress_tokens: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Generate transcription from audio features.
        
        Args:
            input_features: Log-mel spectrogram features
            max_length: Maximum output length
            num_beams: Number of beams for beam search
            temperature: Temperature for sampling (0 = greedy)
            language: Language for generation
            
        Returns:
            Generated token IDs
        """
        # Get forced decoder IDs for Serbian
        # Some checkpoints persist forced ids in configs; ensure they are off
        self.model.generation_config.forced_decoder_ids = None

        predicted_ids = self.model.generate(
            input_features=input_features,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            temperature=temperature,
            language=language,
            task="transcribe",
            suppress_tokens=suppress_tokens,
            begin_suppress_tokens=begin_suppress_tokens,
        )
        
        return predicted_ids
    
    def transcribe(
        self,
        audio_array: torch.Tensor,
        sampling_rate: int = 16000,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio_array: Audio waveform [n_samples]
            sampling_rate: Sample rate of audio
            
        Returns:
            Transcribed text
        """
        # Process audio
        inputs = self.processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )
        
        gen_kwargs = {**self.default_gen_kwargs, **(generation_kwargs or {})}

        # Generate
        with torch.no_grad():
            predicted_ids = self.generate(inputs.input_features.to(self.model.device), **gen_kwargs)
        
        # Decode
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        return transcription.strip()
    
    def get_processor(self) -> WhisperProcessor:
        """Get the processor for handling audio and text."""
        return self.processor
    
    def get_model(self) -> WhisperForConditionalGeneration:
        """Get the underlying Whisper model."""
        return self.model


def create_whisper_model(config: Dict[str, Any]) -> WhisperASRModel:
    """
    Create Whisper ASR model from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Whisper ASR model
    """
    model_config = config.get("model", {})
    
    model = WhisperASRModel(
        model_name=model_config.get("pretrained_model", "openai/whisper-base"),
        language=model_config.get("language", "Serbian"),
        freeze_encoder=model_config.get("freeze_encoder", True),
        freeze_n_layers=model_config.get("freeze_n_layers", 0),
    )
    
    return model
