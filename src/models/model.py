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
        
        # Load pretrained Whisper model
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            attn_implementation="sdpa"  # Scaled dot-product attention
        )
        
        # Load processor (handles feature extraction and tokenization)
        self.processor = WhisperProcessor.from_pretrained(model_name)
        
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
        max_length: int = 225,
        num_beams: int = 1,
        temperature: float = 0.0,
        language: str = "Serbian",
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
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language,
            task="transcribe"
        )
        
        # Generate
        predicted_ids = self.model.generate(
            input_features=input_features,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            forced_decoder_ids=forced_decoder_ids,
        )
        
        return predicted_ids
    
    def transcribe(
        self,
        audio_array: torch.Tensor,
        sampling_rate: int = 16000,
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
        
        # Generate
        with torch.no_grad():
            predicted_ids = self.generate(inputs.input_features.to(self.model.device))
        
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
