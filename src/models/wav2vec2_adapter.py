"""Wav2Vec2 adapter implementation conforming to generic ASR model contract."""

from __future__ import annotations

from typing import Any, Optional

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from .contracts import ASRModelAdapter, GenerationConfig


class Wav2Vec2ASRAdapter(ASRModelAdapter):
    """Wav2Vec2 CTC adapter for Serbian ASR use case."""

    def __init__(
        self,
        model_name_or_path: str,
        language: str = "sr",
        freeze_encoder: bool = False,
        freeze_n_layers: int = 0,
        processor: Optional[Wav2Vec2Processor] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.language = language

        self._model = Wav2Vec2ForCTC.from_pretrained(model_name_or_path)
        
        # Use provided processor or load from pretrained
        if processor is not None:
            self._processor = processor
        else:
            self._processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
        
        # Sync pad_token_id for CTC blank token
        self._model.config.pad_token_id = self._processor.tokenizer.pad_token_id
        self._model.config.ctc_loss_reduction = "mean"

        # Handle vocabulary size mismatch
        tokenizer_vocab_size = len(self._processor.tokenizer)
        if self._model.config.vocab_size != tokenizer_vocab_size:
            # Resize lm_head to match new vocabulary
            # Use from_pretrained's expand method if available, else manual resize
            import torch.nn as nn
            
            old_lm_head = self._model.lm_head
            old_vocab_size = old_lm_head.out_features
            new_lm_head = nn.Linear(
                self._model.config.hidden_size,
                tokenizer_vocab_size,
            )
            
            # Copy old weights where possible (for overlapping vocabulary)
            with torch.no_grad():
                copy_size = min(old_vocab_size, tokenizer_vocab_size)
                if copy_size > 0:
                    new_lm_head.weight[:copy_size, :] = old_lm_head.weight[:copy_size, :]
                    new_lm_head.bias[:copy_size] = old_lm_head.bias[:copy_size]
                    
                # Initialize new vocab slots (if vocab grew) with small random values
                # This prevents PAD token collapse
                if tokenizer_vocab_size > old_vocab_size:
                    new_lm_head.weight[copy_size:, :].normal_(0.0, 0.02)
                    new_lm_head.bias[copy_size:].zero_()
            
            self._model.lm_head = new_lm_head
            self._model.config.vocab_size = tokenizer_vocab_size

        self._apply_freezing(freeze_encoder=freeze_encoder, freeze_n_layers=freeze_n_layers)

    def _apply_freezing(self, freeze_encoder: bool, freeze_n_layers: int) -> None:
        """Apply parameter freezing strategy.

        Standard wav2vec2 fine-tuning *always* freezes the CNN feature
        extractor because those low-level audio features are already
        well-learned.  Optionally freeze the first *freeze_n_layers*
        transformer encoder layers as well.

        When *freeze_encoder* is True the entire wav2vec2 backbone
        (feature extractor + all encoder layers) is frozen and only
        ``lm_head`` remains trainable.
        """
        if not hasattr(self._model, "wav2vec2"):
            return

        # Always freeze CNN feature extractor (standard practice)
        self._model.wav2vec2.feature_extractor._freeze_parameters()

        if freeze_encoder:
            for param in self._model.wav2vec2.parameters():
                param.requires_grad = False
            return

        if freeze_n_layers > 0:
            encoder_layers = getattr(self._model.wav2vec2.encoder, "layers", None)
            if encoder_layers is not None:
                for layer in encoder_layers[:freeze_n_layers]:
                    for param in layer.parameters():
                        param.requires_grad = False

    def to(self, device: torch.device) -> None:
        self._model.to(device)

    def eval(self) -> None:
        self._model.eval()

    def transcribe_array(
        self,
        audio_array: torch.Tensor,
        sampling_rate: int,
        generation_config: Optional[GenerationConfig] = None,
    ) -> str:
        del generation_config

        inputs = self._processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )

        input_values = inputs.input_values.to(self._model.device)
        attention_mask = inputs.attention_mask.to(self._model.device) if hasattr(inputs, "attention_mask") else None

        with torch.no_grad():
            logits = self._model(input_values=input_values, attention_mask=attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self._processor.batch_decode(predicted_ids)[0]
        return transcription.strip()

    def generate_batch(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> torch.Tensor:
        del generation_config

        with torch.no_grad():
            logits = self._model(input_values=input_features, attention_mask=attention_mask).logits
        return torch.argmax(logits, dim=-1)

    def get_processor(self) -> Any:
        return self._processor

    def unwrap(self) -> Any:
        return self._model
