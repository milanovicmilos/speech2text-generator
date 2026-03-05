"""Whisper adapter implementation conforming to generic ASR model contract."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .contracts import ASRModelAdapter, GenerationConfig

logger = logging.getLogger(__name__)


class WhisperASRAdapter(ASRModelAdapter):
    """OpenAI Whisper adapter for Serbian ASR use case."""

    def __init__(
        self,
        model_name_or_path: str,
        language: str = "Serbian",
        freeze_encoder: bool = True,
        freeze_n_layers: int = 0,
    ):
        self.model_name_or_path = model_name_or_path
        self.language = language
        self._generation_defaults = GenerationConfig()

        self._model = WhisperForConditionalGeneration.from_pretrained(
            model_name_or_path,
            attn_implementation="sdpa",
        )
        self._processor = WhisperProcessor.from_pretrained(model_name_or_path)

        self._clear_conflicting_generation_settings()
        self._apply_freezing(freeze_encoder=freeze_encoder, freeze_n_layers=freeze_n_layers)

    def _clear_conflicting_generation_settings(self) -> None:
        try:
            if hasattr(self._model, "generation_config"):
                self._model.generation_config.forced_decoder_ids = None
                self._model.generation_config.suppress_tokens = None
                self._model.generation_config.begin_suppress_tokens = None
            if hasattr(self._model, "config"):
                self._model.config.forced_decoder_ids = None
        except Exception:
            logger.exception("Failed clearing forced/suppress generation settings")

    def _apply_freezing(self, freeze_encoder: bool, freeze_n_layers: int) -> None:
        if freeze_encoder:
            for param in self._model.model.encoder.parameters():
                param.requires_grad = False
            logger.info("Froze all Whisper encoder parameters")
            return

        if freeze_n_layers > 0:
            for layer in self._model.model.encoder.layers[:freeze_n_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
            logger.info("Froze first %s Whisper encoder layers", freeze_n_layers)

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
        inputs = self._processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        generated_ids = self.generate_batch(
            input_features=inputs.input_features.to(self._model.device),
            attention_mask=None,
            generation_config=generation_config,
        )
        return self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def generate_batch(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> torch.Tensor:
        decode_cfg = generation_config or self._generation_defaults

        kwargs = decode_cfg.to_generate_kwargs(language=self.language)
        return self._model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            **kwargs,
        )

    def get_processor(self) -> WhisperProcessor:
        return self._processor

    def unwrap(self) -> Any:
        return self._model
