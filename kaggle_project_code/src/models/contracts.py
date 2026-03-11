"""Model contracts and shared configuration objects for ASR models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from transformers import WhisperProcessor


@dataclass(frozen=True)
class GenerationConfig:
    """Runtime decoding configuration used across ASR model adapters."""

    num_beams: int = 8
    no_repeat_ngram_size: int = 10
    repetition_penalty: float = 5.0
    length_penalty: float = 1.0
    early_stopping: bool = True
    temperature: float = 0.0
    max_new_tokens: int = 128
    max_length: int = 256
    suppress_tokens: Optional[list[int]] = None
    begin_suppress_tokens: Optional[list[int]] = None

    def to_generate_kwargs(self, language: str) -> Dict[str, Any]:
        """Convert to kwargs accepted by HF `generate` API."""
        return {
            "num_beams": self.num_beams,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
            "early_stopping": self.early_stopping,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "max_length": self.max_length,
            "language": language,
            "task": "transcribe",
            "suppress_tokens": self.suppress_tokens,
            "begin_suppress_tokens": self.begin_suppress_tokens,
        }


class ASRModelAdapter(ABC):
    """Abstraction over concrete ASR model implementations."""

    @abstractmethod
    def to(self, device: torch.device) -> None:
        """Move model to target device."""

    @abstractmethod
    def eval(self) -> None:
        """Switch model into evaluation mode."""

    @abstractmethod
    def transcribe_array(
        self,
        audio_array: torch.Tensor,
        sampling_rate: int,
        generation_config: Optional[GenerationConfig] = None,
    ) -> str:
        """Transcribe raw waveform to text."""

    @abstractmethod
    def generate_batch(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> torch.Tensor:
        """Generate token IDs from precomputed input features."""

    @abstractmethod
    def get_processor(self) -> WhisperProcessor:
        """Return processor used for feature extraction/tokenization."""

    @abstractmethod
    def unwrap(self) -> Any:
        """Return underlying framework model (e.g., HF model) when needed."""
