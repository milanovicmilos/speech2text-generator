"""Model package exports.

SOLID-oriented contracts and registry abstractions.
"""

from .contracts import ASRModelAdapter, GenerationConfig
from .registry import ModelRegistry, get_model_registry
from .whisper_adapter import WhisperASRAdapter

__all__ = [
	"ASRModelAdapter",
	"GenerationConfig",
	"ModelRegistry",
	"get_model_registry",
	"WhisperASRAdapter",
]
