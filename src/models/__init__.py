"""Model package exports.

SOLID-oriented contracts and registry abstractions.
"""

from .contracts import ASRModelAdapter, GenerationConfig
from .registry import ModelRegistry, get_model_registry
from .wav2vec2_adapter import Wav2Vec2ASRAdapter
from .whisper_adapter import WhisperASRAdapter

__all__ = [
	"ASRModelAdapter",
	"GenerationConfig",
	"ModelRegistry",
	"get_model_registry",
	"WhisperASRAdapter",
	"Wav2Vec2ASRAdapter",
]
