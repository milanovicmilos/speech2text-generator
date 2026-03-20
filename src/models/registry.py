"""Model registry and factory helpers for ASR adapters."""

from __future__ import annotations

from typing import Callable, Dict, Optional

from .contracts import ASRModelAdapter
from .wav2vec2_adapter import Wav2Vec2ASRAdapter
from .whisper_adapter import WhisperASRAdapter

ModelBuilder = Callable[..., ASRModelAdapter]


class ModelRegistry:
    """Simple registry for decoupling model selection from consumers."""

    def __init__(self):
        self._builders: Dict[str, ModelBuilder] = {}

    def register(self, model_type: str, builder: ModelBuilder) -> None:
        self._builders[model_type.lower()] = builder

    def create(self, model_type: str, **kwargs) -> ASRModelAdapter:
        key = model_type.lower()
        if key not in self._builders:
            supported = ", ".join(sorted(self._builders.keys()))
            raise ValueError(f"Unknown model_type '{model_type}'. Supported: {supported}")
        return self._builders[key](**kwargs)


_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
        _registry.register("whisper", lambda **kwargs: WhisperASRAdapter(**kwargs))
        _registry.register("wav2vec2", lambda **kwargs: Wav2Vec2ASRAdapter(**kwargs))
    return _registry
