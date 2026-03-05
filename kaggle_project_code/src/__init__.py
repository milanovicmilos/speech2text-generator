"""
Whisper Serbian ASR - A speech recognition library for Serbian language.

Package structure:
- models: Model definitions and wrappers
- data: Data loading and preprocessing
- inference: Inference utilities
- utils: Configuration, logging, and helpers
"""

__version__ = "1.0.0"
__author__ = "Milos"

from .data import WhisperSpeechDataset, WhisperDataCollator, create_dataloaders
from .models import (
    ASRModelAdapter,
    GenerationConfig,
    ModelRegistry,
    WhisperASRAdapter,
    get_model_registry,
)
from .utils import (
    load_config,
    save_config,
    setup_logging,
    set_seed,
    get_device,
    count_parameters,
    format_time,
    SerbianTextPreprocessor,
)

__all__ = [
    "WhisperSpeechDataset",
    "WhisperDataCollator",
    "create_dataloaders",
    "ASRModelAdapter",
    "GenerationConfig",
    "ModelRegistry",
    "get_model_registry",
    "WhisperASRAdapter",
    "load_config",
    "save_config",
    "setup_logging",
    "set_seed",
    "get_device",
    "count_parameters",
    "format_time",
    "SerbianTextPreprocessor",
]
