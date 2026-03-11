"""Transcriber classes for performing ASR inference."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import librosa

from ..models import GenerationConfig, get_model_registry
from ..utils import get_device

logger = logging.getLogger(__name__)


class ASRTranscriber:
    """
    High-level interface for transcribing audio with ASR models.
    
    Handles model loading, audio loading, and transcription.
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        language: str = "Serbian",
        model_type: str = "whisper",
        generation_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize transcriber.
        
        Args:
            model_path: Path to fine-tuned ASR model
            device: Device to use (cuda, cpu, auto)
            language: Language for transcription
        """
        self.model_path = Path(model_path)
        self.language = language
        self.model_type = model_type
        
        # Setup device
        if device is None or device == "auto":
            self.device = get_device("auto")
        else:
            self.device = torch.device(device)
        
        # Load model
        logger.info(f"Loading model from {self.model_path}...")
        registry = get_model_registry()
        self.model = registry.create(
            self.model_type,
            model_name_or_path=str(self.model_path),
            language=language,
            freeze_encoder=False,
        )

        # Generation defaults (can be overridden per call)
        self.generation_params: Dict[str, Any] = generation_params or {}
        
        # Move to device
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded on {self.device}")
    
    def transcribe(
        self,
        audio_path: str,
        sr: int = 16000,
        return_confidence: bool = False,
    ) -> str:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            sr: Sample rate
            return_confidence: Whether to return confidence scores
            
        Returns:
            Transcribed text
        """
        # Load audio
        logger.info(f"Loading audio from {audio_path}...")
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=40)
        
        # Transcribe
        generation_config = GenerationConfig(
            num_beams=self.generation_params.get("num_beams", 8),
            no_repeat_ngram_size=self.generation_params.get("no_repeat_ngram_size", 10),
            repetition_penalty=self.generation_params.get("repetition_penalty", 5.0),
            length_penalty=self.generation_params.get("length_penalty", 1.0),
            early_stopping=self.generation_params.get("early_stopping", True),
            temperature=self.generation_params.get("temperature", 0.0),
            max_new_tokens=self.generation_params.get("max_new_tokens", 128),
            max_length=self.generation_params.get("max_length", 256),
            suppress_tokens=self.generation_params.get("suppress_tokens", None),
            begin_suppress_tokens=self.generation_params.get("begin_suppress_tokens", None),
        )
        with torch.no_grad():
            transcription = self.model.transcribe_array(
                audio_array=audio,
                sampling_rate=sr,
                generation_config=generation_config,
            )
        
        logger.info(f"Transcribed: {transcription}")
        return transcription
    
    def transcribe_batch(
        self,
        audio_paths: list,
        sr: int = 16000,
    ) -> list:
        """
        Transcribe multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            sr: Sample rate
            
        Returns:
            List of transcribed texts
        """
        transcriptions = []
        
        for audio_path in audio_paths:
            try:
                text = self.transcribe(audio_path, sr=sr)
                transcriptions.append({
                    "path": audio_path,
                    "text": text,
                    "status": "success"
                })
            except Exception as e:
                logger.error(f"Error transcribing {audio_path}: {e}")
                transcriptions.append({
                    "path": audio_path,
                    "text": None,
                    "status": "error",
                    "error": str(e)
                })
        
        return transcriptions


WhisperTranscriber = ASRTranscriber
