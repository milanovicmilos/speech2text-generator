"""
Transcriber class for performing inference with Whisper ASR model.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import librosa
import numpy as np

from ..models import WhisperASRModel
from ..utils import get_device

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """
    High-level interface for transcribing audio with Whisper.
    
    Handles model loading, audio loading, and transcription.
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        language: str = "Serbian",
        generation_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize transcriber.
        
        Args:
            model_path: Path to fine-tuned Whisper model
            device: Device to use (cuda, cpu, auto)
            language: Language for transcription
        """
        self.model_path = Path(model_path)
        self.language = language
        
        # Setup device
        if device is None or device == "auto":
            self.device = get_device("auto")
        else:
            self.device = torch.device(device)
        
        # Load model
        logger.info(f"Loading model from {self.model_path}...")
        self.model = WhisperASRModel(model_name=str(self.model_path), language=language)

        # Generation defaults (can be overridden per call)
        self.generation_params: Dict[str, Any] = generation_params or {}
        
        # Move to device
        self.model.get_model().to(self.device)
        self.model.get_model().eval()
        
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
        with torch.no_grad():
            transcription = self.model.transcribe(
                audio,
                sampling_rate=sr,
                generation_kwargs=self.generation_params,
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
