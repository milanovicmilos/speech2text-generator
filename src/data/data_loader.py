"""
Data loading for Whisper ASR model.

Handles:
- Loading audio files
- Processing transcriptions
- Feature extraction for Whisper
- Batching and padding
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
import librosa
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import WhisperProcessor

from ..utils.text_preprocessing import SerbianTextPreprocessor

logger = logging.getLogger(__name__)


class WhisperSpeechDataset(Dataset):
    """
    Dataset for Whisper ASR.
    
    Loads audio files and their transcriptions, preprocesses them,
    and prepares them for Whisper model training.
    """
    
    def __init__(
        self,
        audio_dir: str,
        text_dir: str,
        processor: WhisperProcessor,
        sample_rate: int = 16000,
        max_duration: float = 30.0,
        text_preprocessor: Optional[SerbianTextPreprocessor] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            audio_dir: Directory with audio files
            text_dir: Directory with text transcriptions
            processor: WhisperProcessor for feature extraction
            sample_rate: Target sample rate
            max_duration: Maximum audio duration in seconds
            text_preprocessor: Text preprocessor instance
        """
        self.audio_dir = Path(audio_dir)
        self.text_dir = Path(text_dir)
        self.processor = processor
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_samples = int(max_duration * sample_rate)
        self.text_preprocessor = text_preprocessor or SerbianTextPreprocessor()
        
        # Find all audio files
        self.audio_files = sorted(self.audio_dir.glob("**/*.mp3")) + \
                          sorted(self.audio_dir.glob("**/*.wav")) + \
                          sorted(self.audio_dir.glob("**/*.flac"))
        
        if not self.audio_files:
            raise ValueError(f"No audio files found in {audio_dir}")
        
        logger.info(f"Found {len(self.audio_files)} audio files")
        
        # Load transcriptions
        self.transcriptions = {}
        self._load_transcriptions()
        
        # Filter to only valid pairs
        self.valid_indices = self._find_valid_pairs()
        
        logger.info(f"Found {len(self.valid_indices)} valid audio-text pairs")
    
    def _load_transcriptions(self):
        """Load transcriptions from text files."""
        for audio_path in self.audio_files:
            # Try different text file formats
            text_path = self.text_dir / (audio_path.stem + ".txt")
            
            if text_path.exists():
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    self.transcriptions[audio_path.stem] = text
            else:
                logger.warning(f"No text file for {audio_path.name}")
    
    def _find_valid_pairs(self) -> List[int]:
        """Find indices with valid audio-text pairs."""
        valid = []
        for idx, audio_path in enumerate(self.audio_files):
            if audio_path.stem in self.transcriptions:
                valid.append(idx)
        return valid
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get dataset item.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with audio features and text labels
        """
        # Get actual file index
        file_idx = self.valid_indices[idx]
        audio_path = self.audio_files[file_idx]
        
        # Load audio
        try:
            audio, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                mono=True
            )
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            # Return dummy data
            audio = np.zeros(self.max_samples)
        
        # Trim or pad to max duration
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        else:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))
        
        # Get transcription
        text = self.transcriptions[audio_path.stem]
        text = self.text_preprocessor.preprocess(text)
        
        # Process with Whisper processor
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        
        # Get input features (log-mel spectrogram)
        input_features = inputs.input_features[0]  # Remove batch dim
        
        # Tokenize text
        labels = self.processor.tokenizer(
            text,
            return_tensors="pt"
        ).input_ids[0]  # Remove batch dim
        
        return {
            "input_features": input_features,
            "labels": labels,
            "audio_path": str(audio_path),
            "text": text,
        }


@dataclass
class WhisperDataCollator:
    """
    Data collator for Whisper.
    
    Handles padding of variable-length sequences.
    """
    
    processor: WhisperProcessor
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch.
        
        Args:
            batch: List of dataset items
            
        Returns:
            Batched tensors
        """
        # Extract inputs and labels
        input_features = [item["input_features"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        # Pad input features (all should be same size already from processor)
        input_features = torch.stack(input_features)
        
        # Pad labels
        max_label_len = max(len(label) for label in labels)
        labels_padded = []
        
        for label in labels:
            padded = torch.full((max_label_len,), -100, dtype=torch.long)
            padded[:len(label)] = label
            labels_padded.append(padded)
        
        labels = torch.stack(labels_padded)
        
        return {
            "input_features": input_features,
            "labels": labels,
        }


def create_dataloaders(
    data_dir: str,
    processor: WhisperProcessor,
    batch_size: int = 4,
    num_workers: int = 0,
    pin_memory: bool = False,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders.
    
    Args:
        data_dir: Directory with audio and text files
        processor: WhisperProcessor
        batch_size: Batch size
        num_workers: Number of workers
        pin_memory: Pin memory for faster transfer to GPU
        train_ratio: Training split ratio
        val_ratio: Validation split ratio
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create dataset
    dataset = WhisperSpeechDataset(
        audio_dir=data_dir,
        text_dir=data_dir,
        processor=processor,
    )
    
    # Split dataset
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    
    # Reproducible split
    indices = np.random.RandomState(seed).permutation(n)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create subsets
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Create data collator
    collator = WhisperDataCollator(processor=processor)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
    )
    
    logger.info(f"Created dataloaders:")
    logger.info(f"  Train: {len(train_loader)} batches ({n_train} samples)")
    logger.info(f"  Val: {len(val_loader)} batches ({n_val} samples)")
    logger.info(f"  Test: {len(test_loader)} batches ({n_test} samples)")
    
    return train_loader, val_loader, test_loader
