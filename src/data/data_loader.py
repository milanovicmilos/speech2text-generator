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
from ..utils.text_preprocessing import SerbianTextPreprocessor
import re

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
        processor: Any,
        sample_rate: int = 16000,
        max_duration: float = 300.0,  # Increased for raw audio
        text_preprocessor: Optional[SerbianTextPreprocessor] = None,
        augment: bool = False,  # Enable data augmentation for training
    ):
        """
        Initialize dataset.
        
        Args:
            audio_dir: Directory with audio files
            text_dir: Directory with text transcriptions
            processor: WhisperProcessor for feature extraction
            sample_rate: Target sample rate (MUST be 16000 Hz for Wav2Vec2/Whisper). Audio will be resampled if necessary.
            max_duration: Maximum audio duration in seconds
            text_preprocessor: Text preprocessor instance
        """
        # Resolve dirs to absolute paths to avoid relative path mismatches
        self.text_dir = Path(text_dir).resolve()
        self.audio_dir = Path(audio_dir).resolve()
        self.processor = processor
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_samples = int(max_duration * sample_rate)
        self.text_preprocessor = text_preprocessor or SerbianTextPreprocessor()
        self.augment = augment
        
        # Find all audio files (recursive)
        self.audio_files = sorted(self.audio_dir.glob("**/*.mp3")) + \
                  sorted(self.audio_dir.glob("**/*.wav")) + \
                  sorted(self.audio_dir.glob("**/*.flac"))
        
        if not self.audio_files:
            raise ValueError(f"No audio files found in {audio_dir}")
        
        logger.info(f"Found {len(self.audio_files)} audio files")
        
        # Load transcriptions
        self.transcriptions = {}
        self._load_transcriptions()
        
        # Filter to only valid pairs and create chunk indices
        # For raw data: split into 30s chunks; for chunked: keep as-is
        self.valid_file_indices = self._find_valid_pairs()
        self.chunk_indices = self._create_chunk_indices()
        
        logger.info(f"Found {len(self.valid_file_indices)} valid audio-text pairs")
        logger.info(f"Created {len(self.chunk_indices)} chunks for training (30s per chunk)")
    
    def _load_transcriptions(self):
        """Load transcriptions from text files."""
        for audio_path in self.audio_files:
            # Try multiple strategies to find matching transcription:
            # 1) Flat mapping: text_dir / <stem>.txt
            # 2) Relative mapping: keep same relative path under text_dir (for mirrored dirs)
            # 3) Filename mapping: text_dir / <audio_name>.txt
            stem = audio_path.stem
            candidates = []

            # Flat
            candidates.append(self.text_dir / f"{stem}.txt")

            # Relative (mirrored subdirs)
            try:
                rel = audio_path.relative_to(self.audio_dir)
                candidates.append(self.text_dir / rel.with_suffix('.txt'))
            except Exception:
                # audio_path not under audio_dir (shouldn't happen), skip
                pass

            # Name-based (use the name without extension)
            candidates.append(self.text_dir / audio_path.with_suffix('.txt').name)

            found = False
            for text_path in candidates:
                text_path = text_path.resolve()
                if text_path.exists():
                    try:
                        with open(text_path, 'r', encoding='utf-8') as f:
                            raw_text = f.read().strip()
                            # Store preprocessed text to keep stats consistent
                            try:
                                text = self.text_preprocessor.preprocess(raw_text)
                            except Exception:
                                text = raw_text
                            self.transcriptions[stem] = text
                            found = True
                            break
                    except Exception as e:
                        logger.error(f"Error reading transcription {text_path}: {e}")

            if not found:
                logger.warning(f"No text file found for {audio_path.name} (tried {len(candidates)} paths)")

    def _audio_duration(self, audio_path: Path) -> float:
        try:
            # use librosa.get_duration which is faster for just duration
            return float(librosa.get_duration(path=str(audio_path)))
        except Exception:
            return 0.0
    
    def _find_valid_pairs(self) -> List[int]:
        """Find indices with valid audio-text pairs."""
        valid = []
        problematic = []
        for idx, audio_path in enumerate(self.audio_files):
            stem = audio_path.stem
            if stem not in self.transcriptions:
                problematic.append((str(audio_path), 'missing_text'))
                continue

            text = self.transcriptions.get(stem, '')
            if not text or not text.strip():
                problematic.append((str(audio_path), 'empty_text'))
                continue

            # duration / words sanity check
            duration = self._audio_duration(audio_path)
            words = len(text.split())
            if duration <= 0:
                problematic.append((str(audio_path), 'zero_duration'))
                continue
            # words per second should be within reasonable bounds
            wps = words / duration if duration > 0 else 0.0
            if wps < 0.3 or wps > 5.0 or words < 5 or duration < 5.0:
                problematic.append((str(audio_path), f'invalid_stats: wps={wps:.2f}, words={words}, dur={duration:.2f}'))
                continue
            valid.append(idx)

        # write problematic report for user's inspection
        if problematic:
            try:
                report_dir = Path('data') / 'problematic'
                report_dir.mkdir(parents=True, exist_ok=True)
                report_file = report_dir / 'problematic_pairs.json'
                with open(report_file, 'w', encoding='utf-8') as rf:
                    json.dump([{'audio': a, 'issue': i} for a, i in problematic], rf, ensure_ascii=False, indent=2)
                logger.info(f"Wrote problematic pairs report: {report_file}")
            except Exception:
                logger.exception('Failed to write problematic pairs report')
        return valid
    
    def _create_chunk_indices(self) -> List[Tuple[int, int]]:
        """
        Create chunk indices for 30s chunks from valid audio files.
        For raw audio (audio_dir == text_dir): split into 30s chunks
        For chunked audio: keep as single chunks
        
        Returns:
            List of tuples: (file_idx, chunk_number)
        """
        chunk_indices = []
        is_raw = self.audio_dir == self.text_dir
        chunk_duration_sec = 30
        
        for file_idx in self.valid_file_indices:
            audio_path = self.audio_files[file_idx]
            
            if is_raw:
                # For raw: calculate how many 30s chunks can fit
                full_dur = self._audio_duration(audio_path)
                if full_dur > 0:
                    num_chunks = max(1, int(np.ceil(full_dur / chunk_duration_sec)))
                    for chunk_num in range(num_chunks):
                        chunk_indices.append((file_idx, chunk_num))
            else:
                # For chunked: treat entire file as one chunk
                chunk_indices.append((file_idx, 0))
        
        return chunk_indices
    
    def __len__(self) -> int:
        """Return dataset size (number of chunks, not files)."""
        return len(self.chunk_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get dataset item (30s chunk).
        
        Args:
            idx: Index (chunk index)
            
        Returns:
            Dictionary with audio features and text labels
        """
        # Get file index and chunk number
        file_idx, chunk_num = self.chunk_indices[idx]
        audio_path = self.audio_files[file_idx]
        
        # Get transcription
        text = self.transcriptions[audio_path.stem]
        text = self.text_preprocessor.preprocess(text)
        
        # Load audio
        try:
            audio, loaded_sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                mono=True
            )
            # Verify resampling to 16kHz
            if loaded_sr != self.sample_rate:
                logger.warning(f"Audio resampled from {loaded_sr} Hz to {self.sample_rate} Hz: {audio_path.name}")
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            # Return dummy data
            audio = np.zeros(30 * self.sample_rate)  # 30s dummy
        
        # Apply data augmentation if enabled (for training robustness)
        if self.augment:
            # Speed perturbation: 0.9x to 1.1x
            speed_factor = np.random.uniform(0.9, 1.1)
            try:
                audio = librosa.effects.time_stretch(audio, rate=speed_factor)
            except Exception:
                # time_stretch can fail for very short audio; skip if error
                pass
            # Add noise: small Gaussian noise
            noise = np.random.normal(0, 0.005, len(audio))
            audio = audio + noise
            audio = np.clip(audio, -1, 1)  # Clip to prevent overflow
        
        # Extract 30s chunk
        chunk_duration_sec = 30
        chunk_samples = chunk_duration_sec * self.sample_rate
        
        # Ensure audio is at correct sample rate (16000 Hz)
        if not hasattr(self, '_sr_verified'):
            logger.info(f"Using sample rate: {self.sample_rate} Hz (target for Wav2Vec2/Whisper)")
            self._sr_verified = True
        
        is_raw = self.audio_dir == self.text_dir
        if is_raw:
            # For raw data: extract chunk at chunk_num position
            start_sample = chunk_num * chunk_samples
            end_sample = start_sample + chunk_samples
            
            if end_sample > len(audio):
                # Last chunk might be shorter
                audio_chunk = audio[start_sample:]
                audio_chunk = np.pad(audio_chunk, (0, chunk_samples - len(audio_chunk)))
            else:
                audio_chunk = audio[start_sample:end_sample]
            
            # Extract corresponding text portion
            full_dur = self._audio_duration(audio_path)
            if full_dur > 0:
                words = text.split()
                num_words = len(words)
                words_per_second = num_words / full_dur if full_dur > 0 else 0
                
                chunk_start_sec = chunk_num * chunk_duration_sec
                chunk_end_sec = (chunk_num + 1) * chunk_duration_sec
                
                start_word_idx = max(0, int(chunk_start_sec * words_per_second))
                end_word_idx = min(num_words, int(chunk_end_sec * words_per_second))
                
                text = ' '.join(words[start_word_idx:end_word_idx]) if start_word_idx < end_word_idx else text
            
            audio = audio_chunk
        else:
            # For chunked data: use full chunk (already ~30s)
            if len(audio) > chunk_samples:
                audio = audio[:chunk_samples]
            else:
                audio = np.pad(audio, (0, chunk_samples - len(audio)))
        
        # Process with model processor
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )

        # Handle processor output shape for both seq2seq (Whisper) and CTC (Wav2Vec2)
        input_features = None
        input_values = None
        if hasattr(inputs, "input_features") and inputs.input_features is not None:
            input_features = inputs.input_features[0]
        if hasattr(inputs, "input_values") and inputs.input_values is not None:
            input_values = inputs.input_values[0]

        attention_mask = None
        if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
            attention_mask = inputs.attention_mask[0]
        elif input_features is not None:
            try:
                import torch as _torch
                attention_mask = _torch.ones(input_features.shape[0], dtype=_torch.long)
            except Exception:
                attention_mask = None
        
        # Tokenize text
        labels = self.processor.tokenizer(
            text,
            return_tensors="pt",
            max_length=448,
            truncation=True
        ).input_ids[0]  # Remove batch dim
        
        return {
            "input_features": input_features,
            "input_values": input_values,
            "attention_mask": attention_mask,
            "labels": labels,
            "text": text,
        }


@dataclass
class WhisperDataCollator:
    """
    Data collator for Whisper.

    Handles padding of variable-length sequences. Optionally returns the
    original `text` strings when `include_text=True` which is useful for
    offline evaluation. For training with `Trainer` keep `include_text=False`
    to avoid passing unexpected kwargs to the model forward.
    """

    processor: Any
    include_text: bool = False

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch.
        
        Args:
            batch: List of dataset items
            
        Returns:
            Batched tensors
        """
        # Extract inputs and labels
        input_key = "input_features"
        if batch[0].get("input_values") is not None:
            input_key = "input_values"
        model_inputs = [item[input_key] for item in batch]
        labels = [item["labels"] for item in batch]

        # Pad model inputs (already standardized in dataset)
        model_inputs = torch.stack(model_inputs)
        # build attention_mask batch if present
        attention_masks = None
        if "attention_mask" in batch[0] and batch[0]["attention_mask"] is not None:
            attention_masks = torch.stack([item["attention_mask"] for item in batch])
        
        # Pad labels
        max_label_len = max(len(label) for label in labels)
        labels_padded = []
        
        for label in labels:
            padded = torch.full((max_label_len,), -100, dtype=torch.long)
            padded[:len(label)] = label
            labels_padded.append(padded)
        
        labels = torch.stack(labels_padded)
        
        texts = [item.get("text", "") for item in batch]

        out = {
            input_key: model_inputs,
            "attention_mask": attention_masks,
            "labels": labels,
        }

        if self.include_text:
            out["text"] = texts

        return out


def create_dataloaders(
    data_dir: str,
    processor: Any,
    batch_size: int = 4,
    num_workers: int = 0,
    pin_memory: bool = False,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    augment_train: bool = True,  # Enable augmentation for training set
    group_split: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders.
    
    Args:
        data_dir: Directory with audio and text files
        processor: Model processor (Whisper or Wav2Vec2)
        batch_size: Batch size
        num_workers: Number of workers
        pin_memory: Pin memory for faster transfer to GPU
        train_ratio: Training split ratio
        val_ratio: Validation split ratio
        seed: Random seed
        augment_train: Enable augmentation for training set
        group_split: Keep all chunks from the same source in one split
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Detect common chunked structure: data_dir/audio and data_dir/text
    base = Path(data_dir)
    audio_dir = base
    text_dir = base

    audio_sub = base / "audio"
    text_sub = base / "text"
    if audio_sub.exists() and text_sub.exists():
        logger.info(f"Detected chunked dataset structure: audio={audio_sub}, text={text_sub}")
        audio_dir = audio_sub
        text_dir = text_sub

        # If text_dir exists but is empty, try to populate chunked text files
        # by mapping back to original transcripts found under data/raw
        try:
            has_text = any(text_sub.glob("**/*.txt"))
        except Exception:
            has_text = False

        if not has_text:
            logger.info(f"Chunked text dir {text_sub} is empty — attempting to populate from data/raw (split per chunk)")
            # root to search for original transcripts; expect repo/data/raw/<set>
            raw_root = base.parents[1] / "raw" if len(base.parents) > 1 else None
            if raw_root and raw_root.exists():
                # Build index of raw transcripts by stem
                raw_index = {p.stem: p for p in raw_root.rglob("*.txt")}

                # Group chunk files by base stem (strip trailing _chunkNN)
                chunk_re = re.compile(r"(?P<base>.+?)_chunk(?P<idx>\d+)$")
                groups = {}
                for a in audio_sub.glob("**/*"):
                    if a.suffix.lower() not in (".wav", ".mp3", ".flac"):
                        continue
                    m = chunk_re.match(a.stem)
                    if m:
                        base_stem = m.group("base")
                    else:
                        base_stem = a.stem
                    groups.setdefault(base_stem, []).append(a)

                created = 0
                for base_stem, files in groups.items():
                    files = sorted(files, key=lambda p: p.name)
                    raw_src = raw_index.get(base_stem)
                    if not raw_src:
                        # try a looser match: sometimes raw file names don't match exactly (strip punctuation)
                        alt = base_stem.replace('---', '-').replace('--', '-').strip('-')
                        raw_src = raw_index.get(alt)

                    if not raw_src:
                        logger.debug(f"No raw transcript found for base '{base_stem}', skipping")
                        continue

                    # Read raw transcript and split into roughly equal parts by words
                    try:
                        text = raw_src.read_text(encoding='utf-8').strip()
                    except Exception as e:
                        logger.error(f"Failed reading raw transcript {raw_src}: {e}")
                        continue

                    if not text:
                        logger.debug(f"Raw transcript {raw_src} is empty, skipping")
                        continue

                    words = text.split()
                    k = len(files)
                    if k == 0:
                        continue

                    # Distribute words into k parts as evenly as possible
                    base_count = len(words) // k
                    remainder = len(words) % k
                    parts = []
                    idx = 0
                    for i in range(k):
                        take = base_count + (1 if i < remainder else 0)
                        part_words = words[idx: idx + take] if take > 0 else []
                        parts.append(' '.join(part_words).strip())
                        idx += take

                    # If splitting yielded empty parts (e.g., short transcripts), fall back to copying whole text
                    if all(not p for p in parts):
                        parts = [text] * k

                    # Write parts to corresponding chunk text files
                    for file_obj, part_text in zip(files, parts):
                        dest = text_sub / f"{file_obj.stem}.txt"
                        try:
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            dest.write_text(part_text, encoding='utf-8')
                            created += 1
                        except Exception as e:
                            logger.error(f"Failed to write chunk transcript {dest}: {e}")

                logger.info(f"Populated {created} chunked transcripts into {text_sub}")
            else:
                logger.warning(f"No raw transcripts root found at {raw_root}; skipping auto-populate")

    # Create datasets (train with augmentation, val/test without)
    train_dataset_full = WhisperSpeechDataset(
        audio_dir=str(audio_dir),
        text_dir=str(text_dir),
        processor=processor,
        augment=augment_train,
    )
    # Create a non-augmenting copy for val/test but ensure it shares the same
    # indexing (valid_indices and audio_files) so splits are consistent.
    val_test_dataset = WhisperSpeechDataset(
        audio_dir=str(audio_dir),
        text_dir=str(text_dir),
        processor=processor,
        augment=False,
    )
    # Align the indexing between train and val/test datasets to avoid mismatches
    try:
        val_test_dataset.valid_file_indices = train_dataset_full.valid_file_indices
        val_test_dataset.chunk_indices = train_dataset_full.chunk_indices
        val_test_dataset.transcriptions = train_dataset_full.transcriptions
        val_test_dataset.audio_files = train_dataset_full.audio_files
    except Exception:
        # If anything goes wrong, proceed but log a warning
        logger.warning('Could not align train/val_test dataset indices; proceeding with defaults')
    
    # Split dataset
    n = len(train_dataset_full)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    
    # Reproducible split
    if group_split:
        # Group by base stem so *_chunkNN variants stay in the same split
        chunk_re = re.compile(r"(?P<base>.+?)_chunk\d+$")
        groups: Dict[str, List[int]] = {}
        for ds_idx in range(n):
            file_idx = train_dataset_full.valid_indices[ds_idx]
            audio_path = train_dataset_full.audio_files[file_idx]
            stem = audio_path.stem
            m = chunk_re.match(stem)
            group_key = m.group('base') if m else stem
            groups.setdefault(group_key, []).append(ds_idx)

        rng = np.random.RandomState(seed)
        group_keys = list(groups.keys())
        rng.shuffle(group_keys)

        train_indices: List[int] = []
        val_indices: List[int] = []
        test_indices: List[int] = []

        for key in group_keys:
            group_ids = groups[key]
            if len(train_indices) < n_train:
                train_indices.extend(group_ids)
            elif len(val_indices) < n_val:
                val_indices.extend(group_ids)
            else:
                test_indices.extend(group_ids)

        # Safety: trim potential overshoot and keep all indices accounted for
        all_indices = train_indices + val_indices + test_indices
        if len(all_indices) > n:
            all_indices = all_indices[:n]

        # Ensure disjointness of groups (for logging/validation)
        train_groups = set()
        val_groups = set()
        test_groups = set()
        for ds_idx in train_indices:
            file_idx = train_dataset_full.valid_indices[ds_idx]
            stem = train_dataset_full.audio_files[file_idx].stem
            m = chunk_re.match(stem)
            train_groups.add(m.group('base') if m else stem)
        for ds_idx in val_indices:
            file_idx = train_dataset_full.valid_indices[ds_idx]
            stem = train_dataset_full.audio_files[file_idx].stem
            m = chunk_re.match(stem)
            val_groups.add(m.group('base') if m else stem)
        for ds_idx in test_indices:
            file_idx = train_dataset_full.valid_indices[ds_idx]
            stem = train_dataset_full.audio_files[file_idx].stem
            m = chunk_re.match(stem)
            test_groups.add(m.group('base') if m else stem)

        overlap_count = len(train_groups & val_groups) + len(train_groups & test_groups) + len(val_groups & test_groups)
        logger.info(f"Group split enabled: groups={len(groups)}, overlap_count={overlap_count}")
        logger.info(f"Group split sample counts: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

        # Convert to numpy arrays for Subset compatibility
        train_indices = np.array(train_indices, dtype=np.int64)
        val_indices = np.array(val_indices, dtype=np.int64)
        test_indices = np.array(test_indices, dtype=np.int64)
    else:
        indices = np.random.RandomState(seed).permutation(n)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
    
    # Create subsets
    from torch.utils.data import Subset
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_test_dataset, val_indices)
    test_dataset = Subset(val_test_dataset, test_indices)
    
    # Create data collators: one for training (no raw text passed to model)
    # and one for evaluation (include original text for reference/metrics).
    collator = WhisperDataCollator(processor=processor, include_text=False)
    collator_eval = WhisperDataCollator(processor=processor, include_text=True)
    
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
        collate_fn=collator_eval,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator_eval,
    )
    
    logger.info(f"Created dataloaders:")
    logger.info(f"  Train: {len(train_loader)} batches ({n_train} samples)")
    logger.info(f"  Val: {len(val_loader)} batches ({n_val} samples)")
    logger.info(f"  Test: {len(test_loader)} batches ({n_test} samples)")
    
    return train_loader, val_loader, test_loader
