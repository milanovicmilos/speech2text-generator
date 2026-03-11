"""Metrics utilities for speech-to-text evaluation."""

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from jiwer import cer, wer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from .text_preprocessing import SerbianTextPreprocessor

logger = logging.getLogger(__name__)


def preprocess_text_pairs(
    predictions: Sequence[str],
    references: Sequence[str],
    preprocessor: Optional[SerbianTextPreprocessor] = None,
) -> Tuple[List[str], List[str]]:
    """Normalize prediction/reference text pairs using one shared preprocessor."""
    processor = preprocessor or SerbianTextPreprocessor()
    normalized_predictions = [processor.preprocess(text) for text in predictions]
    normalized_references = [processor.preprocess(text) for text in references]
    return normalized_predictions, normalized_references


def evaluate_predictions(
    predictions: Sequence[str],
    references: Sequence[str],
    preprocessor: Optional[SerbianTextPreprocessor] = None,
) -> Dict[str, object]:
    """Compute WER/CER on consistently preprocessed predictions and references."""
    normalized_predictions, normalized_references = preprocess_text_pairs(
        predictions=predictions,
        references=references,
        preprocessor=preprocessor,
    )

    if len(normalized_predictions) != len(normalized_references):
        n = min(len(normalized_predictions), len(normalized_references))
        logger.warning(
            "Prediction/reference length mismatch (%s vs %s). Truncating to %s.",
            len(normalized_predictions),
            len(normalized_references),
            n,
        )
        normalized_predictions = normalized_predictions[:n]
        normalized_references = normalized_references[:n]

    wer_value = float(wer(normalized_references, normalized_predictions)) if normalized_predictions else 1.0
    cer_value = float(cer(normalized_references, normalized_predictions)) if normalized_predictions else 1.0

    return {
        "wer": wer_value,
        "cer": cer_value,
        "predictions_preprocessed": normalized_predictions,
        "references_preprocessed": normalized_references,
    }


class MetricsCalculator:
    """Calculate evaluation metrics for speech-to-text."""
    
    def __init__(self, processor=None):
        """
        Initialize metrics calculator.
        
        Args:
            processor: Text processor for decoding
        """
        self.processor = processor
        self.smoothing = SmoothingFunction()
    
    def calculate_wer(self, predictions: List[str], references: List[str]) -> float:
        """
        Calculate Word Error Rate.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            WER score (lower is better)
        """
        try:
            return wer(references, predictions)
        except Exception as e:
            logger.warning(f"Error calculating WER: {e}")
            return 1.0
    
    def calculate_cer(self, predictions: List[str], references: List[str]) -> float:
        """
        Calculate Character Error Rate.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            CER score (lower is better)
        """
        try:
            return cer(references, predictions)
        except Exception as e:
            logger.warning(f"Error calculating CER: {e}")
            return 1.0
    
    def calculate_bleu(self, predictions: List[str], references: List[str]) -> float:
        """
        Calculate BLEU score.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            BLEU score (higher is better)
        """
        try:
            bleu_scores = []
            for pred, ref in zip(predictions, references):
                pred_tokens = pred.split()
                ref_tokens = [ref.split()]  # BLEU expects list of references
                
                score = sentence_bleu(
                    ref_tokens,
                    pred_tokens,
                    smoothing_function=self.smoothing.method1
                )
                bleu_scores.append(score)
            
            return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
        except Exception as e:
            logger.warning(f"Error calculating BLEU: {e}")
            return 0.0
    
    def calculate_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Calculate all metrics.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # WER
        metrics["wer"] = self.calculate_wer(predictions, references)
        
        # CER
        metrics["cer"] = self.calculate_cer(predictions, references)
        
        # BLEU
        metrics["bleu"] = self.calculate_bleu(predictions, references)
        
        return metrics
