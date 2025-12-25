"""
Metrics calculation for speech-to-text evaluation.
Includes WER, CER, BLEU, and other standard metrics.
"""

import logging
from typing import List, Dict
from jiwer import wer, cer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

logger = logging.getLogger(__name__)


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
