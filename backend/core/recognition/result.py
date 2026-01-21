"""
RecognitionResult - Data class for recognition results.

Pattern: Facade (supporting class)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class PredictionItem:
    """Single prediction result."""
    interval: str
    confidence: float
    rank: int = 0
    
    @property
    def percentage(self) -> float:
        """Confidence as percentage."""
        return round(self.confidence * 100, 1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'interval': self.interval,
            'confidence': self.confidence,
            'percentage': self.percentage,
            'rank': self.rank
        }


@dataclass
class RecognitionResult:
    """
    Result of audio recognition operation.
    
    Encapsulates all recognition data including:
    - Success/failure status
    - Best prediction
    - Top predictions list
    - Processing metadata
    - Error information
    """
    
    success: bool
    best_prediction: Optional[PredictionItem] = None
    top_predictions: List[PredictionItem] = field(default_factory=list)
    processing_time: float = 0.0
    audio_duration: float = 0.0
    error: Optional[str] = None
    error_code: Optional[str] = None
    
    @classmethod
    def from_prediction(
        cls,
        prediction: Dict[str, Any],
        processing_time: float = 0.0,
        audio_duration: float = 0.0
    ) -> 'RecognitionResult':
        """
        Create result from classifier prediction dict.
        
        Args:
            prediction: Dict from IntervalClassifier.predict()
            processing_time: Time taken for processing
            audio_duration: Duration of audio in seconds
        """
        if prediction.get('error', False):
            return cls(
                success=False,
                error=prediction.get('message', 'Recognition failed'),
                error_code='recognition_error'
            )
        
        best = prediction.get('best_prediction', {})
        best_item = PredictionItem(
            interval=best.get('interval', 'unknown'),
            confidence=best.get('confidence', 0.0),
            rank=1
        )
        
        top_items = []
        for i, pred in enumerate(prediction.get('predictions', [])[:3]):
            top_items.append(PredictionItem(
                interval=pred.get('interval', 'unknown'),
                confidence=pred.get('confidence', 0.0),
                rank=i + 1
            ))
        
        return cls(
            success=True,
            best_prediction=best_item,
            top_predictions=top_items,
            processing_time=processing_time,
            audio_duration=audio_duration
        )
    
    @classmethod
    def error_result(cls, error: str, error_code: str = 'error') -> 'RecognitionResult':
        """Create error result."""
        return cls(
            success=False,
            error=error,
            error_code=error_code
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response dictionary."""
        if not self.success:
            return {
                'status': 'error',
                'error': self.error,
                'error_code': self.error_code
            }
        
        return {
            'status': 'success',
            'best_prediction': self.best_prediction.to_dict() if self.best_prediction else None,
            'top_predictions': [p.to_dict() for p in self.top_predictions],
            'processing_time': round(self.processing_time, 2),
            'audio_duration': round(self.audio_duration, 1)
        }
