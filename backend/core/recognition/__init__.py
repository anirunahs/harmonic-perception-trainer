"""Recognition Module - Audio processing facade for interval recognition."""

from .result import RecognitionResult
from .facade import AudioProcessingFacade
from .strategies import FeatureExtractionStrategy, FFTStrategy, MFCCStrategy

__all__ = [
    'RecognitionResult',
    'AudioProcessingFacade',
    'FeatureExtractionStrategy',
    'FFTStrategy',
    'MFCCStrategy',
]
