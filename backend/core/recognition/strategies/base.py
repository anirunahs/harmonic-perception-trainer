"""
Base class for feature extraction strategies.

Pattern: Strategy (Behavioral)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class FeatureExtractionStrategy(ABC):
    """
    Abstract base class for feature extraction strategies.
    
    Defines the interface for all feature extraction algorithms.
    Concrete strategies implement different approaches:
    - FFT-based (current production)
    - MFCC-based (alternative)
    - Chroma-based (future)
    
    Usage:
        strategy = FFTStrategy()
        features = strategy.extract_features(audio, sample_rate)
    """
    
    @abstractmethod
    def extract_features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract features from audio signal.
        
        Args:
            audio: Audio signal as numpy array
            sample_rate: Sample rate in Hz
            
        Returns:
            Feature vector as numpy array
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the strategy name.
        
        Returns:
            Human-readable strategy name
        """
        pass
    
    @abstractmethod
    def get_feature_count(self) -> int:
        """
        Get the expected number of features.
        
        Returns:
            Number of features in output vector
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get strategy information.
        
        Returns:
            Dict with strategy metadata
        """
        return {
            'name': self.get_name(),
            'feature_count': self.get_feature_count(),
            'type': self.__class__.__name__
        }
