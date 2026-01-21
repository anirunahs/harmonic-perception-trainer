"""
MFCC-based feature extraction strategy.

Pattern: Strategy (Behavioral) - Concrete Strategy
"""

import numpy as np
import librosa
import logging

from .base import FeatureExtractionStrategy

logger = logging.getLogger(__name__)


class MFCCStrategy(FeatureExtractionStrategy):
    """
    MFCC-based feature extraction strategy.
    
    Extracts Mel-frequency cepstral coefficients (MFCCs),
    which are commonly used in speech and music recognition.
    
    Features:
    - 13 MFCC coefficients (mean and std)
    - Delta MFCCs
    - Delta-delta MFCCs
    - Spectral contrast
    - Chroma features
    
    This is an alternative strategy for comparison with FFT.
    """
    
    DEFAULT_N_MFCC = 13
    DEFAULT_N_MELS = 128
    DEFAULT_HOP_LENGTH = 512
    
    def __init__(
        self,
        n_mfcc: int = DEFAULT_N_MFCC,
        n_mels: int = DEFAULT_N_MELS,
        hop_length: int = DEFAULT_HOP_LENGTH
    ):
        """
        Initialize MFCC strategy.
        
        Args:
            n_mfcc: Number of MFCC coefficients (default: 13)
            n_mels: Number of mel bands (default: 128)
            hop_length: Hop length for STFT (default: 512)
        """
        self._n_mfcc = n_mfcc
        self._n_mels = n_mels
        self._hop_length = hop_length
        # 13 MFCCs * 2 (mean, std) + 13 deltas * 2 + 13 delta-deltas * 2 + 7 contrast + 12 chroma = 97
        self._feature_count = n_mfcc * 6 + 7 + 12
    
    def extract_features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract MFCC-based features from audio.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Feature vector
        """
        try:
            features = []
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=sample_rate,
                n_mfcc=self._n_mfcc,
                n_mels=self._n_mels,
                hop_length=self._hop_length
            )
            
            # MFCC statistics (mean and std)
            features.extend(np.mean(mfccs, axis=1))
            features.extend(np.std(mfccs, axis=1))
            
            # Delta MFCCs
            delta_mfccs = librosa.feature.delta(mfccs)
            features.extend(np.mean(delta_mfccs, axis=1))
            features.extend(np.std(delta_mfccs, axis=1))
            
            # Delta-delta MFCCs
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            features.extend(np.mean(delta2_mfccs, axis=1))
            features.extend(np.std(delta2_mfccs, axis=1))
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(
                y=audio,
                sr=sample_rate,
                hop_length=self._hop_length
            )
            features.extend(np.mean(contrast, axis=1))
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(
                y=audio,
                sr=sample_rate,
                hop_length=self._hop_length
            )
            features.extend(np.mean(chroma, axis=1))
            
            feature_array = np.array(features)
            logger.debug(f"MFCCStrategy: extracted {len(feature_array)} features")
            return feature_array
            
        except Exception as e:
            logger.error(f"MFCCStrategy: feature extraction error: {e}")
            return np.zeros(self._feature_count)
    
    def get_name(self) -> str:
        """Get strategy name."""
        return "MFCC-based"
    
    def get_feature_count(self) -> int:
        """Get expected feature count."""
        return self._feature_count
