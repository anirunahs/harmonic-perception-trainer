"""
AudioProcessingFacade - Simplified interface for audio recognition.

Pattern: Facade (Structural)

Provides a single entry point for the complex audio recognition subsystem,
hiding the complexity of:
- Audio loading and decoding
- Audio preprocessing (normalization, filtering)
- Feature extraction
- Classification
- Result formatting
"""

import time
import io
import base64
import logging
import numpy as np
import librosa
from typing import Optional, Tuple

from core.ml import ModelManager
from .result import RecognitionResult

logger = logging.getLogger(__name__)


class AudioProcessingFacade:
    """
    Facade for audio processing and interval recognition.
    
    Simplifies the recognition process to a single method call,
    encapsulating all the complexity of the subsystem.
    
    Usage:
        facade = AudioProcessingFacade()
        result = facade.recognize_interval(audio_base64)
        
        if result.success:
            print(f"Detected: {result.best_prediction.interval}")
        else:
            print(f"Error: {result.error}")
    """
    
    DEFAULT_SAMPLE_RATE = 22050
    TARGET_DURATION = 2.0  # seconds
    
    def __init__(self, sample_rate: int = None):
        """
        Initialize facade.
        
        Args:
            sample_rate: Sample rate for audio processing (default: 22050)
        """
        self._sample_rate = sample_rate or self.DEFAULT_SAMPLE_RATE
        self._model_manager = ModelManager.get_instance()
    
    @property
    def is_ready(self) -> bool:
        """Check if the recognition system is ready."""
        return self._model_manager.is_ready
    
    @property
    def model_error(self) -> Optional[str]:
        """Get model loading error if any."""
        return self._model_manager.load_error
    
    def recognize_interval(self, audio_base64: str) -> RecognitionResult:
        """
        Recognize musical interval from base64 encoded audio.
        
        This is the main entry point that orchestrates:
        1. Audio decoding
        2. Preprocessing
        3. Feature extraction
        4. Classification
        5. Result formatting
        
        Args:
            audio_base64: Base64 encoded audio data
            
        Returns:
            RecognitionResult with prediction or error
        """
        start_time = time.time()
        
        # Check if model is ready
        if not self._model_manager.is_ready:
            return RecognitionResult.error_result(
                error=self._model_manager.load_error or 'Model not loaded',
                error_code='model_unavailable'
            )
        
        # Step 1: Decode audio
        audio, sr = self._decode_audio(audio_base64)
        if audio is None:
            return RecognitionResult.error_result(
                error='Failed to decode audio data',
                error_code='decode_error'
            )
        
        # Step 2: Validate audio
        if len(audio) == 0:
            return RecognitionResult.error_result(
                error='Empty audio data',
                error_code='empty_audio'
            )
        
        audio_duration = len(audio) / sr
        
        # Step 3: Preprocess audio
        try:
            processed_audio = self._preprocess_audio(audio, sr)
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            return RecognitionResult.error_result(
                error='Audio processing failed',
                error_code='processing_error'
            )
        
        # Step 4: Extract features
        try:
            features = self._model_manager.feature_extractor.extract_features(processed_audio)
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return RecognitionResult.error_result(
                error='Feature extraction failed',
                error_code='extraction_error'
            )
        
        # Step 5: Classify
        try:
            prediction = self._model_manager.classifier.predict(features)
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return RecognitionResult.error_result(
                error='Classification failed',
                error_code='classification_error'
            )
        
        processing_time = time.time() - start_time
        
        # Step 6: Create result
        return RecognitionResult.from_prediction(
            prediction=prediction,
            processing_time=processing_time,
            audio_duration=audio_duration
        )
    
    def _decode_audio(self, audio_base64: str) -> Tuple[Optional[np.ndarray], int]:
        """
        Decode base64 audio to numpy array.
        
        Args:
            audio_base64: Base64 encoded audio
            
        Returns:
            Tuple of (audio array, sample rate) or (None, 0) on error
        """
        try:
            audio_bytes = base64.b64decode(audio_base64)
            audio_buffer = io.BytesIO(audio_bytes)
            audio, sr = librosa.load(audio_buffer, sr=self._sample_rate, mono=True)
            return audio, sr
        except Exception as e:
            logger.error(f"Audio decode error: {e}")
            return None, 0
    
    def _preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Preprocess audio for recognition.
        
        Steps:
        1. Remove DC offset
        2. Normalize amplitude
        3. Apply soft limiting
        4. Extract best segment
        
        Args:
            audio: Raw audio array
            sr: Sample rate
            
        Returns:
            Preprocessed audio array
        """
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # Normalize to peak
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95
        
        # Apply gain normalization
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            current_rms_db = 20 * np.log10(rms)
            target_rms_db = -18
            gain_db = min(target_rms_db - current_rms_db, 20)
            
            if gain_db > 0:
                gain_linear = 10**(gain_db / 20)
                audio = audio * gain_linear
                
                # Prevent clipping
                peak = np.max(np.abs(audio))
                if peak > 0.95:
                    audio = audio * (0.90 / peak)
        
        # Soft limiting
        audio = np.tanh(audio)
        
        # Extract best segment
        audio = self._extract_best_segment(audio, sr)
        
        return audio
    
    def _extract_best_segment(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract the segment with highest energy.
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Best segment of target duration
        """
        target_samples = int(self.TARGET_DURATION * sr)
        
        if len(audio) >= target_samples:
            best_start = 0
            best_energy = 0
            step = max(target_samples // 10, 1)
            
            for start in range(0, len(audio) - target_samples + 1, step):
                segment = audio[start:start + target_samples]
                energy = np.mean(segment**2)
                if energy > best_energy:
                    best_energy = energy
                    best_start = start
            
            return audio[best_start:best_start + target_samples]
        else:
            # Pad if too short
            return np.pad(audio, (0, target_samples - len(audio)), mode='constant')
    
    def analyze_audio_quality(self, audio_base64: str) -> dict:
        """
        Analyze audio quality without recognition.
        
        Args:
            audio_base64: Base64 encoded audio
            
        Returns:
            Dict with quality metrics and recommendations
        """
        audio, sr = self._decode_audio(audio_base64)
        if audio is None:
            return {'error': 'Failed to decode audio'}
        
        duration = len(audio) / sr
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        
        rms_db = 20 * np.log10(rms + 1e-10)
        peak_db = 20 * np.log10(peak + 1e-10)
        
        # Determine quality
        if rms_db > -25:
            quality = 'good'
            quality_score = 0.9
        elif rms_db > -40:
            quality = 'fair'
            quality_score = 0.6
        else:
            quality = 'poor'
            quality_score = 0.3
        
        # Generate recommendations
        recommendations = []
        
        if rms_db < -40:
            recommendations.append({
                'type': 'volume',
                'message': f'Recording too quiet ({rms_db:.1f}dB). Move closer to microphone.',
                'priority': 'high'
            })
        
        if duration < 1.5:
            recommendations.append({
                'type': 'duration',
                'message': 'Recording too short. Recommended: 2-3 seconds.',
                'priority': 'medium'
            })
        
        if peak > 0.99:
            recommendations.append({
                'type': 'clipping',
                'message': 'Audio clipping detected. Reduce recording volume.',
                'priority': 'high'
            })
        
        if not recommendations:
            recommendations.append({
                'type': 'quality',
                'message': 'Good recording quality.',
                'priority': 'info'
            })
        
        return {
            'duration': round(duration, 2),
            'rms_db': round(rms_db, 1),
            'peak_db': round(peak_db, 1),
            'quality': quality,
            'quality_score': quality_score,
            'recommendations': recommendations
        }
