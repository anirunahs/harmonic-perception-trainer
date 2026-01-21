"""
ModelManager - Singleton for ML model management.

Pattern: Singleton (Creational)
"""

import threading
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Thread-safe Singleton manager for ML models.
    
    Provides centralized access to IntervalClassifier and FFTFeatureExtractor.
    Models are loaded immediately on first instantiation (eager loading).
    """
    
    _instance: Optional['ModelManager'] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False
    
    def __new__(cls) -> 'ModelManager':
        """Create or return existing instance (thread-safe with double-checked locking)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize manager and load models immediately."""
        if ModelManager._initialized:
            return
            
        with ModelManager._lock:
            if ModelManager._initialized:
                return
                
            self._classifier = None
            self._feature_extractor = None
            self._is_loaded = False
            self._load_error: Optional[str] = None
            
            self._load_models()
            ModelManager._initialized = True
    
    @classmethod
    def get_instance(cls) -> 'ModelManager':
        """Get the singleton instance."""
        return cls()
    
    def _load_models(self) -> None:
        """Load ML models (called on initialization)."""
        try:
            logger.info("ModelManager: Loading ML models...")
            
            from core.feature_extraction import FFTFeatureExtractor
            from core.model_inference import IntervalClassifier
            
            self._feature_extractor = FFTFeatureExtractor()
            logger.info("ModelManager: FFTFeatureExtractor loaded")
            
            self._classifier = IntervalClassifier()
            logger.info("ModelManager: IntervalClassifier loaded")
            
            if self._classifier.is_loaded:
                self._is_loaded = True
                self._load_error = None
                logger.info("ModelManager: All models loaded successfully")
            else:
                self._is_loaded = False
                self._load_error = "Classifier model not loaded"
                logger.error(f"ModelManager: {self._load_error}")
                
        except Exception as e:
            self._is_loaded = False
            self._load_error = str(e)
            logger.error(f"ModelManager: Failed to load models: {e}")
    
    @property
    def is_ready(self) -> bool:
        """Check if models are loaded and ready."""
        return self._is_loaded
    
    @property
    def classifier(self):
        """Get the interval classifier. Raises RuntimeError if not loaded."""
        if not self._is_loaded:
            raise RuntimeError(f"Classifier unavailable: {self._load_error or 'Not loaded'}")
        return self._classifier
    
    @property
    def feature_extractor(self):
        """Get the feature extractor. Raises RuntimeError if not loaded."""
        if not self._is_loaded:
            raise RuntimeError(f"Feature extractor unavailable: {self._load_error or 'Not loaded'}")
        return self._feature_extractor
    
    @property
    def load_error(self) -> Optional[str]:
        """Get load error message if any."""
        return self._load_error
    
    def get_status(self) -> Dict[str, Any]:
        """Get full status of ML components."""
        status = {
            'is_ready': self._is_loaded,
            'load_error': self._load_error,
            'components': {
                'feature_extractor': {
                    'loaded': self._feature_extractor is not None,
                    'type': type(self._feature_extractor).__name__ if self._feature_extractor else None
                },
                'classifier': {
                    'loaded': self._classifier is not None,
                    'type': type(self._classifier).__name__ if self._classifier else None
                }
            }
        }
        
        if self._classifier and hasattr(self._classifier, 'get_model_info'):
            status['model_info'] = self._classifier.get_model_info()
        
        return status
    
    def reload(self) -> bool:
        """Reload models (for recovery after failure)."""
        logger.info("ModelManager: Reloading models...")
        
        self._classifier = None
        self._feature_extractor = None
        self._is_loaded = False
        self._load_error = None
        
        self._load_models()
        return self._is_loaded
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing only)."""
        with cls._lock:
            cls._instance = None
            cls._initialized = False
