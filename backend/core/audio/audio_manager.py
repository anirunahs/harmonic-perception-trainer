"""
Audio Manager - Singleton pattern for audio resource management.

Pattern: Singleton (Creational)
Purpose: Ensure single instance for managing audio resources and caching.

This manager provides centralized access to audio generation resources,
caching generators and managing audio file operations.
"""

import threading
import logging
from typing import Dict, Optional
import os
from django.conf import settings

from .factory import InstrumentFactory
from .abstract_factory import MusicElementFactoryRegistry, MusicElementFactory
from .generators import AudioGenerator

logger = logging.getLogger(__name__)


class AudioManager:
    """
    Singleton manager for audio resources.
    
    Pattern: Singleton (Creational)
    
    Manages:
    - Audio generators (cached)
    - Music element factories (cached)
    - Audio file operations
    """
    
    _instance: Optional['AudioManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Create or return existing singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize audio manager (only once)."""
        if self._initialized:
            return
        
        self._instrument_factory = InstrumentFactory()
        self._music_element_factories: Dict[str, MusicElementFactory] = {}
        self._audio_generators: Dict[str, AudioGenerator] = {}
        self._initialized = True
        
        logger.info("AudioManager initialized")
    
    def get_instrument_factory(self) -> InstrumentFactory:
        """
        Get the instrument factory instance.
        
        Returns:
            InstrumentFactory instance
        """
        return self._instrument_factory
    
    def get_music_element_factory(self, instrument: str) -> MusicElementFactory:
        """
        Get or create music element factory for instrument.
        
        Caches factories to avoid recreating them.
        
        Args:
            instrument: Instrument name ('synth')
            
        Returns:
            MusicElementFactory instance
        """
        instrument = instrument.lower()
        
        if instrument not in self._music_element_factories:
            factory = MusicElementFactoryRegistry.create_factory(
                instrument,
                self._instrument_factory
            )
            self._music_element_factories[instrument] = factory
            logger.debug(f"Created music element factory for {instrument}")
        
        return self._music_element_factories[instrument]
    
    def get_audio_generator(self, instrument: str) -> AudioGenerator:
        """
        Get or create audio generator for instrument.
        
        Caches generators to avoid recreating them.
        
        Args:
            instrument: Instrument name
            
        Returns:
            AudioGenerator instance
        """
        instrument = instrument.lower()
        
        if instrument not in self._audio_generators:
            generator = self._instrument_factory.create_generator(instrument)
            self._audio_generators[instrument] = generator
            logger.debug(f"Created audio generator for {instrument}")
        
        return self._audio_generators[instrument]
    
    def clear_cache(self):
        """Clear cached generators and factories."""
        self._music_element_factories.clear()
        self._audio_generators.clear()
        logger.info("AudioManager cache cleared")
    
    def get_audio_directory(self, subdirectory: str = '') -> str:
        """
        Get audio directory path.
        
        Args:
            subdirectory: Optional subdirectory name
            
        Returns:
            Full path to audio directory
        """
        base_dir = os.path.join(settings.MEDIA_ROOT, 'audio')
        if subdirectory:
            return os.path.join(base_dir, subdirectory)
        return base_dir
    
    def ensure_audio_directory(self, subdirectory: str = '') -> str:
        """
        Ensure audio directory exists and return path.
        
        Args:
            subdirectory: Optional subdirectory name
            
        Returns:
            Full path to audio directory
        """
        directory = self.get_audio_directory(subdirectory)
        os.makedirs(directory, exist_ok=True)
        return directory


# Global singleton instance accessor
def get_audio_manager() -> AudioManager:
    """
    Get the AudioManager singleton instance.
    
    Returns:
        AudioManager instance
    """
    return AudioManager()
