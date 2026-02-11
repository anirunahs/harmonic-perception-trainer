"""
Abstract Factory for Music Elements.

Pattern: Abstract Factory (Creational)
Purpose: Create families of related objects (instrument + music element type)

This factory creates interval and chord generators for specific instruments,
ensuring that all generators for a given instrument are compatible.
"""

from abc import ABC, abstractmethod
from typing import Optional

from .factory import InstrumentFactory
from .generators import AudioGenerator
from .music_elements import (
    IntervalGenerator,
    ChordGenerator,
    SynthIntervalGenerator,
    SynthChordGenerator,
)


class MusicElementFactory(ABC):
    """
    Abstract Factory for creating music element generators.
    
    Pattern: Abstract Factory (Creational) - Abstract Factory
    
    Each concrete factory creates a family of related products:
    - IntervalGenerator for a specific instrument
    - ChordGenerator for the same instrument
    
    This ensures that interval and chord generators are compatible
    (use the same instrument timbre).
    """
    
    def __init__(self, instrument_factory: Optional[InstrumentFactory] = None):
        """
        Initialize factory with instrument factory.
        
        Args:
            instrument_factory: InstrumentFactory instance (creates if None)
        """
        self._instrument_factory = instrument_factory or InstrumentFactory()
        self._audio_generator: Optional[AudioGenerator] = None
    
    def _get_audio_generator(self) -> AudioGenerator:
        """
        Get or create audio generator for this factory's instrument.
        
        Returns:
            AudioGenerator instance
        """
        if self._audio_generator is None:
            self._audio_generator = self._create_audio_generator()
        return self._audio_generator
    
    @abstractmethod
    def _create_audio_generator(self) -> AudioGenerator:
        """
        Create audio generator for this factory's instrument.
        
        Returns:
            AudioGenerator instance
        """
        pass
    
    @abstractmethod
    def create_interval_generator(self) -> IntervalGenerator:
        """
        Create interval generator for this factory's instrument.
        
        Returns:
            IntervalGenerator instance
        """
        pass
    
    @abstractmethod
    def create_chord_generator(self) -> ChordGenerator:
        """
        Create chord generator for this factory's instrument.
        
        Returns:
            ChordGenerator instance
        """
        pass
    
    @abstractmethod
    def get_instrument_name(self) -> str:
        """
        Get the name of the instrument this factory creates generators for.
        
        Returns:
            Instrument name (e.g., 'synth')
        """
        pass


class SynthMusicElementFactory(MusicElementFactory):
    """
    Concrete Factory for Synth music elements.
    
    Pattern: Abstract Factory - Concrete Factory
    """

    def _create_audio_generator(self) -> AudioGenerator:
        """Create synth audio generator."""
        return self._instrument_factory.create_synth()

    def create_interval_generator(self) -> IntervalGenerator:
        """Create synth interval generator."""
        return SynthIntervalGenerator(self._get_audio_generator())

    def create_chord_generator(self) -> ChordGenerator:
        """Create synth chord generator."""
        return SynthChordGenerator(self._get_audio_generator())

    def get_instrument_name(self) -> str:
        """Get instrument name."""
        return 'synth'


class MusicElementFactoryRegistry:
    """
    Registry for music element factories.
    
    Provides a way to get factories by instrument name.
    """
    
    _factories = {
        'synth': SynthMusicElementFactory,
    }

    @classmethod
    def create_factory(cls, instrument: str,
                      instrument_factory: Optional[InstrumentFactory] = None) -> MusicElementFactory:
        """
        Create music element factory for specified instrument.

        Args:
            instrument: Instrument name ('synth')
            instrument_factory: Optional InstrumentFactory instance
            
        Returns:
            MusicElementFactory instance
            
        Raises:
            ValueError: If instrument is not supported
        """
        instrument = instrument.lower()
        factory_class = cls._factories.get(instrument)
        
        if factory_class is None:
            available = ', '.join(cls._factories.keys())
            raise ValueError(
                f"Unknown instrument: '{instrument}'. "
                f"Available instruments: {available}"
            )
        
        return factory_class(instrument_factory)
    
    @classmethod
    def register_factory(cls, instrument: str, factory_class: type):
        """
        Register a new music element factory.
        
        Args:
            instrument: Instrument name
            factory_class: MusicElementFactory subclass
        """
        if not issubclass(factory_class, MusicElementFactory):
            raise TypeError(f"{factory_class} must be a subclass of MusicElementFactory")
        cls._factories[instrument.lower()] = factory_class
    
    @classmethod
    def get_available_instruments(cls) -> list:
        """Get list of available instrument names."""
        return list(cls._factories.keys())
