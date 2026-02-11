"""
Instrument Factory - Factory Method pattern implementation.

This module provides a factory for creating audio generators
based on instrument type. New instruments can be registered
dynamically.
"""

from typing import Dict, Type, Optional, Any
from .generators import AudioGenerator, SynthGenerator
from .constants import DEFAULT_SAMPLE_RATE, ADSREnvelope


class InstrumentFactory:
    """
    Factory for creating audio generators.
    
    Implements the Factory Method pattern to create instrument-specific
    audio generators. Supports registration of new instrument types.
    
    Usage:
        factory = InstrumentFactory()
        synth = factory.create_generator('synth')
        tone = synth.generate_tone(440.0, duration=2.0)
    """
    
    # Registry of available generator classes (synth only; piano/guitar use client-side samples)
    _generators: Dict[str, Type[AudioGenerator]] = {
        'synth': SynthGenerator,
        'synth_saw': SynthGenerator,
        'synth_square': SynthGenerator,
        'synth_sine': SynthGenerator,
        'synth_triangle': SynthGenerator,
    }
    
    # Default parameters for specific instruments
    _default_params: Dict[str, Dict[str, Any]] = {
        'synth_saw': {'wave_type': 'saw'},
        'synth_square': {'wave_type': 'square'},
        'synth_sine': {'wave_type': 'sine'},
        'synth_triangle': {'wave_type': 'triangle'},
    }
    
    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE):
        """
        Initialize the factory.
        
        Args:
            sample_rate: Default sample rate for all generators
        """
        self.sample_rate = sample_rate
    
    @classmethod
    def register(cls, instrument_name: str, generator_class: Type[AudioGenerator], 
                 default_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a new instrument type.
        
        Args:
            instrument_name: Name identifier for the instrument
            generator_class: AudioGenerator subclass for this instrument
            default_params: Optional default parameters for the generator
        
        Raises:
            TypeError: If generator_class is not a subclass of AudioGenerator
        """
        if not issubclass(generator_class, AudioGenerator):
            raise TypeError(f"{generator_class} must be a subclass of AudioGenerator")
        
        cls._generators[instrument_name] = generator_class
        if default_params:
            cls._default_params[instrument_name] = default_params
    
    @classmethod
    def unregister(cls, instrument_name: str) -> bool:
        """
        Unregister an instrument type.
        
        Args:
            instrument_name: Name of the instrument to remove
        
        Returns:
            True if instrument was removed, False if it didn't exist
        """
        if instrument_name in cls._generators:
            del cls._generators[instrument_name]
            cls._default_params.pop(instrument_name, None)
            return True
        return False
    
    @classmethod
    def get_available_instruments(cls) -> list:
        """
        Get list of available instrument names.
        
        Returns:
            List of registered instrument names
        """
        return list(cls._generators.keys())
    
    def create_generator(self, instrument: str, 
                         adsr: Optional[ADSREnvelope] = None,
                         **kwargs) -> AudioGenerator:
        """
        Create an audio generator for the specified instrument.
        
        This is the Factory Method - it creates the appropriate
        concrete generator based on the instrument name.
        
        Args:
            instrument: Name of the instrument ('synth', 'synth_saw', etc.)
            adsr: Optional custom ADSR envelope
            **kwargs: Additional instrument-specific parameters
        
        Returns:
            AudioGenerator instance for the specified instrument
        
        Raises:
            ValueError: If instrument type is not recognized
        
        Examples:
            >>> factory = InstrumentFactory()
            >>> synth = factory.create_generator('synth', wave_type='square')
        """
        instrument = instrument.lower()
        
        if instrument not in self._generators:
            available = ', '.join(self.get_available_instruments())
            raise ValueError(
                f"Unknown instrument: '{instrument}'. "
                f"Available instruments: {available}"
            )
        
        generator_class = self._generators[instrument]
        
        # Merge default params with provided kwargs
        params = {'sample_rate': self.sample_rate}
        if adsr is not None:
            params['adsr'] = adsr
        
        # Add default params for this instrument
        if instrument in self._default_params:
            params.update(self._default_params[instrument])
        
        # Override with user-provided kwargs
        params.update(kwargs)
        
        return generator_class(**params)
    
    def create_synth(self, wave_type: str = 'saw',
                     adsr: Optional[ADSREnvelope] = None) -> SynthGenerator:
        """
        Create a synth generator.
        
        Args:
            wave_type: Type of waveform ('saw', 'square', 'sine', 'triangle')
            adsr: Optional custom ADSR envelope
        
        Returns:
            SynthGenerator instance
        """
        return self.create_generator('synth', adsr=adsr, wave_type=wave_type)


# Global factory instance for convenience
_default_factory: Optional[InstrumentFactory] = None


def get_factory(sample_rate: int = DEFAULT_SAMPLE_RATE) -> InstrumentFactory:
    """
    Get the default factory instance.
    
    Creates a singleton-like default factory for convenience.
    
    Args:
        sample_rate: Sample rate for the factory
    
    Returns:
        InstrumentFactory instance
    """
    global _default_factory
    if _default_factory is None or _default_factory.sample_rate != sample_rate:
        _default_factory = InstrumentFactory(sample_rate)
    return _default_factory
