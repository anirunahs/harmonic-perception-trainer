"""
Audio generators implementing Factory Method pattern.

This module provides abstract base class and concrete implementations
for generating audio tones with different instrument timbres.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np

from .constants import (
    DEFAULT_SAMPLE_RATE,
    ADSREnvelope,
    InstrumentEnvelopes,
    InstrumentHarmonics,
)


class AudioGenerator(ABC):
    """
    Abstract base class for audio generators.
    
    Implements Factory Method pattern - each concrete generator
    knows how to create its specific type of audio tone.
    
    Attributes:
        sample_rate: Sample rate in Hz for audio generation
        adsr: ADSR envelope parameters for amplitude shaping
    """
    
    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, adsr: Optional[ADSREnvelope] = None):
        self.sample_rate = sample_rate
        self.adsr = adsr or self._get_default_adsr()
    
    @abstractmethod
    def _get_default_adsr(self) -> ADSREnvelope:
        """Return default ADSR envelope for this instrument."""
        pass
    
    @abstractmethod
    def _get_harmonics(self) -> List[Tuple[int, float]]:
        """Return harmonic content definition for this instrument."""
        pass
    
    @abstractmethod
    def get_instrument_name(self) -> str:
        """Return the name of this instrument."""
        pass
    
    def generate_tone(self, frequency: float, duration: float = 2.0) -> np.ndarray:
        """
        Generate an audio tone at the specified frequency.
        
        Args:
            frequency: Frequency in Hz
            duration: Duration in seconds
        
        Returns:
            numpy array of audio samples (normalized to -1.0 to 1.0)
        """
        # Generate time array
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Generate waveform with harmonics
        wave = self._generate_waveform(t, frequency)
        
        # Apply ADSR envelope
        envelope = self._create_envelope(num_samples)
        wave *= envelope
        
        # Normalize to prevent clipping
        wave = self._normalize(wave)
        
        return wave
    
    def _generate_waveform(self, t: np.ndarray, frequency: float) -> np.ndarray:
        """
        Generate the raw waveform with harmonic content.
        
        Args:
            t: Time array
            frequency: Fundamental frequency in Hz
        
        Returns:
            Raw waveform array
        """
        wave = np.zeros_like(t)
        harmonics = self._get_harmonics()
        
        for harmonic_num, amplitude in harmonics:
            wave += amplitude * np.sin(2 * np.pi * frequency * harmonic_num * t)
        
        return wave
    
    def _create_envelope(self, num_samples: int) -> np.ndarray:
        """
        Create ADSR envelope for amplitude shaping.
        
        Args:
            num_samples: Total number of samples
        
        Returns:
            Envelope array (0.0 to 1.0)
        """
        envelope = np.ones(num_samples)
        
        attack_samples = int(self.adsr.attack * self.sample_rate)
        decay_samples = int(self.adsr.decay * self.sample_rate)
        release_samples = int(self.adsr.release * self.sample_rate)
        
        # Attack phase
        if attack_samples > 0 and attack_samples < num_samples:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay phase
        decay_end = attack_samples + decay_samples
        if decay_samples > 0 and decay_end < num_samples:
            envelope[attack_samples:decay_end] = np.linspace(1, self.adsr.sustain, decay_samples)
        
        # Sustain phase
        sustain_start = decay_end
        release_start = num_samples - release_samples
        if sustain_start < release_start:
            envelope[sustain_start:release_start] = self.adsr.sustain
        
        # Release phase
        if release_samples > 0 and release_start >= 0:
            envelope[release_start:] = np.linspace(self.adsr.sustain, 0, release_samples)
        
        return envelope
    
    def _normalize(self, wave: np.ndarray, target_amplitude: float = 0.8) -> np.ndarray:
        """
        Normalize waveform to target amplitude.
        
        Args:
            wave: Input waveform
            target_amplitude: Maximum amplitude (0.0 to 1.0)
        
        Returns:
            Normalized waveform
        """
        max_val = np.max(np.abs(wave))
        if max_val > 0:
            wave = wave / max_val * target_amplitude
        return wave


class PianoGenerator(AudioGenerator):
    """
    Piano sound generator.
    
    Generates a piano-like tone using additive synthesis with
    harmonics that simulate piano timbre.
    """
    
    def _get_default_adsr(self) -> ADSREnvelope:
        return InstrumentEnvelopes.PIANO
    
    def _get_harmonics(self) -> List[Tuple[int, float]]:
        return InstrumentHarmonics.PIANO
    
    def get_instrument_name(self) -> str:
        return "piano"
    
    def _generate_waveform(self, t: np.ndarray, frequency: float) -> np.ndarray:
        """
        Generate piano waveform with slight detuning for realism.
        
        Piano strings have slight inharmonicity - upper harmonics
        are slightly sharper than perfect integer multiples.
        """
        wave = np.zeros_like(t)
        harmonics = self._get_harmonics()
        
        for harmonic_num, amplitude in harmonics:
            # Slight inharmonicity factor for piano realism
            inharmonicity = 1.0 + (harmonic_num - 1) * 0.0001
            freq = frequency * harmonic_num * inharmonicity
            wave += amplitude * np.sin(2 * np.pi * freq * t)
        
        return wave


class GuitarGenerator(AudioGenerator):
    """
    Guitar sound generator.
    
    Uses Karplus-Strong-inspired synthesis combined with
    additive synthesis for a guitar-like tone.
    """
    
    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, 
                 adsr: Optional[ADSREnvelope] = None,
                 pluck_position: float = 0.2):
        """
        Initialize guitar generator.
        
        Args:
            sample_rate: Sample rate in Hz
            adsr: ADSR envelope parameters
            pluck_position: Position of pluck (0.0 to 0.5), affects tone brightness
        """
        super().__init__(sample_rate, adsr)
        self.pluck_position = pluck_position
    
    def _get_default_adsr(self) -> ADSREnvelope:
        return InstrumentEnvelopes.GUITAR
    
    def _get_harmonics(self) -> List[Tuple[int, float]]:
        return InstrumentHarmonics.GUITAR
    
    def get_instrument_name(self) -> str:
        return "guitar"
    
    def _generate_waveform(self, t: np.ndarray, frequency: float) -> np.ndarray:
        """
        Generate guitar waveform with pluck characteristic.
        
        Simulates the effect of plucking position on harmonic content.
        Harmonics near nodes of the pluck position are suppressed.
        """
        wave = np.zeros_like(t)
        harmonics = self._get_harmonics()
        
        for harmonic_num, base_amplitude in harmonics:
            # Suppress harmonics based on pluck position
            # (simulates nodes at pluck point)
            pluck_factor = np.sin(np.pi * harmonic_num * self.pluck_position)
            amplitude = base_amplitude * abs(pluck_factor)
            
            # Add slight frequency wobble for realism
            vibrato = 1.0 + 0.002 * np.sin(2 * np.pi * 5 * t)
            freq = frequency * harmonic_num * vibrato
            
            wave += amplitude * np.sin(2 * np.pi * freq * t)
        
        # Add initial transient (pick attack)
        attack_samples = int(0.01 * self.sample_rate)
        if attack_samples > 0 and attack_samples < len(t):
            noise = np.random.uniform(-0.1, 0.1, attack_samples)
            noise *= np.linspace(1, 0, attack_samples)
            wave[:attack_samples] += noise
        
        return wave


class SynthGenerator(AudioGenerator):
    """
    Synthesizer sound generator.
    
    Generates various synth waveforms (saw, square, etc.)
    with configurable wave type.
    """
    
    WAVE_TYPES = ['saw', 'square', 'sine', 'triangle']
    
    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE,
                 adsr: Optional[ADSREnvelope] = None,
                 wave_type: str = 'saw'):
        """
        Initialize synth generator.
        
        Args:
            sample_rate: Sample rate in Hz
            adsr: ADSR envelope parameters
            wave_type: Type of waveform ('saw', 'square', 'sine', 'triangle')
        """
        super().__init__(sample_rate, adsr)
        if wave_type not in self.WAVE_TYPES:
            raise ValueError(f"Unknown wave type: {wave_type}. Must be one of {self.WAVE_TYPES}")
        self.wave_type = wave_type
    
    def _get_default_adsr(self) -> ADSREnvelope:
        return InstrumentEnvelopes.SYNTH_LEAD
    
    def _get_harmonics(self) -> List[Tuple[int, float]]:
        if self.wave_type == 'saw':
            return InstrumentHarmonics.SYNTH_SAW
        elif self.wave_type == 'square':
            return InstrumentHarmonics.SYNTH_SQUARE
        elif self.wave_type == 'sine':
            return [(1, 1.0)]  # Pure sine wave
        elif self.wave_type == 'triangle':
            # Triangle wave: odd harmonics with 1/n² amplitude
            return [(i, 1.0 / (i ** 2)) for i in range(1, 10, 2)]
        return [(1, 1.0)]
    
    def get_instrument_name(self) -> str:
        return f"synth_{self.wave_type}"
    
    def _generate_waveform(self, t: np.ndarray, frequency: float) -> np.ndarray:
        """
        Generate synth waveform.
        
        For 'sine' type, uses pure sine wave.
        For others, uses additive synthesis with appropriate harmonics.
        """
        if self.wave_type == 'sine':
            # Pure sine wave
            return np.sin(2 * np.pi * frequency * t)
        
        # Use parent implementation for harmonic-based synthesis
        return super()._generate_waveform(t, frequency)
