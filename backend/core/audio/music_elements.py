"""
Music Element Generators - Abstract Factory pattern implementation.

This module provides abstract interfaces and concrete implementations
for generating intervals and chords with different instruments.

Pattern: Abstract Factory (Creational)
Purpose: Create families of related objects (instrument + music element type)
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np

from .generators import AudioGenerator
from .constants import (
    DEFAULT_SAMPLE_RATE,
    INTERVALS_SEMITONES,
    CHORDS_SEMITONES,
    get_note_frequency,
    get_target_note,
    CHROMATIC_SCALE,
)
from .utils import combine_tones_harmonic


class IntervalGenerator(ABC):
    """
    Abstract interface for generating interval audio.
    
    Pattern: Abstract Factory - Abstract Product
    """
    
    def __init__(self, audio_generator: AudioGenerator):
        """
        Initialize interval generator with audio generator.
        
        Args:
            audio_generator: AudioGenerator instance for tone generation
        """
        self.audio_generator = audio_generator
        self.sample_rate = audio_generator.sample_rate
    
    @abstractmethod
    def generate_harmonic_interval(self, base_note: str, interval_type: str, 
                                   octave: int = 4, duration: float = 2.0) -> np.ndarray:
        """
        Generate harmonic interval (two notes played simultaneously).
        
        Args:
            base_note: Base note name (e.g., 'C')
            interval_type: Interval type (e.g., 'major_third')
            octave: Octave number
            duration: Duration in seconds
            
        Returns:
            Audio array with combined tones
        """
        pass
    
    @abstractmethod
    def generate_melodic_interval(self, base_note: str, interval_type: str,
                                  octave: int = 4, duration: float = 2.0,
                                  gap: float = 0.1) -> np.ndarray:
        """
        Generate melodic interval (two notes played sequentially).
        
        Args:
            base_note: Base note name
            interval_type: Interval type
            octave: Octave number
            duration: Duration of each note
            gap: Gap between notes in seconds
            
        Returns:
            Audio array with sequential tones
        """
        pass
    
    def get_instrument_name(self) -> str:
        """Get the name of the instrument."""
        return self.audio_generator.get_instrument_name()


class ChordGenerator(ABC):
    """
    Abstract interface for generating chord audio.
    
    Pattern: Abstract Factory - Abstract Product
    """
    
    def __init__(self, audio_generator: AudioGenerator):
        """
        Initialize chord generator with audio generator.
        
        Args:
            audio_generator: AudioGenerator instance for tone generation
        """
        self.audio_generator = audio_generator
        self.sample_rate = audio_generator.sample_rate
    
    @abstractmethod
    def generate_chord(self, root_note: str, chord_type: str,
                       octave: int = 4, duration: float = 2.0) -> np.ndarray:
        """
        Generate chord audio (multiple notes played simultaneously).
        
        Args:
            root_note: Root note name (e.g., 'C')
            chord_type: Chord type (e.g., 'major', 'minor')
            octave: Octave number for root note
            duration: Duration in seconds
            
        Returns:
            Audio array with combined chord tones
        """
        pass
    
    def get_instrument_name(self) -> str:
        """Get the name of the instrument."""
        return self.audio_generator.get_instrument_name()


class PianoIntervalGenerator(IntervalGenerator):
    """
    Piano-specific interval generator.
    
    Pattern: Abstract Factory - Concrete Product
    """
    
    def generate_harmonic_interval(self, base_note: str, interval_type: str,
                                   octave: int = 4, duration: float = 2.0) -> np.ndarray:
        """Generate harmonic interval with piano timbre."""
        semitones = INTERVALS_SEMITONES.get(interval_type)
        if semitones is None:
            raise ValueError(f"Unknown interval type: {interval_type}")
        
        # Get frequencies
        base_freq = get_note_frequency(base_note, octave)
        target_note = get_target_note(base_note, semitones)
        target_freq = get_note_frequency(target_note, octave)
        
        # Generate tones
        base_tone = self.audio_generator.generate_tone(base_freq, duration)
        target_tone = self.audio_generator.generate_tone(target_freq, duration)
        
        # Combine harmonically
        return combine_tones_harmonic(base_tone, target_tone)
    
    def generate_melodic_interval(self, base_note: str, interval_type: str,
                                  octave: int = 4, duration: float = 2.0,
                                  gap: float = 0.1) -> np.ndarray:
        """Generate melodic interval with piano timbre."""
        semitones = INTERVALS_SEMITONES.get(interval_type)
        if semitones is None:
            raise ValueError(f"Unknown interval type: {interval_type}")
        
        # Get frequencies
        base_freq = get_note_frequency(base_note, octave)
        target_note = get_target_note(base_note, semitones)
        target_freq = get_note_frequency(target_note, octave)
        
        # Generate tones
        base_tone = self.audio_generator.generate_tone(base_freq, duration)
        target_tone = self.audio_generator.generate_tone(target_freq, duration)
        
        # Combine melodically with gap
        gap_samples = int(self.sample_rate * gap)
        gap_audio = np.zeros(gap_samples)
        return np.concatenate([base_tone, gap_audio, target_tone])


class GuitarIntervalGenerator(IntervalGenerator):
    """
    Guitar-specific interval generator.
    
    Pattern: Abstract Factory - Concrete Product
    """
    
    def generate_harmonic_interval(self, base_note: str, interval_type: str,
                                   octave: int = 4, duration: float = 2.0) -> np.ndarray:
        """Generate harmonic interval with guitar timbre."""
        semitones = INTERVALS_SEMITONES.get(interval_type)
        if semitones is None:
            raise ValueError(f"Unknown interval type: {interval_type}")
        
        # Get frequencies
        base_freq = get_note_frequency(base_note, octave)
        target_note = get_target_note(base_note, semitones)
        target_freq = get_note_frequency(target_note, octave)
        
        # Generate tones
        base_tone = self.audio_generator.generate_tone(base_freq, duration)
        target_tone = self.audio_generator.generate_tone(target_freq, duration)
        
        # Combine harmonically
        return combine_tones_harmonic(base_tone, target_tone)
    
    def generate_melodic_interval(self, base_note: str, interval_type: str,
                                  octave: int = 4, duration: float = 2.0,
                                  gap: float = 0.1) -> np.ndarray:
        """Generate melodic interval with guitar timbre."""
        semitones = INTERVALS_SEMITONES.get(interval_type)
        if semitones is None:
            raise ValueError(f"Unknown interval type: {interval_type}")
        
        # Get frequencies
        base_freq = get_note_frequency(base_note, octave)
        target_note = get_target_note(base_note, semitones)
        target_freq = get_note_frequency(target_note, octave)
        
        # Generate tones
        base_tone = self.audio_generator.generate_tone(base_freq, duration)
        target_tone = self.audio_generator.generate_tone(target_freq, duration)
        
        # Combine melodically with gap
        gap_samples = int(self.sample_rate * gap)
        gap_audio = np.zeros(gap_samples)
        return np.concatenate([base_tone, gap_audio, target_tone])


class PianoChordGenerator(ChordGenerator):
    """
    Piano-specific chord generator.
    
    Pattern: Abstract Factory - Concrete Product
    """
    
    def generate_chord(self, root_note: str, chord_type: str,
                      octave: int = 4, duration: float = 2.0) -> np.ndarray:
        """Generate chord with piano timbre."""
        semitones_list = CHORDS_SEMITONES.get(chord_type)
        if semitones_list is None:
            raise ValueError(f"Unknown chord type: {chord_type}")
        
        # Generate tones for each note in chord
        tones = []
        for semitones in semitones_list:
            if semitones == 0:
                note = root_note
                note_octave = octave
            else:
                note = get_target_note(root_note, semitones)
                # Calculate octave (if semitones >= 12, move to next octave)
                note_octave = octave + (semitones // 12)
            
            freq = get_note_frequency(note, note_octave)
            tone = self.audio_generator.generate_tone(freq, duration)
            tones.append(tone)
        
        # Combine all tones harmonically
        if not tones:
            raise ValueError("Chord must have at least one note")
        
        combined = tones[0]
        for tone in tones[1:]:
            combined = combine_tones_harmonic(combined, tone)
        
        return combined


class GuitarChordGenerator(ChordGenerator):
    """
    Guitar-specific chord generator.
    
    Pattern: Abstract Factory - Concrete Product
    """
    
    def generate_chord(self, root_note: str, chord_type: str,
                      octave: int = 4, duration: float = 2.0) -> np.ndarray:
        """Generate chord with guitar timbre."""
        semitones_list = CHORDS_SEMITONES.get(chord_type)
        if semitones_list is None:
            raise ValueError(f"Unknown chord type: {chord_type}")
        
        # Generate tones for each note in chord
        tones = []
        for semitones in semitones_list:
            if semitones == 0:
                note = root_note
                note_octave = octave
            else:
                note = get_target_note(root_note, semitones)
                # Calculate octave (if semitones >= 12, move to next octave)
                note_octave = octave + (semitones // 12)
            
            freq = get_note_frequency(note, note_octave)
            tone = self.audio_generator.generate_tone(freq, duration)
            tones.append(tone)
        
        # Combine all tones harmonically
        if not tones:
            raise ValueError("Chord must have at least one note")
        
        combined = tones[0]
        for tone in tones[1:]:
            combined = combine_tones_harmonic(combined, tone)
        
        return combined
