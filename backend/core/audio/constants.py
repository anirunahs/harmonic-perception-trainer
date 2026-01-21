"""
Audio constants for sound generation.

Contains note frequencies, interval definitions, and ADSR envelope parameters.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

# Default sample rate for audio generation (CD quality)
DEFAULT_SAMPLE_RATE = 44100

# Note frequencies in Hz (4th octave, A4 = 440Hz standard tuning)
NOTE_FREQUENCIES: Dict[str, float] = {
    'C': 261.63,
    'C#': 277.18,
    'Db': 277.18,
    'D': 293.66,
    'D#': 311.13,
    'Eb': 311.13,
    'E': 329.63,
    'F': 349.23,
    'F#': 369.99,
    'Gb': 369.99,
    'G': 392.00,
    'G#': 415.30,
    'Ab': 415.30,
    'A': 440.00,
    'A#': 466.16,
    'Bb': 466.16,
    'B': 493.88,
}

# Chromatic scale for interval calculations
CHROMATIC_SCALE: List[str] = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Interval definitions: name -> semitones
INTERVALS_SEMITONES: Dict[str, int] = {
    'unison': 0,
    'minor_second': 1,
    'major_second': 2,
    'minor_third': 3,
    'major_third': 4,
    'perfect_fourth': 5,
    'tritone': 6,
    'augmented_fourth': 6,
    'diminished_fifth': 6,
    'perfect_fifth': 7,
    'minor_sixth': 8,
    'major_sixth': 9,
    'minor_seventh': 10,
    'major_seventh': 11,
    'perfect_octave': 12,
}


@dataclass
class ADSREnvelope:
    """
    ADSR (Attack, Decay, Sustain, Release) envelope parameters.
    
    Attributes:
        attack: Time in seconds for the sound to reach peak amplitude
        decay: Time in seconds to decay from peak to sustain level
        sustain: Sustain level (0.0 to 1.0) - amplitude during sustain phase
        release: Time in seconds to fade from sustain level to silence
    """
    attack: float = 0.05
    decay: float = 0.2
    sustain: float = 0.7
    release: float = 0.5
    
    def validate(self) -> bool:
        """Validate ADSR parameters are within acceptable ranges."""
        return (
            0.0 <= self.attack <= 2.0 and
            0.0 <= self.decay <= 2.0 and
            0.0 <= self.sustain <= 1.0 and
            0.0 <= self.release <= 5.0
        )


# Predefined ADSR envelopes for different instruments
class InstrumentEnvelopes:
    """Predefined ADSR envelopes for various instruments."""
    
    PIANO = ADSREnvelope(attack=0.01, decay=0.3, sustain=0.4, release=0.8)
    GUITAR = ADSREnvelope(attack=0.005, decay=0.1, sustain=0.3, release=1.0)
    SYNTH_PAD = ADSREnvelope(attack=0.3, decay=0.2, sustain=0.8, release=1.0)
    SYNTH_LEAD = ADSREnvelope(attack=0.01, decay=0.1, sustain=0.7, release=0.3)
    ORGAN = ADSREnvelope(attack=0.02, decay=0.0, sustain=1.0, release=0.1)


# Harmonic content definitions: (harmonic_number, relative_amplitude)
# These define the timbre of each instrument
class InstrumentHarmonics:
    """Harmonic content definitions for various instruments."""
    
    # Piano: strong fundamental, decreasing upper harmonics
    PIANO: List[Tuple[int, float]] = [
        (1, 1.0),    # Fundamental
        (2, 0.5),    # 2nd harmonic
        (3, 0.25),   # 3rd harmonic
        (4, 0.125),  # 4th harmonic
        (5, 0.0625), # 5th harmonic
    ]
    
    # Guitar: more complex harmonic content
    GUITAR: List[Tuple[int, float]] = [
        (1, 1.0),
        (2, 0.4),
        (3, 0.3),
        (4, 0.2),
        (5, 0.15),
        (6, 0.1),
    ]
    
    # Synth: can vary, this is a saw-like wave
    SYNTH_SAW: List[Tuple[int, float]] = [
        (1, 1.0),
        (2, 0.5),
        (3, 0.333),
        (4, 0.25),
        (5, 0.2),
        (6, 0.167),
        (7, 0.143),
        (8, 0.125),
    ]
    
    # Synth: square-like wave (odd harmonics only)
    SYNTH_SQUARE: List[Tuple[int, float]] = [
        (1, 1.0),
        (3, 0.333),
        (5, 0.2),
        (7, 0.143),
        (9, 0.111),
    ]


def get_note_frequency(note: str, octave: int = 4) -> float:
    """
    Get the frequency of a note at a specific octave.
    
    Args:
        note: Note name (e.g., 'C', 'C#', 'Db')
        octave: Octave number (default 4, where A4 = 440Hz)
    
    Returns:
        Frequency in Hz
    
    Raises:
        ValueError: If note is not recognized
    """
    base_freq = NOTE_FREQUENCIES.get(note)
    if base_freq is None:
        raise ValueError(f"Unknown note: {note}")
    
    # Calculate octave multiplier (octave 4 is the base)
    octave_multiplier = 2 ** (octave - 4)
    return base_freq * octave_multiplier


def get_frequency_with_semitone_offset(base_note: str, semitones: int, octave: int = 4) -> float:
    """
    Get frequency of a note offset by semitones from a base note.
    
    Args:
        base_note: Starting note name
        semitones: Number of semitones to offset (can be negative)
        octave: Base octave
    
    Returns:
        Frequency in Hz
    """
    base_freq = get_note_frequency(base_note, octave)
    # Each semitone is multiplication by 2^(1/12)
    return base_freq * (2 ** (semitones / 12))


def get_target_note(base_note: str, semitones: int) -> str:
    """
    Get the note name that is a certain number of semitones from the base note.
    
    Args:
        base_note: Starting note name (without octave)
        semitones: Number of semitones to offset
    
    Returns:
        Target note name
    """
    # Handle enharmonic equivalents
    note_map = {
        'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'
    }
    normalized_note = note_map.get(base_note, base_note)
    
    try:
        base_index = CHROMATIC_SCALE.index(normalized_note)
    except ValueError:
        raise ValueError(f"Unknown note: {base_note}")
    
    target_index = (base_index + semitones) % 12
    return CHROMATIC_SCALE[target_index]
