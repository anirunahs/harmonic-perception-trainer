# Audio generation module
# Implements Factory Method pattern for instrument-based sound generation

from .constants import (
    NOTE_FREQUENCIES, 
    INTERVALS_SEMITONES, 
    DEFAULT_SAMPLE_RATE,
    ADSREnvelope,
    InstrumentEnvelopes,
    get_note_frequency,
    get_frequency_with_semitone_offset,
    get_target_note,
)
from .generators import AudioGenerator, PianoGenerator, GuitarGenerator, SynthGenerator
from .factory import InstrumentFactory, get_factory
from .utils import (
    save_audio,
    combine_tones_harmonic,
    combine_tones_melodic,
    frequency_to_note,
    note_to_frequency,
)

__all__ = [
    # Constants
    'NOTE_FREQUENCIES',
    'INTERVALS_SEMITONES', 
    'DEFAULT_SAMPLE_RATE',
    'ADSREnvelope',
    'InstrumentEnvelopes',
    # Helper functions
    'get_note_frequency',
    'get_frequency_with_semitone_offset',
    'get_target_note',
    # Generators
    'AudioGenerator',
    'PianoGenerator',
    'GuitarGenerator',
    'SynthGenerator',
    # Factory
    'InstrumentFactory',
    'get_factory',
    # Utilities
    'save_audio',
    'combine_tones_harmonic',
    'combine_tones_melodic',
    'frequency_to_note',
    'note_to_frequency',
]
