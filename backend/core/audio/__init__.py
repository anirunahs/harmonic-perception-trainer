# Audio generation module
# Implements Factory Method and Abstract Factory patterns for instrument-based sound generation

from .constants import (
    NOTE_FREQUENCIES, 
    INTERVALS_SEMITONES,
    CHORDS_SEMITONES,
    DEFAULT_SAMPLE_RATE,
    ADSREnvelope,
    InstrumentEnvelopes,
    get_note_frequency,
    get_frequency_with_semitone_offset,
    get_target_note,
)
from .generators import AudioGenerator, SynthGenerator, PianoGenerator
from .factory import InstrumentFactory, get_factory
from .music_elements import (
    IntervalGenerator,
    ChordGenerator,
    SynthIntervalGenerator,
    SynthChordGenerator,
)
from .abstract_factory import (
    MusicElementFactory,
    SynthMusicElementFactory,
    MusicElementFactoryRegistry,
)
from .audio_manager import AudioManager, get_audio_manager
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
    'CHORDS_SEMITONES',
    'DEFAULT_SAMPLE_RATE',
    'ADSREnvelope',
    'InstrumentEnvelopes',
    # Helper functions
    'get_note_frequency',
    'get_frequency_with_semitone_offset',
    'get_target_note',
    # Generators
    'AudioGenerator',
    'SynthGenerator',
    'PianoGenerator',
    # Factory Method
    'InstrumentFactory',
    'get_factory',
    # Abstract Factory - Music Elements
    'IntervalGenerator',
    'ChordGenerator',
    'SynthIntervalGenerator',
    'SynthChordGenerator',
    'MusicElementFactory',
    'SynthMusicElementFactory',
    'MusicElementFactoryRegistry',
    # Singleton - Audio Manager
    'AudioManager',
    'get_audio_manager',
    # Utilities
    'save_audio',
    'combine_tones_harmonic',
    'combine_tones_melodic',
    'frequency_to_note',
    'note_to_frequency',
]
