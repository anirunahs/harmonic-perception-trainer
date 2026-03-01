"""
Configuration for synthetic interval dataset generation.

Defines instruments, pitch ranges, intervals, split strategies, and audio parameters.

"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path


SAMPLE_RATE = 22050
SEGMENT_DURATION = 2.0     # seconds per interval sample
FADE_IN_MS = 5             # click-free fade-in
FADE_OUT_MS = 50           # natural fade-out
PEAK_NORMALIZE_DB = -1.0   # target peak after normalisation


NOTE_NAMES_SHARP = [
    'C', 'C#', 'D', 'D#', 'E', 'F',
    'F#', 'G', 'G#', 'A', 'A#', 'B',
]

NOTE_NAMES_FLAT = [
    'C', 'Db', 'D', 'Eb', 'E', 'F',
    'Gb', 'G', 'Ab', 'A', 'Bb', 'B',
]

NOTE_TO_SEMITONE = {
    'C': 0,  'B#': 0,
    'C#': 1, 'Db': 1,
    'D': 2,
    'D#': 3, 'Eb': 3,
    'E': 4,  'Fb': 4,
    'E#': 5, 'F': 5,
    'F#': 6, 'Gb': 6,
    'G': 7,
    'G#': 8, 'Ab': 8,
    'A': 9,
    'A#': 10, 'Bb': 10,
    'B': 11,  'Cb': 11,
}

INTERVALS = {
    'minor_2nd':    1,
    'major_2nd':    2,
    'minor_3rd':    3,
    'major_3rd':    4,
    'perfect_4th':  5,
    'tritone':      6,
    'perfect_5th':  7,
    'minor_6th':    8,
    'major_6th':    9,
    'minor_7th':    10,
    'major_7th':    11,
    'perfect_8th':  12,
}


def midi_to_note(midi: int) -> str:
    """MIDI note number → note name with octave.  60 → 'C4', 69 → 'A4'."""
    octave = (midi // 12) - 1
    return f"{NOTE_NAMES_SHARP[midi % 12]}{octave}"


def note_to_midi(note_str: str) -> int:
    """Note name with octave → MIDI number.  'C4' → 60, 'Db3' → 49."""
    if len(note_str) >= 3 and note_str[1] in '#b':
        note_name, octave = note_str[:2], int(note_str[2:])
    else:
        note_name, octave = note_str[0], int(note_str[1:])
    return (octave + 1) * 12 + NOTE_TO_SEMITONE[note_name]


def midi_to_hz(midi: int) -> float:
    """MIDI note number → frequency in Hz.  A4 (69) = 440 Hz."""
    return 440.0 * (2 ** ((midi - 69) / 12))


def midi_to_safe_filename(midi: int) -> str:
    """MIDI → filesystem-safe note name ('C#2' → 'Cs2')."""
    return midi_to_note(midi).replace('#', 's')


def _build_salamander_anchors() -> Dict[int, str]:
    anchors = {}
    for midi in range(21, 109, 3):
        note = midi_to_note(midi)
        anchors[midi] = note.replace('#', 's')
    return anchors

SALAMANDER_ANCHORS: Dict[int, str] = _build_salamander_anchors()
SALAMANDER_BASE_URL = 'https://tonejs.github.io/audio/salamander/'


FLUIDR3_BASE_URL = 'https://gleitz.github.io/midi-js-soundfonts/FluidR3_GM'


def midi_to_fluidr3_filename(midi: int) -> str:
    """MIDI → FluidR3_GM mp3 filename.  49 → 'Db3.mp3'."""
    octave = (midi // 12) - 1
    note = NOTE_NAMES_FLAT[midi % 12]
    return f"{note}{octave}.mp3"


# Instrument Definitions

@dataclass(frozen=True)
class InstrumentConfig:
    """Configuration for a single instrument sample source."""
    name: str
    instrument_id: str
    source_type: str
    midi_range: Tuple[int, int]
    fluidr3_instrument: str = ''

    @property
    def base_url(self) -> str:
        if self.source_type == 'salamander':
            return SALAMANDER_BASE_URL
        return f'{FLUIDR3_BASE_URL}/{self.fluidr3_instrument}-mp3/'

    @property
    def range_semitones(self) -> int:
        return self.midi_range[1] - self.midi_range[0]

    @property
    def range_display(self) -> str:
        return f"{midi_to_note(self.midi_range[0])}–{midi_to_note(self.midi_range[1])}"


INSTRUMENTS: Dict[str, InstrumentConfig] = {
    'piano': InstrumentConfig(
        name='Piano (Salamander Grand)',
        instrument_id='piano',
        source_type='salamander',
        midi_range=(36, 84),        # C2 – C6
    ),
    'guitar_nylon': InstrumentConfig(
        name='Acoustic Guitar (Nylon)',
        instrument_id='guitar_nylon',
        source_type='fluidr3',
        midi_range=(40, 76),        # E2 – E5
        fluidr3_instrument='acoustic_guitar_nylon',
    ),
    'guitar_steel': InstrumentConfig(
        name='Acoustic Guitar (Steel)',
        instrument_id='guitar_steel',
        source_type='fluidr3',
        midi_range=(40, 76),        # E2 – E5
        fluidr3_instrument='acoustic_guitar_steel',
    ),
    'violin': InstrumentConfig(
        name='Violin',
        instrument_id='violin',
        source_type='fluidr3',
        midi_range=(55, 88),        # G3 – E6
        fluidr3_instrument='violin',
    ),
    'cello': InstrumentConfig(
        name='Cello',
        instrument_id='cello',
        source_type='fluidr3',
        midi_range=(36, 72),        # C2 – C5
        fluidr3_instrument='cello',
    ),
    'flute': InstrumentConfig(
        name='Flute',
        instrument_id='flute',
        source_type='fluidr3',
        midi_range=(60, 96),        # C4 – C7
        fluidr3_instrument='flute',
    ),
    'clarinet': InstrumentConfig(
        name='Clarinet',
        instrument_id='clarinet',
        source_type='fluidr3',
        midi_range=(50, 84),        # D3 – C6
        fluidr3_instrument='clarinet',
    ),
    'trumpet': InstrumentConfig(
        name='Trumpet',
        instrument_id='trumpet',
        source_type='fluidr3',
        midi_range=(55, 84),        # G3 – C6
        fluidr3_instrument='trumpet',
    ),
    'church_organ': InstrumentConfig(
        name='Church Organ',
        instrument_id='church_organ',
        source_type='fluidr3',
        midi_range=(36, 96),        # C2 – C7
        fluidr3_instrument='church_organ',
    ),
    'marimba': InstrumentConfig(
        name='Marimba',
        instrument_id='marimba',
        source_type='fluidr3',
        midi_range=(48, 84),        # C3 – C6
        fluidr3_instrument='marimba',
    ),
}


SPLIT_BY_INSTRUMENT = {
    'train': ['piano', 'guitar_nylon', 'guitar_steel', 'cello', 'church_organ'],
    'val':   ['clarinet', 'marimba'],
    'test':  ['violin', 'flute', 'trumpet'],
}

SPLIT_BY_REGISTER = {
    'train': (36, 67),   # C2 – G4
    'val':   (68, 73),   # G#4 – C#5
    'test':  (74, 96),   # D5 – C7
}

DEFAULT_SPLIT_STRATEGY = 'instrument'


def get_split(instrument_id: str, base_midi: int,
              strategy: str = DEFAULT_SPLIT_STRATEGY) -> str:
    """Determine train/val/test split for a given sample."""
    if strategy == 'instrument':
        for split_name, instruments in SPLIT_BY_INSTRUMENT.items():
            if instrument_id in instruments:
                return split_name
        return 'train'

    if strategy == 'register':
        for split_name, (lo, hi) in SPLIT_BY_REGISTER.items():
            if lo <= base_midi <= hi:
                return split_name
        return 'train'

    raise ValueError(f"Unknown split strategy: {strategy}")


DEFAULT_OUTPUT_DIR = Path('dataset-preparation/synthetic-dataset')
DEFAULT_CACHE_DIR = Path('dataset-preparation/.sample-cache')
