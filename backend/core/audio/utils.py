"""
Audio utility functions for file operations and conversions.
"""

import os
import numpy as np
from scipy.io import wavfile
from typing import Optional

from .constants import DEFAULT_SAMPLE_RATE


def save_audio(audio_data: np.ndarray, 
               filepath: str, 
               sample_rate: int = DEFAULT_SAMPLE_RATE) -> str:
    """
    Save audio data to a WAV file.
    
    Args:
        audio_data: Numpy array of audio samples (normalized -1.0 to 1.0)
        filepath: Full path where to save the file
        sample_rate: Sample rate in Hz
    
    Returns:
        The filepath where the file was saved
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert to 16-bit integer
    audio_int16 = np.int16(audio_data * 32767)
    
    # Write WAV file
    wavfile.write(filepath, sample_rate, audio_int16)
    
    return filepath


def combine_tones_harmonic(tone1: np.ndarray, tone2: np.ndarray) -> np.ndarray:
    """
    Combine two tones harmonically (played simultaneously).
    
    Args:
        tone1: First audio tone
        tone2: Second audio tone
    
    Returns:
        Combined audio with normalized amplitude
    """
    # Ensure same length (pad shorter one with zeros)
    max_len = max(len(tone1), len(tone2))
    if len(tone1) < max_len:
        tone1 = np.pad(tone1, (0, max_len - len(tone1)))
    if len(tone2) < max_len:
        tone2 = np.pad(tone2, (0, max_len - len(tone2)))
    
    # Combine and normalize
    combined = (tone1 + tone2) / 2
    return combined


def combine_tones_melodic(tone1: np.ndarray, tone2: np.ndarray, 
                          gap_samples: int = 0) -> np.ndarray:
    """
    Combine two tones melodically (played sequentially).
    
    Args:
        tone1: First audio tone
        tone2: Second audio tone
        gap_samples: Number of silent samples between tones
    
    Returns:
        Concatenated audio
    """
    if gap_samples > 0:
        gap = np.zeros(gap_samples)
        return np.concatenate([tone1, gap, tone2])
    return np.concatenate([tone1, tone2])


def frequency_to_note(frequency: float) -> str:
    """
    Convert frequency to note name with octave.
    
    Args:
        frequency: Frequency in Hz
    
    Returns:
        Note name with octave (e.g., 'A4')
    """
    A4 = 440.0
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Calculate semitones from A4
    semitones_from_a4 = round(12 * np.log2(frequency / A4))
    
    # A4 is 57 semitones from C0 (4 octaves * 12 + 9 semitones for A)
    # So total semitones from C0 = 57 + semitones_from_a4
    total_semitones_from_c0 = 57 + semitones_from_a4
    
    octave = total_semitones_from_c0 // 12
    note_index = total_semitones_from_c0 % 12
    
    return f"{notes[note_index]}{octave}"


def note_to_frequency(note_name: str) -> float:
    """
    Convert note name with octave to frequency.
    
    Args:
        note_name: Note name with octave (e.g., 'A4', 'C#5')
    
    Returns:
        Frequency in Hz
    """
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Parse note and octave
    if len(note_name) >= 2 and note_name[-1].isdigit():
        if len(note_name) >= 3 and note_name[-2].isdigit():
            # Handle octave 10+
            octave = int(note_name[-2:])
            note = note_name[:-2]
        else:
            octave = int(note_name[-1])
            note = note_name[:-1]
    else:
        raise ValueError(f"Invalid note format: {note_name}")
    
    # Handle enharmonic equivalents
    note_map = {
        'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'
    }
    note = note_map.get(note, note)
    
    if note not in notes:
        raise ValueError(f"Unknown note: {note}")
    
    note_index = notes.index(note)
    
    # Calculate frequency (A4 = 440Hz)
    semitones_from_a4 = (octave - 4) * 12 + (note_index - 9)
    frequency = 440.0 * (2 ** (semitones_from_a4 / 12))
    
    return frequency
