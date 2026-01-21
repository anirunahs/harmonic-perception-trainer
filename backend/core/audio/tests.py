"""
Unit tests for audio generation module.

Tests the Factory Method pattern implementation for audio generators.
"""

import unittest
import numpy as np

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
from .generators import (
    AudioGenerator,
    PianoGenerator,
    GuitarGenerator,
    SynthGenerator,
)
from .factory import InstrumentFactory, get_factory
from .utils import (
    combine_tones_harmonic,
    combine_tones_melodic,
    frequency_to_note,
    note_to_frequency,
)


class TestConstants(unittest.TestCase):
    """Test audio constants and helper functions."""
    
    def test_note_frequencies_exist(self):
        """Test that all standard notes have frequencies defined."""
        standard_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for note in standard_notes:
            self.assertIn(note, NOTE_FREQUENCIES)
            self.assertGreater(NOTE_FREQUENCIES[note], 0)
    
    def test_a4_frequency(self):
        """Test that A4 = 440 Hz (standard tuning)."""
        self.assertEqual(NOTE_FREQUENCIES['A'], 440.0)
    
    def test_intervals_semitones(self):
        """Test interval definitions."""
        self.assertEqual(INTERVALS_SEMITONES['perfect_octave'], 12)
        self.assertEqual(INTERVALS_SEMITONES['perfect_fifth'], 7)
        self.assertEqual(INTERVALS_SEMITONES['major_third'], 4)
    
    def test_get_note_frequency_octave_scaling(self):
        """Test that octave changes double/halve the frequency."""
        freq_a4 = get_note_frequency('A', 4)
        freq_a5 = get_note_frequency('A', 5)
        freq_a3 = get_note_frequency('A', 3)
        
        self.assertAlmostEqual(freq_a5, freq_a4 * 2, places=2)
        self.assertAlmostEqual(freq_a3, freq_a4 / 2, places=2)
    
    def test_get_frequency_with_semitone_offset(self):
        """Test semitone offset calculation."""
        base_freq = get_note_frequency('C', 4)
        octave_freq = get_frequency_with_semitone_offset('C', 12, 4)
        
        self.assertAlmostEqual(octave_freq, base_freq * 2, places=2)
    
    def test_get_target_note(self):
        """Test target note calculation."""
        self.assertEqual(get_target_note('C', 4), 'E')  # Major third
        self.assertEqual(get_target_note('C', 7), 'G')  # Perfect fifth
        self.assertEqual(get_target_note('G', 5), 'C')  # Perfect fourth (wraps)


class TestADSREnvelope(unittest.TestCase):
    """Test ADSR envelope dataclass."""
    
    def test_default_values(self):
        """Test default ADSR values."""
        adsr = ADSREnvelope()
        self.assertIsNotNone(adsr.attack)
        self.assertIsNotNone(adsr.decay)
        self.assertIsNotNone(adsr.sustain)
        self.assertIsNotNone(adsr.release)
    
    def test_validation_valid(self):
        """Test that valid ADSR passes validation."""
        adsr = ADSREnvelope(attack=0.1, decay=0.2, sustain=0.7, release=0.5)
        self.assertTrue(adsr.validate())
    
    def test_validation_invalid_sustain(self):
        """Test that invalid sustain fails validation."""
        adsr = ADSREnvelope(attack=0.1, decay=0.2, sustain=1.5, release=0.5)
        self.assertFalse(adsr.validate())
    
    def test_instrument_envelopes(self):
        """Test predefined instrument envelopes exist."""
        self.assertIsNotNone(InstrumentEnvelopes.PIANO)
        self.assertIsNotNone(InstrumentEnvelopes.GUITAR)
        self.assertIsNotNone(InstrumentEnvelopes.SYNTH_PAD)


class TestPianoGenerator(unittest.TestCase):
    """Test PianoGenerator class."""
    
    def setUp(self):
        self.generator = PianoGenerator()
    
    def test_instrument_name(self):
        """Test that instrument name is correct."""
        self.assertEqual(self.generator.get_instrument_name(), 'piano')
    
    def test_generate_tone_returns_array(self):
        """Test that generate_tone returns numpy array."""
        tone = self.generator.generate_tone(440.0, duration=0.5)
        self.assertIsInstance(tone, np.ndarray)
    
    def test_generate_tone_length(self):
        """Test that generated tone has correct length."""
        duration = 1.0
        tone = self.generator.generate_tone(440.0, duration=duration)
        expected_samples = int(DEFAULT_SAMPLE_RATE * duration)
        self.assertEqual(len(tone), expected_samples)
    
    def test_generate_tone_normalized(self):
        """Test that tone is normalized to reasonable amplitude."""
        tone = self.generator.generate_tone(440.0, duration=0.5)
        self.assertLessEqual(np.max(np.abs(tone)), 1.0)
    
    def test_generate_tone_not_silent(self):
        """Test that generated tone is not silent."""
        tone = self.generator.generate_tone(440.0, duration=0.5)
        self.assertGreater(np.max(np.abs(tone)), 0.01)


class TestGuitarGenerator(unittest.TestCase):
    """Test GuitarGenerator class."""
    
    def setUp(self):
        self.generator = GuitarGenerator()
    
    def test_instrument_name(self):
        """Test that instrument name is correct."""
        self.assertEqual(self.generator.get_instrument_name(), 'guitar')
    
    def test_pluck_position_parameter(self):
        """Test that pluck position can be configured."""
        generator = GuitarGenerator(pluck_position=0.3)
        self.assertEqual(generator.pluck_position, 0.3)
    
    def test_generate_tone_not_silent(self):
        """Test that generated tone is not silent."""
        tone = self.generator.generate_tone(330.0, duration=0.5)
        self.assertGreater(np.max(np.abs(tone)), 0.01)


class TestSynthGenerator(unittest.TestCase):
    """Test SynthGenerator class."""
    
    def test_wave_types(self):
        """Test that all wave types can be created."""
        for wave_type in SynthGenerator.WAVE_TYPES:
            generator = SynthGenerator(wave_type=wave_type)
            self.assertIn(wave_type, generator.get_instrument_name())
    
    def test_invalid_wave_type(self):
        """Test that invalid wave type raises error."""
        with self.assertRaises(ValueError):
            SynthGenerator(wave_type='invalid')
    
    def test_sine_wave_purity(self):
        """Test that sine wave has minimal harmonics."""
        generator = SynthGenerator(wave_type='sine')
        tone = generator.generate_tone(440.0, duration=0.5)
        self.assertIsInstance(tone, np.ndarray)


class TestInstrumentFactory(unittest.TestCase):
    """Test InstrumentFactory class."""
    
    def setUp(self):
        self.factory = InstrumentFactory()
    
    def test_create_piano(self):
        """Test creating piano generator via factory."""
        generator = self.factory.create_generator('piano')
        self.assertIsInstance(generator, PianoGenerator)
    
    def test_create_guitar(self):
        """Test creating guitar generator via factory."""
        generator = self.factory.create_generator('guitar')
        self.assertIsInstance(generator, GuitarGenerator)
    
    def test_create_synth(self):
        """Test creating synth generator via factory."""
        generator = self.factory.create_generator('synth')
        self.assertIsInstance(generator, SynthGenerator)
    
    def test_create_synth_with_wave_type(self):
        """Test creating synth with specific wave type."""
        generator = self.factory.create_generator('synth', wave_type='square')
        self.assertEqual(generator.wave_type, 'square')
    
    def test_unknown_instrument_raises(self):
        """Test that unknown instrument raises ValueError."""
        with self.assertRaises(ValueError):
            self.factory.create_generator('unknown_instrument')
    
    def test_case_insensitive(self):
        """Test that instrument names are case-insensitive."""
        generator = self.factory.create_generator('PIANO')
        self.assertIsInstance(generator, PianoGenerator)
    
    def test_get_available_instruments(self):
        """Test listing available instruments."""
        instruments = self.factory.get_available_instruments()
        self.assertIn('piano', instruments)
        self.assertIn('guitar', instruments)
        self.assertIn('synth', instruments)
    
    def test_convenience_methods(self):
        """Test convenience factory methods."""
        piano = self.factory.create_piano()
        guitar = self.factory.create_guitar()
        synth = self.factory.create_synth()
        
        self.assertIsInstance(piano, PianoGenerator)
        self.assertIsInstance(guitar, GuitarGenerator)
        self.assertIsInstance(synth, SynthGenerator)
    
    def test_custom_adsr(self):
        """Test creating generator with custom ADSR."""
        custom_adsr = ADSREnvelope(attack=0.5, decay=0.5, sustain=0.5, release=0.5)
        generator = self.factory.create_generator('piano', adsr=custom_adsr)
        self.assertEqual(generator.adsr.attack, 0.5)


class TestGetFactory(unittest.TestCase):
    """Test get_factory singleton-like function."""
    
    def test_returns_factory(self):
        """Test that get_factory returns InstrumentFactory."""
        factory = get_factory()
        self.assertIsInstance(factory, InstrumentFactory)
    
    def test_same_instance_same_sample_rate(self):
        """Test that same sample rate returns same instance."""
        factory1 = get_factory(44100)
        factory2 = get_factory(44100)
        self.assertIs(factory1, factory2)


class TestAudioUtils(unittest.TestCase):
    """Test audio utility functions."""
    
    def test_combine_tones_harmonic(self):
        """Test harmonic tone combination."""
        tone1 = np.array([0.5, 0.5, 0.5])
        tone2 = np.array([0.3, 0.3, 0.3])
        combined = combine_tones_harmonic(tone1, tone2)
        
        self.assertEqual(len(combined), 3)
        np.testing.assert_array_almost_equal(combined, [0.4, 0.4, 0.4])
    
    def test_combine_tones_harmonic_different_lengths(self):
        """Test harmonic combination with different length tones."""
        tone1 = np.array([0.5, 0.5, 0.5])
        tone2 = np.array([0.3, 0.3])
        combined = combine_tones_harmonic(tone1, tone2)
        
        self.assertEqual(len(combined), 3)
    
    def test_combine_tones_melodic(self):
        """Test melodic tone combination."""
        tone1 = np.array([1.0, 1.0])
        tone2 = np.array([2.0, 2.0])
        combined = combine_tones_melodic(tone1, tone2)
        
        self.assertEqual(len(combined), 4)
        np.testing.assert_array_equal(combined, [1.0, 1.0, 2.0, 2.0])
    
    def test_combine_tones_melodic_with_gap(self):
        """Test melodic combination with gap."""
        tone1 = np.array([1.0, 1.0])
        tone2 = np.array([2.0, 2.0])
        combined = combine_tones_melodic(tone1, tone2, gap_samples=2)
        
        self.assertEqual(len(combined), 6)
        np.testing.assert_array_equal(combined, [1.0, 1.0, 0.0, 0.0, 2.0, 2.0])
    
    def test_frequency_to_note_a4(self):
        """Test frequency to note conversion for A4."""
        note = frequency_to_note(440.0)
        self.assertEqual(note, 'A4')
    
    def test_frequency_to_note_c4(self):
        """Test frequency to note conversion for C4."""
        # C4 is approximately 261.63 Hz
        note = frequency_to_note(261.63)
        self.assertEqual(note, 'C4')
    
    def test_note_to_frequency_a4(self):
        """Test note to frequency conversion for A4."""
        freq = note_to_frequency('A4')
        self.assertAlmostEqual(freq, 440.0, places=1)
    
    def test_note_to_frequency_round_trip(self):
        """Test that note->freq->note round trip works."""
        original_freq = 440.0
        note = frequency_to_note(original_freq)
        back_freq = note_to_frequency(note)
        self.assertAlmostEqual(back_freq, original_freq, places=1)


class TestFactoryRegistration(unittest.TestCase):
    """Test dynamic registration of new instruments."""
    
    def test_register_new_instrument(self):
        """Test registering a new instrument type."""
        # Create a custom generator class
        class CustomGenerator(AudioGenerator):
            def _get_default_adsr(self):
                return InstrumentEnvelopes.PIANO
            
            def _get_harmonics(self):
                return [(1, 1.0)]
            
            def get_instrument_name(self):
                return 'custom'
        
        # Register it
        InstrumentFactory.register('custom', CustomGenerator)
        
        # Create via factory
        factory = InstrumentFactory()
        generator = factory.create_generator('custom')
        self.assertIsInstance(generator, CustomGenerator)
        
        # Clean up
        InstrumentFactory.unregister('custom')
    
    def test_unregister_instrument(self):
        """Test unregistering an instrument."""
        # First register
        class TempGenerator(AudioGenerator):
            def _get_default_adsr(self):
                return InstrumentEnvelopes.PIANO
            
            def _get_harmonics(self):
                return [(1, 1.0)]
            
            def get_instrument_name(self):
                return 'temp'
        
        InstrumentFactory.register('temp', TempGenerator)
        
        # Unregister
        result = InstrumentFactory.unregister('temp')
        self.assertTrue(result)
        
        # Should no longer be available
        factory = InstrumentFactory()
        with self.assertRaises(ValueError):
            factory.create_generator('temp')


if __name__ == '__main__':
    unittest.main()
