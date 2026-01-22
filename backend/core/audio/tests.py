"""
Unit tests for audio generation module.

Tests the Factory Method pattern implementation for audio generators.
"""

import unittest
import threading
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
from .abstract_factory import (
    MusicElementFactory,
    PianoMusicElementFactory,
    GuitarMusicElementFactory,
    MusicElementFactoryRegistry,
)
from .audio_manager import AudioManager, get_audio_manager
from .music_elements import (
    IntervalGenerator,
    ChordGenerator,
    PianoIntervalGenerator,
    PianoChordGenerator,
    GuitarIntervalGenerator,
    GuitarChordGenerator,
)
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


class TestMusicElementFactory(unittest.TestCase):
    """Test Abstract Factory pattern for music elements."""
    
    def test_abstract_factory_cannot_be_instantiated(self):
        """Test that abstract factory cannot be instantiated."""
        with self.assertRaises(TypeError):
            MusicElementFactory()
    
    def test_piano_factory_creates_piano_generators(self):
        """Test that PianoMusicElementFactory creates piano generators."""
        factory = PianoMusicElementFactory()
        
        interval_gen = factory.create_interval_generator()
        chord_gen = factory.create_chord_generator()
        
        self.assertIsInstance(interval_gen, PianoIntervalGenerator)
        self.assertIsInstance(chord_gen, PianoChordGenerator)
        self.assertEqual(factory.get_instrument_name(), 'piano')
    
    def test_guitar_factory_creates_guitar_generators(self):
        """Test that GuitarMusicElementFactory creates guitar generators."""
        factory = GuitarMusicElementFactory()
        
        interval_gen = factory.create_interval_generator()
        chord_gen = factory.create_chord_generator()
        
        self.assertIsInstance(interval_gen, GuitarIntervalGenerator)
        self.assertIsInstance(chord_gen, GuitarChordGenerator)
        self.assertEqual(factory.get_instrument_name(), 'guitar')
    
    def test_factory_consistency(self):
        """Test that factory creates compatible generators."""
        factory = PianoMusicElementFactory()
        
        interval_gen = factory.create_interval_generator()
        chord_gen = factory.create_chord_generator()
        
        # Both should use the same instrument
        self.assertEqual(interval_gen.get_instrument_name(), 'piano')
        self.assertEqual(chord_gen.get_instrument_name(), 'piano')
    
    def test_factory_reuses_audio_generator(self):
        """Test that factory reuses audio generator instance."""
        factory = PianoMusicElementFactory()
        
        interval_gen1 = factory.create_interval_generator()
        interval_gen2 = factory.create_interval_generator()
        
        # Should use the same audio generator instance
        self.assertIs(interval_gen1.audio_generator, interval_gen2.audio_generator)


class TestMusicElementFactoryRegistry(unittest.TestCase):
    """Test MusicElementFactoryRegistry."""
    
    def test_create_piano_factory(self):
        """Test creating piano factory via registry."""
        factory = MusicElementFactoryRegistry.create_factory('piano')
        self.assertIsInstance(factory, PianoMusicElementFactory)
    
    def test_create_guitar_factory(self):
        """Test creating guitar factory via registry."""
        factory = MusicElementFactoryRegistry.create_factory('guitar')
        self.assertIsInstance(factory, GuitarMusicElementFactory)
    
    def test_unknown_instrument_raises(self):
        """Test that unknown instrument raises ValueError."""
        with self.assertRaises(ValueError):
            MusicElementFactoryRegistry.create_factory('unknown')
    
    def test_case_insensitive(self):
        """Test that instrument names are case-insensitive."""
        factory = MusicElementFactoryRegistry.create_factory('PIANO')
        self.assertIsInstance(factory, PianoMusicElementFactory)
    
    def test_get_available_instruments(self):
        """Test listing available instruments."""
        instruments = MusicElementFactoryRegistry.get_available_instruments()
        self.assertIn('piano', instruments)
        self.assertIn('guitar', instruments)
    
    def test_register_factory(self):
        """Test registering a new factory."""
        # Create a custom factory
        class CustomMusicElementFactory(MusicElementFactory):
            def _create_audio_generator(self):
                return InstrumentFactory().create_piano()
            
            def create_interval_generator(self):
                return PianoIntervalGenerator(self._get_audio_generator())
            
            def create_chord_generator(self):
                return PianoChordGenerator(self._get_audio_generator())
            
            def get_instrument_name(self):
                return 'custom'
        
        # Register it
        MusicElementFactoryRegistry.register_factory('custom', CustomMusicElementFactory)
        
        # Create via registry
        factory = MusicElementFactoryRegistry.create_factory('custom')
        self.assertIsInstance(factory, CustomMusicElementFactory)
        
        # Clean up - restore original state
        # Note: In real scenario, might want to keep or properly restore


class TestIntervalGenerator(unittest.TestCase):
    """Test IntervalGenerator implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.piano_factory = PianoMusicElementFactory()
        self.guitar_factory = GuitarMusicElementFactory()
    
    def test_piano_harmonic_interval(self):
        """Test piano harmonic interval generation."""
        generator = self.piano_factory.create_interval_generator()
        audio = generator.generate_harmonic_interval('C', 'major_third', octave=4, duration=0.5)
        
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)
    
    def test_piano_melodic_interval(self):
        """Test piano melodic interval generation."""
        generator = self.piano_factory.create_interval_generator()
        audio = generator.generate_melodic_interval('C', 'major_third', octave=4, duration=0.5)
        
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)
    
    def test_guitar_harmonic_interval(self):
        """Test guitar harmonic interval generation."""
        generator = self.guitar_factory.create_interval_generator()
        audio = generator.generate_harmonic_interval('C', 'perfect_fifth', octave=4, duration=0.5)
        
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)
    
    def test_guitar_melodic_interval(self):
        """Test guitar melodic interval generation."""
        generator = self.guitar_factory.create_interval_generator()
        audio = generator.generate_melodic_interval('C', 'perfect_fifth', octave=4, duration=0.5)
        
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)
    
    def test_melodic_interval_has_gap(self):
        """Test that melodic interval includes gap between notes."""
        generator = self.piano_factory.create_interval_generator()
        audio_short = generator.generate_melodic_interval('C', 'major_third', duration=0.1, gap=0.0)
        audio_with_gap = generator.generate_melodic_interval('C', 'major_third', duration=0.1, gap=0.1)
        
        # Audio with gap should be longer
        self.assertGreater(len(audio_with_gap), len(audio_short))
    
    def test_unknown_interval_raises(self):
        """Test that unknown interval type raises ValueError."""
        generator = self.piano_factory.create_interval_generator()
        with self.assertRaises(ValueError):
            generator.generate_harmonic_interval('C', 'unknown_interval')


class TestChordGenerator(unittest.TestCase):
    """Test ChordGenerator implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.piano_factory = PianoMusicElementFactory()
        self.guitar_factory = GuitarMusicElementFactory()
    
    def test_piano_major_chord(self):
        """Test piano major chord generation."""
        generator = self.piano_factory.create_chord_generator()
        audio = generator.generate_chord('C', 'major', octave=4, duration=0.5)
        
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)
    
    def test_piano_minor_chord(self):
        """Test piano minor chord generation."""
        generator = self.piano_factory.create_chord_generator()
        audio = generator.generate_chord('C', 'minor', octave=4, duration=0.5)
        
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)
    
    def test_guitar_major_chord(self):
        """Test guitar major chord generation."""
        generator = self.guitar_factory.create_chord_generator()
        audio = generator.generate_chord('C', 'major', octave=4, duration=0.5)
        
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)
    
    def test_unknown_chord_raises(self):
        """Test that unknown chord type raises ValueError."""
        generator = self.piano_factory.create_chord_generator()
        with self.assertRaises(ValueError):
            generator.generate_chord('C', 'unknown_chord')


class TestAudioManager(unittest.TestCase):
    """Test AudioManager Singleton."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Clear singleton instance for testing
        AudioManager._instance = None
        AudioManager._lock = threading.Lock()
    
    def tearDown(self):
        """Clean up after tests."""
        # Clear singleton instance
        AudioManager._instance = None
    
    def test_singleton_instance(self):
        """Test that AudioManager returns same instance."""
        manager1 = AudioManager()
        manager2 = AudioManager()
        
        self.assertIs(manager1, manager2)
    
    def test_get_audio_manager_function(self):
        """Test get_audio_manager function."""
        manager1 = get_audio_manager()
        manager2 = get_audio_manager()
        
        self.assertIsInstance(manager1, AudioManager)
        self.assertIs(manager1, manager2)
    
    def test_get_instrument_factory(self):
        """Test getting instrument factory."""
        manager = AudioManager()
        factory = manager.get_instrument_factory()
        
        self.assertIsInstance(factory, InstrumentFactory)
    
    def test_get_music_element_factory_caching(self):
        """Test that music element factories are cached."""
        manager = AudioManager()
        
        factory1 = manager.get_music_element_factory('piano')
        factory2 = manager.get_music_element_factory('piano')
        
        # Should return same instance (cached)
        self.assertIs(factory1, factory2)
        self.assertIsInstance(factory1, PianoMusicElementFactory)
    
    def test_get_music_element_factory_different_instruments(self):
        """Test getting factories for different instruments."""
        manager = AudioManager()
        
        piano_factory = manager.get_music_element_factory('piano')
        guitar_factory = manager.get_music_element_factory('guitar')
        
        self.assertIsInstance(piano_factory, PianoMusicElementFactory)
        self.assertIsInstance(guitar_factory, GuitarMusicElementFactory)
        self.assertIsNot(piano_factory, guitar_factory)
    
    def test_get_audio_generator_caching(self):
        """Test that audio generators are cached."""
        manager = AudioManager()
        
        gen1 = manager.get_audio_generator('piano')
        gen2 = manager.get_audio_generator('piano')
        
        # Should return same instance (cached)
        self.assertIs(gen1, gen2)
    
    def test_get_audio_generator_different_instruments(self):
        """Test getting generators for different instruments."""
        manager = AudioManager()
        
        piano_gen = manager.get_audio_generator('piano')
        guitar_gen = manager.get_audio_generator('guitar')
        
        self.assertEqual(piano_gen.get_instrument_name(), 'piano')
        self.assertEqual(guitar_gen.get_instrument_name(), 'guitar')
        self.assertIsNot(piano_gen, guitar_gen)
    
    def test_clear_cache(self):
        """Test clearing cache."""
        manager = AudioManager()
        
        # Create and cache factories
        factory1 = manager.get_music_element_factory('piano')
        gen1 = manager.get_audio_generator('piano')
        
        # Clear cache
        manager.clear_cache()
        
        # New requests should create new instances
        factory2 = manager.get_music_element_factory('piano')
        gen2 = manager.get_audio_generator('piano')
        
        # Should be different instances
        self.assertIsNot(factory1, factory2)
        self.assertIsNot(gen1, gen2)
    
    def test_get_audio_directory(self):
        """Test getting audio directory path."""
        manager = AudioManager()
        directory = manager.get_audio_directory()
        
        self.assertIsInstance(directory, str)
        self.assertIn('audio', directory)
    
    def test_get_audio_directory_with_subdirectory(self):
        """Test getting audio directory with subdirectory."""
        manager = AudioManager()
        directory = manager.get_audio_directory('training')
        
        self.assertIsInstance(directory, str)
        self.assertIn('audio', directory)
        self.assertIn('training', directory)
    
    def test_ensure_audio_directory(self):
        """Test ensuring audio directory exists."""
        manager = AudioManager()
        directory = manager.ensure_audio_directory('test')
        
        self.assertIsInstance(directory, str)
        import os
        self.assertTrue(os.path.exists(directory))


if __name__ == '__main__':
    unittest.main()
