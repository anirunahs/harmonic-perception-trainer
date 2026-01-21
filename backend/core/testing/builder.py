"""
Builder Pattern for TestSession creation.

Pattern: Builder (Creational)
Purpose: Construct complex TestSession objects step by step with validation.
"""

import random
import logging
from typing import List, Optional, Dict, Any
from django.contrib.auth.models import User

from api.models import TestSession, TestQuestion, UserProfile
from core.audio import (
    InstrumentFactory,
    NOTE_FREQUENCIES,
    INTERVALS_SEMITONES,
    get_target_note,
    frequency_to_note,
)

logger = logging.getLogger(__name__)


class TestSessionBuilder:
    """
    Builder for constructing TestSession objects.
    
    Provides fluent interface for step-by-step construction of test sessions
    with validation and error handling.
    
    Usage:
        builder = TestSessionBuilder(user)
        session = (builder
            .set_test_type('interval_recognition')
            .set_total_questions(10)
            .set_intervals(['major_third', 'perfect_fifth'])
            .set_instrument('piano')
            .build())
    """
    
    def __init__(self, user: User):
        """
        Initialize builder with user.
        
        Args:
            user: User for whom the test session is created
        """
        self._user = user
        self._reset()
    
    def _reset(self):
        """Reset builder state to defaults."""
        self._test_type: Optional[str] = None
        self._total_questions: int = 10
        self._intervals: List[str] = []
        self._base_notes: List[str] = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        self._octaves: List[int] = [3, 4, 5]
        self._instrument: str = 'piano'
        self._difficulty: str = 'medium'
        self._time_limit: Optional[int] = None
        self._enable_hints: bool = False
        self._auto_next: bool = False
        self._random_order: bool = False
        self._include_reference_note: bool = True
    
    def set_test_type(self, test_type: str) -> 'TestSessionBuilder':
        """
        Set test type.
        
        Args:
            test_type: 'interval_recognition' or 'note_reproduction'
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If test type is invalid
        """
        if test_type not in [choice[0] for choice in TestSession.TEST_TYPES]:
            raise ValueError(f"Invalid test type: {test_type}")
        self._test_type = test_type
        return self
    
    def set_total_questions(self, count: int) -> 'TestSessionBuilder':
        """
        Set total number of questions.
        
        Args:
            count: Number of questions (1-50)
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If count is out of valid range
        """
        if count < 1 or count > 50:
            raise ValueError("Total questions must be between 1 and 50")
        self._total_questions = count
        return self
    
    def set_intervals(self, intervals: List[str]) -> 'TestSessionBuilder':
        """
        Set available intervals for interval recognition test.
        
        Args:
            intervals: List of interval types
            
        Returns:
            Self for method chaining
        """
        self._intervals = intervals
        return self
    
    def set_base_notes(self, notes: List[str]) -> 'TestSessionBuilder':
        """
        Set base notes for interval generation.
        
        Args:
            notes: List of note names (e.g., ['C', 'D', 'E'])
            
        Returns:
            Self for method chaining
        """
        self._base_notes = notes
        return self
    
    def set_octaves(self, octaves: List[int]) -> 'TestSessionBuilder':
        """
        Set octaves for note generation.
        
        Args:
            octaves: List of octave numbers (e.g., [3, 4, 5])
            
        Returns:
            Self for method chaining
        """
        self._octaves = octaves
        return self
    
    def set_instrument(self, instrument: str) -> 'TestSessionBuilder':
        """
        Set instrument for audio generation.
        
        Args:
            instrument: Instrument type ('piano', 'guitar', 'synth')
            
        Returns:
            Self for method chaining
        """
        self._instrument = instrument
        return self
    
    def set_difficulty(self, difficulty: str) -> 'TestSessionBuilder':
        """
        Set test difficulty level.
        
        Args:
            difficulty: 'easy', 'medium', or 'hard'
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If difficulty is invalid
        """
        valid_difficulties = ['easy', 'medium', 'hard']
        if difficulty not in valid_difficulties:
            raise ValueError(f"Difficulty must be one of: {', '.join(valid_difficulties)}")
        self._difficulty = difficulty
        return self
    
    def set_time_limit(self, seconds: Optional[int]) -> 'TestSessionBuilder':
        """
        Set time limit for test completion.
        
        Args:
            seconds: Time limit in seconds (None to disable)
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If time limit is negative
        """
        if seconds is not None and seconds < 0:
            raise ValueError("Time limit must be positive or None")
        self._time_limit = seconds
        return self
    
    def enable_hints(self, enabled: bool = True) -> 'TestSessionBuilder':
        """
        Enable or disable hints.
        
        Args:
            enabled: Whether hints are enabled
            
        Returns:
            Self for method chaining
        """
        self._enable_hints = enabled
        return self
    
    def set_auto_next(self, enabled: bool = True) -> 'TestSessionBuilder':
        """
        Enable or disable auto-advance to next question.
        
        Args:
            enabled: Whether to auto-advance
            
        Returns:
            Self for method chaining
        """
        self._auto_next = enabled
        return self
    
    def set_random_order(self, enabled: bool = True) -> 'TestSessionBuilder':
        """
        Enable or disable random question order.
        
        Args:
            enabled: Whether to randomize order
            
        Returns:
            Self for method chaining
        """
        self._random_order = enabled
        return self
    
    def set_include_reference_note(self, enabled: bool = True) -> 'TestSessionBuilder':
        """
        Enable or disable reference note playback.
        
        Args:
            enabled: Whether to include reference note
            
        Returns:
            Self for method chaining
        """
        self._include_reference_note = enabled
        return self
    
    def build(self) -> TestSession:
        """
        Build and return TestSession with all questions.
        
        Returns:
            Created TestSession instance
            
        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If session creation fails
        """
        self._validate()
        
        try:
            # Create session with all parameters
            session = TestSession.objects.create(
                user=self._user,
                test_type=self._test_type,
                total_questions=self._total_questions
            )
            
            # Store additional settings (can be extended to model fields if needed)
            # For now, we'll use session metadata or extend the model
            
            # Create questions based on test type
            if self._test_type == 'interval_recognition':
                self._create_interval_questions(session)
            elif self._test_type == 'note_reproduction':
                self._create_note_questions(session)
            
            # Generate audio files
            self._generate_test_audio(session)
            
            logger.info(f"TestSession {session.id} created successfully for user {self._user.id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to create TestSession: {e}")
            raise RuntimeError(f"Failed to create test session: {str(e)}")
    
    def _validate(self):
        """Validate builder state before building."""
        if not self._test_type:
            raise ValueError("Test type is required")
        
        if self._test_type == 'interval_recognition' and not self._intervals:
            # Use default intervals if none specified
            self._intervals = [
                'minor_second', 'major_second', 'minor_third', 'major_third',
                'perfect_fourth', 'tritone', 'perfect_fifth', 'minor_sixth',
                'major_sixth', 'minor_seventh', 'major_seventh', 'perfect_octave'
            ]
    
    def _create_interval_questions(self, session: TestSession):
        """Create interval recognition questions."""
        for i in range(self._total_questions):
            interval_type = random.choice(self._intervals)
            base_note = random.choice(self._base_notes)
            octave = random.choice(self._octaves)
            base_note_with_octave = f"{base_note}{octave}"
            
            target_note = self._calculate_target_note(base_note, interval_type)
            target_note_with_octave = f"{target_note}{octave}"
            
            TestQuestion.objects.create(
                session=session,
                question_number=i + 1,
                interval_type=interval_type,
                base_note=base_note_with_octave,
                target_note=target_note_with_octave,
                harmonic_audio_url="",
                melodic_audio_url=""
            )
    
    def _create_note_questions(self, session: TestSession):
        """Create note reproduction questions."""
        user_profile, _ = UserProfile.objects.get_or_create(user=self._user)
        
        if user_profile.vocal_range_min_frequency and user_profile.vocal_range_max_frequency:
            min_freq = user_profile.vocal_range_min_frequency
            max_freq = user_profile.vocal_range_max_frequency
        else:
            min_freq = 130.81  # C3
            max_freq = 523.25  # C5
        
        for i in range(self._total_questions):
            target_frequency = self._generate_random_frequency(min_freq, max_freq)
            target_note = frequency_to_note(target_frequency)
            
            TestQuestion.objects.create(
                session=session,
                question_number=i + 1,
                target_frequency=target_frequency,
                target_note_name=target_note,
                reference_audio_url=f"/api/testing/audio/placeholder_note_{i+1}.wav",
                frequency_tolerance=20.0
            )
    
    def _calculate_target_note(self, base_note: str, interval_type: str) -> str:
        """Calculate target note based on base note and interval."""
        semitones = INTERVALS_SEMITONES.get(interval_type, 0)
        return get_target_note(base_note, semitones)
    
    def _generate_random_frequency(self, min_freq: float, max_freq: float) -> float:
        """Generate random frequency that corresponds to a musical note."""
        frequencies = []
        current_freq = min_freq
        
        while current_freq <= max_freq:
            frequencies.append(current_freq)
            current_freq *= 2**(1/12)
        
        return random.choice(frequencies)
    
    def _generate_test_audio(self, session: TestSession):
        """Generate audio files for test questions."""
        from django.conf import settings
        import os
        import shutil
        from scipy.io import wavfile
        import numpy as np
        from core.audio import (
            get_note_frequency,
            combine_tones_harmonic,
            combine_tones_melodic,
            save_audio,
        )
        
        audio_dir = os.path.join(settings.MEDIA_ROOT, 'testing_audio')
        os.makedirs(audio_dir, exist_ok=True)
        
        user_audio_dir = os.path.join(audio_dir, str(self._user.id))
        if os.path.exists(user_audio_dir):
            shutil.rmtree(user_audio_dir)
        os.makedirs(user_audio_dir, exist_ok=True)
        
        # Create generator using Factory Method
        factory = InstrumentFactory()
        generator = factory.create_generator(self._instrument)
        
        for question in session.questions.all():
            if session.test_type == 'interval_recognition':
                # Extract note name (without octave) from question
                base_note_name = question.base_note[:-1] if question.base_note[-1].isdigit() else question.base_note
                target_note_name = question.target_note[:-1] if question.target_note[-1].isdigit() else question.target_note
                
                # Get frequencies
                base_freq = get_note_frequency(base_note_name)
                target_freq = get_note_frequency(target_note_name)
                
                # Generate tones using factory-created generator
                base_tone = generator.generate_tone(base_freq, duration=2.0)
                target_tone = generator.generate_tone(target_freq, duration=2.0)
                
                # Combine tones
                harmonic_audio = combine_tones_harmonic(base_tone, target_tone)
                melodic_audio = combine_tones_melodic(base_tone, target_tone)
                
                # Save audio files
                harmonic_filename = f"q{question.question_number}_harmonic.wav"
                melodic_filename = f"q{question.question_number}_melodic.wav"
                
                harmonic_path = os.path.join(user_audio_dir, harmonic_filename)
                melodic_path = os.path.join(user_audio_dir, melodic_filename)
                
                save_audio(harmonic_audio, harmonic_path)
                save_audio(melodic_audio, melodic_path)
                
                question.harmonic_audio_url = f"/api/testing/audio/{self._user.id}/{harmonic_filename}"
                question.melodic_audio_url = f"/api/testing/audio/{self._user.id}/{melodic_filename}"
                question.save()
