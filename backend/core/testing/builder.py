"""
Builder Pattern for TestSession creation.

Pattern: Builder (Creational)
Purpose: Construct complex TestSession objects step by step with validation.

Clean implementation based on requirements:
- Interval recognition tests only
- Simple, clear logic
- Proper interval selection and question generation
"""

import random
import logging
from typing import List, Optional
from django.contrib.auth.models import User

from api.models import TestSession, TestQuestion
from core.audio import (
    InstrumentFactory,
    INTERVALS_SEMITONES,
    get_target_note,
)

logger = logging.getLogger(__name__)


class TestSessionBuilder:
    """
    Builder for constructing TestSession objects.
    
    Provides fluent interface for step-by-step construction of test sessions.
    
    Usage:
        builder = TestSessionBuilder(user)
        session = (builder
            .set_test_type('interval_recognition')
            .set_total_questions(10)
            .set_intervals(['major_third', 'perfect_fifth'])
            .set_instrument('piano')
            .build())
    """
    
    # Available intervals for tests
    AVAILABLE_INTERVALS = [
        'minor_second', 'major_second', 'minor_third', 'major_third',
        'perfect_fourth', 'tritone', 'perfect_fifth', 'minor_sixth',
        'major_sixth', 'minor_seventh', 'major_seventh', 'perfect_octave'
    ]
    
    # Base notes for question generation
    BASE_NOTES = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    
    # Octaves for question generation
    OCTAVES = [3, 4, 5]
    
    def __init__(self, user: User):
        """Initialize builder with user."""
        self._user = user
        self._reset()
        self._audio_factory = InstrumentFactory()
    
    def _reset(self):
        """Reset builder state to defaults."""
        self._test_type: Optional[str] = None
        self._total_questions: int = 10
        self._intervals: List[str] = []
        self._instrument: str = 'piano'
        self._base_notes: List[str] = self.BASE_NOTES.copy()
        self._octaves: List[int] = self.OCTAVES.copy()
    
    def set_test_type(self, test_type: str) -> 'TestSessionBuilder':
        """
        Set test type.
        
        Args:
            test_type: 'interval_recognition' (only supported type)
            
        Returns:
            Self for method chaining
        """
        if test_type != 'interval_recognition':
            raise ValueError(f"Unsupported test type: {test_type}. Only 'interval_recognition' is supported.")
        self._test_type = test_type
        return self
    
    def set_total_questions(self, count: int) -> 'TestSessionBuilder':
        """
        Set total number of questions.
        
        Args:
            count: Number of questions (1-50)
            
        Returns:
            Self for method chaining
        """
        if not 1 <= count <= 50:
            raise ValueError("Total questions must be between 1 and 50")
        self._total_questions = count
        return self
    
    def set_intervals(self, intervals: List[str]) -> 'TestSessionBuilder':
        """
        Set intervals to use in test.
        
        Args:
            intervals: List of interval types (e.g., ['major_third', 'perfect_fifth'])
            
        Returns:
            Self for method chaining
        """
        invalid = [i for i in intervals if i not in self.AVAILABLE_INTERVALS]
        if invalid:
            raise ValueError(f"Invalid intervals: {invalid}")
        self._intervals = intervals.copy()
        return self
    
    def set_instrument(self, instrument: str) -> 'TestSessionBuilder':
        """
        Set instrument for audio generation.
        
        Args:
            instrument: Instrument type ('piano', 'guitar')
            
        Returns:
            Self for method chaining
        """
        available = self._audio_factory.get_available_instruments()
        if instrument not in available:
            raise ValueError(f"Unsupported instrument: {instrument}. Available: {available}")
        self._instrument = instrument
        return self
    
    def set_base_notes(self, notes: List[str]) -> 'TestSessionBuilder':
        """
        Set base notes for question generation.
        
        Args:
            notes: List of note names (e.g., ['C', 'D', 'E'])
            
        Returns:
            Self for method chaining
        """
        self._base_notes = notes.copy()
        return self
    
    def set_octaves(self, octaves: List[int]) -> 'TestSessionBuilder':
        """
        Set octaves for question generation.
        
        Args:
            octaves: List of octave numbers (e.g., [3, 4, 5])
            
        Returns:
            Self for method chaining
        """
        self._octaves = octaves.copy()
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
            # Create session
            session = TestSession.objects.create(
                user=self._user,
                test_type=self._test_type,
                total_questions=self._total_questions
            )
            
            # Create questions
            self._create_interval_questions(session)
            
            # Generate audio files
            self._generate_test_audio(session)
            
            logger.info(f"TestSession {session.id} created for user {self._user.id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to create TestSession: {e}")
            raise RuntimeError(f"Failed to create test session: {str(e)}")
    
    def _validate(self):
        """Validate builder state before building."""
        if not self._test_type:
            raise ValueError("Test type is required")
        
        if self._test_type == 'interval_recognition':
            if not self._intervals:
                # Use all available intervals if none specified
                self._intervals = self.AVAILABLE_INTERVALS.copy()
                logger.info("No intervals specified, using all available intervals")
    
    def _create_interval_questions(self, session: TestSession):
        """
        Create interval recognition questions.
        
        Args:
            session: TestSession instance
        """
        for i in range(self._total_questions):
            # Select random interval
            interval_type = random.choice(self._intervals)
            
            # Select random base note and octave
            base_note = random.choice(self._base_notes)
            octave = random.choice(self._octaves)
            base_note_with_octave = f"{base_note}{octave}"
            
            # Calculate target note
            target_note = self._calculate_target_note(base_note, interval_type)
            target_note_with_octave = f"{target_note}{octave}"
            
            # Create question
            TestQuestion.objects.create(
                session=session,
                question_number=i + 1,
                interval_type=interval_type,
                base_note=base_note_with_octave,
                target_note=target_note_with_octave,
                harmonic_audio_url="",  # Will be set after audio generation
                melodic_audio_url=""
            )
    
    def _calculate_target_note(self, base_note: str, interval_type: str) -> str:
        """
        Calculate target note based on base note and interval.
        
        Args:
            base_note: Base note name (e.g., 'C')
            interval_type: Interval type (e.g., 'major_third')
            
        Returns:
            Target note name (e.g., 'E')
        """
        semitones = INTERVALS_SEMITONES.get(interval_type, 0)
        return get_target_note(base_note, semitones)
    
    def _generate_test_audio(self, session: TestSession):
        """
        Generate audio files for test questions.
        
        Args:
            session: TestSession instance
        """
        from django.conf import settings
        import os
        import shutil
        from core.audio import (
            get_note_frequency,
            combine_tones_harmonic,
            combine_tones_melodic,
            save_audio,
        )
        
        # Setup audio directory
        audio_dir = os.path.join(settings.MEDIA_ROOT, 'testing_audio')
        os.makedirs(audio_dir, exist_ok=True)
        
        user_audio_dir = os.path.join(audio_dir, str(self._user.id))
        if os.path.exists(user_audio_dir):
            shutil.rmtree(user_audio_dir)
        os.makedirs(user_audio_dir, exist_ok=True)
        
        # Create audio generator
        generator = self._audio_factory.create_generator(self._instrument)
        
        # Generate audio for each question
        for question in session.questions.all():
            # Extract note names (without octave)
            base_note_name = question.base_note[:-1] if question.base_note[-1].isdigit() else question.base_note
            target_note_name = question.target_note[:-1] if question.target_note[-1].isdigit() else question.target_note
            
            # Get frequencies
            base_freq = get_note_frequency(base_note_name)
            target_freq = get_note_frequency(target_note_name)
            
            # Generate tones
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
            
            # Update question with audio URLs
            question.harmonic_audio_url = f"/api/testing/audio/{self._user.id}/{harmonic_filename}"
            question.melodic_audio_url = f"/api/testing/audio/{self._user.id}/{melodic_filename}"
            question.save()
