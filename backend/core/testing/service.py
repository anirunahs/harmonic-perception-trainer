"""
Service layer for test session business logic.

Separates business logic from API views.
"""

import logging
from typing import Optional, Dict, Any
from django.contrib.auth.models import User

from api.models import TestSession, TestQuestion
from .builder import TestSessionBuilder
from .observers import ProgressSubject, XPObserver, AchievementObserver, AnalyticsObserver, ProgressEvent
from datetime import datetime

logger = logging.getLogger(__name__)


class TestSessionService:
    """
    Service for test session business logic.
    
    Encapsulates complex operations and coordinates between
    Builder, Observer patterns, and models.
    """
    
    @staticmethod
    def create_session(user: User, test_type: str, **options) -> TestSession:
        """
        Create a new test session with all parameters.
        
        Args:
            user: User creating the session
            test_type: Type of test ('interval_recognition' or 'note_reproduction')
            **options: Additional test parameters
            
        Returns:
            Created TestSession instance
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If session creation fails
        """
        builder = TestSessionBuilder(user=user)
        
        # Set required parameters
        builder.set_test_type(test_type)
        builder.set_total_questions(options.get('total_questions', 10))
        
        # Set optional parameters
        if 'difficulty' in options:
            builder.set_difficulty(options['difficulty'])
        if 'time_limit' in options:
            builder.set_time_limit(options['time_limit'])
        if 'enable_hints' in options:
            builder.enable_hints(options['enable_hints'])
        if 'auto_next' in options:
            builder.set_auto_next(options['auto_next'])
        if 'random_order' in options:
            builder.set_random_order(options['random_order'])
        if 'include_reference_note' in options:
            builder.set_include_reference_note(options['include_reference_note'])
        
        # Set test-type specific parameters
        if test_type == 'interval_recognition':
            if 'intervals' in options and options['intervals']:
                builder.set_intervals(options['intervals'])
            if 'instrument' in options:
                builder.set_instrument(options['instrument'])
        elif test_type == 'note_reproduction':
            # Note reproduction specific settings
            pass
        
        # Build session
        session = builder.build()
        logger.info(f"Created test session {session.id} for user {user.id}")
        
        return session
    
    @staticmethod
    def submit_answer(
        session: TestSession,
        question: TestQuestion,
        answer: Optional[str] = None,
        recorded_frequency: Optional[float] = None,
        confidence: Optional[float] = None,
        response_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Submit answer to a test question.
        
        Args:
            session: TestSession instance
            question: TestQuestion instance
            answer: Text answer (for interval recognition)
            recorded_frequency: Recorded frequency (for note reproduction)
            confidence: Confidence level (0.0-1.0)
            response_time: Time taken to answer in seconds
            
        Returns:
            Dict with result information
        """
        # Validate answer based on test type
        if session.test_type == 'interval_recognition':
            if not answer:
                raise ValueError("Answer is required for interval recognition")
            question.user_answer = answer
            question.is_correct = (answer == question.interval_type)
        elif session.test_type == 'note_reproduction':
            if recorded_frequency is None:
                raise ValueError("Recorded frequency is required for note reproduction")
            question.is_correct = question.check_frequency_answer(recorded_frequency)
        
        question.answered_at = datetime.now()
        question.save()
        
        # Check if session is completed
        answered_count = session.questions.filter(answered_at__isnull=False).count()
        session_completed = answered_count >= session.total_questions
        
        if session_completed:
            TestSessionService._complete_session(session)
        
        return {
            'is_correct': question.is_correct,
            'session_completed': session_completed,
            'answered_count': answered_count,
            'total_questions': session.total_questions
        }
    
    @staticmethod
    def _complete_session(session: TestSession):
        """
        Complete test session and notify observers.
        
        Args:
            session: TestSession to complete
        """
        session.correct_answers = session.questions.filter(is_correct=True).count()
        session.calculate_accuracy()
        session.completed_at = datetime.now()
        session.is_completed = True
        session.save()
        
        # Create subject and attach observers
        subject = ProgressSubject()
        subject.attach(XPObserver())
        subject.attach(AchievementObserver())
        subject.attach(AnalyticsObserver())
        
        # Notify observers about test completion
        completion_event = ProgressEvent(
            event_type='test_completed',
            session=session,
            user=session.user,
            data={
                'accuracy': session.accuracy_percentage,
                'correct_answers': session.correct_answers,
                'total_questions': session.total_questions,
            },
            timestamp=datetime.now()
        )
        subject.notify(completion_event)
        
        # Update session with calculated experience
        session.refresh_from_db()
        session.calculate_experience()
        session.save()
        
        logger.info(f"Completed test session {session.id} for user {session.user.id}")
