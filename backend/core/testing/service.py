"""
Service layer for test session business logic.

Clean, simple implementation for test session management.
Separates business logic from API views.

Pattern: Service Layer
Purpose: Encapsulate test session operations and coordinate between components.
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
    
    Provides clean interface for creating sessions and submitting answers.
    Coordinates Builder and Observer patterns.
    """
    
    @staticmethod
    def create_session(user: User, test_type: str, **options) -> TestSession:
        """
        Create a new test session.
        
        Simple interface for creating test sessions with all parameters.
        
        Args:
            user: User creating the session
            test_type: Type of test ('interval_recognition')
            **options: Additional parameters:
                - total_questions: Number of questions (default: 10)
                - intervals: List of intervals (default: all available)
                - instrument: Instrument type ('piano', 'guitar', default: 'piano')
            
        Returns:
            Created TestSession instance
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If session creation fails
        """
        builder = TestSessionBuilder(user=user)
        
        # Required parameters
        builder.set_test_type(test_type)
        builder.set_total_questions(options.get('total_questions', 10))
        
        # Optional parameters
        if 'intervals' in options and options['intervals']:
            builder.set_intervals(options['intervals'])
        
        if 'instrument' in options:
            builder.set_instrument(options['instrument'])
        
        # Build and return session
        session = builder.build()
        logger.info(f"Created test session {session.id} for user {user.id}, type: {test_type}")
        
        return session
    
    @staticmethod
    def submit_answer(
        session: TestSession,
        question: TestQuestion,
        answer: Optional[str] = None,
        response_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Submit answer to a test question.
        
        Validates answer, updates question, and checks if session is completed.
        
        Args:
            session: TestSession instance
            question: TestQuestion instance
            answer: Text answer (interval type for interval recognition)
            response_time: Time taken to answer in seconds
            
        Returns:
            Dict with result information:
                - is_correct: Whether answer is correct
                - session_completed: Whether session is completed
                - answered_count: Number of answered questions
                - total_questions: Total number of questions
        """
        # Validate answer
        if session.test_type == 'interval_recognition':
            if not answer:
                raise ValueError("Answer is required for interval recognition")
            
            question.user_answer = answer
            question.is_correct = (answer == question.interval_type)
        else:
            raise ValueError(f"Unsupported test type: {session.test_type}")
        
        # Save question
        question.answered_at = datetime.now()
        if response_time:
            question.response_time = response_time
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
        
        Calculates results, awards XP, and notifies observers about completion.
        
        Args:
            session: TestSession to complete
        """
        # Calculate session results
        session.correct_answers = session.questions.filter(is_correct=True).count()
        session.calculate_accuracy()
        session.calculate_experience()
        session.completed_at = datetime.now()
        session.is_completed = True
        session.save()
        
        # Notify observers about completion
        subject = ProgressSubject()
        subject.attach(XPObserver())
        subject.attach(AchievementObserver())
        subject.attach(AnalyticsObserver())
        
        completion_event = ProgressEvent(
            event_type='test_completed',
            session=session,
            user=session.user,
            data={
                'accuracy': session.accuracy_percentage,
                'correct_answers': session.correct_answers,
                'total_questions': session.total_questions,
                'experience_gained': session.experience_gained,
            },
            timestamp=datetime.now()
        )
        subject.notify(completion_event)
        
        logger.info(f"Completed test session {session.id} for user {session.user.id}")
