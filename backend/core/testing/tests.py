"""Unit tests for Builder and Observer patterns."""

import os
import unittest
from unittest.mock import patch, MagicMock
from django.test import TestCase
from django.contrib.auth.models import User

# Suppress TensorFlow warnings for cleaner test output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from api.models import TestSession, TestQuestion, UserProfile, Achievement, UserAchievement
from .builder import TestSessionBuilder
from .observers import XPObserver, AchievementObserver, AnalyticsObserver, ProgressEvent, ProgressSubject
from .service import TestSessionService


class TestTestSessionBuilder(TestCase):
    """Tests for TestSessionBuilder."""
    
    def setUp(self):
        self.user = User.objects.create_user('testuser', 'test@test.com', 'password')
        UserProfile.objects.create(user=self.user)
    
    def test_builder_creates_session(self):
        """Verify builder creates TestSession."""
        builder = TestSessionBuilder(user=self.user)
        session = (builder
            .set_test_type('interval_recognition')
            .set_total_questions(5)
            .set_intervals(['major_third'])
            .set_instrument('piano')
            .build())
        
        self.assertIsInstance(session, TestSession)
        self.assertEqual(session.user, self.user)
        self.assertEqual(session.test_type, 'interval_recognition')
        self.assertEqual(session.total_questions, 5)
    
    def test_builder_creates_questions(self):
        """Verify builder creates questions."""
        builder = TestSessionBuilder(user=self.user)
        session = (builder
            .set_test_type('interval_recognition')
            .set_total_questions(3)
            .set_intervals(['major_third'])
            .set_instrument('piano')
            .build())
        
        self.assertEqual(session.questions.count(), 3)
        for question in session.questions.all():
            self.assertIsInstance(question, TestQuestion)
    
    def test_builder_fluent_interface(self):
        """Verify builder supports method chaining."""
        builder = TestSessionBuilder(user=self.user)
        result = (builder
            .set_test_type('interval_recognition')
            .set_total_questions(10)
            .set_intervals(['major_third'])
            .set_instrument('piano'))
        
        self.assertIsInstance(result, TestSessionBuilder)
    
    def test_builder_validates_test_type(self):
        """Verify builder validates test type."""
        builder = TestSessionBuilder(user=self.user)
        with self.assertRaises(ValueError):
            builder.build()


class TestXPObserver(TestCase):
    """Tests for XPObserver."""
    
    def setUp(self):
        self.user = User.objects.create_user('testuser', 'test@test.com', 'password')
        self.profile = UserProfile.objects.create(user=self.user, experience_points=0, level=1)
        self.session = TestSession.objects.create(
            user=self.user,
            test_type='interval_recognition',
            total_questions=5
        )
        self.observer = XPObserver()
    
    def test_xp_observer_awards_xp(self):
        """Verify XP observer awards experience."""
        event = ProgressEvent(
            event_type='test_completed',
            session=self.session,
            user=self.user,
            data={'experience_gained': 50},
            timestamp=None
        )
        
        self.observer.update(event)
        
        self.profile.refresh_from_db()
        self.assertGreater(self.profile.experience_points, 0)
    
    def test_xp_observer_handles_level_up(self):
        """Verify XP observer handles level up."""
        # Set profile to near level up
        self.profile.experience_points = 90
        self.profile.level = 1
        self.profile.save()
        
        event = ProgressEvent(
            event_type='test_completed',
            session=self.session,
            user=self.user,
            data={'experience_gained': 20},
            timestamp=None
        )
        
        self.observer.update(event)
        
        self.profile.refresh_from_db()
        self.assertGreater(self.profile.level, 1)


class TestAchievementObserver(TestCase):
    """Tests for AchievementObserver."""
    
    def setUp(self):
        self.user = User.objects.create_user('testuser', 'test@test.com', 'password')
        self.profile = UserProfile.objects.create(user=self.user)
        self.session = TestSession.objects.create(
            user=self.user,
            test_type='interval_recognition',
            total_questions=5,
            correct_answers=5,
            accuracy_percentage=100.0
        )
        self.observer = AchievementObserver()
    
    def test_achievement_observer_checks_achievements(self):
        """Verify achievement observer checks achievements."""
        # Mark session as completed
        self.session.is_completed = True
        self.session.save()
        
        # Create test achievement
        achievement = Achievement.objects.create(
            name='Perfect Score',
            description='Get 100% accuracy',
            achievement_type='accuracy',
            requirement_value=100
        )
        
        from datetime import datetime
        event = ProgressEvent(
            event_type='test_completed',
            session=self.session,
            user=self.user,
            data={},
            timestamp=datetime.now()
        )
        
        self.observer.update(event)
        
        # Check if achievement was awarded
        user_achievements = UserAchievement.objects.filter(
            user=self.user,
            achievement=achievement
        )
        self.assertGreater(user_achievements.count(), 0)


