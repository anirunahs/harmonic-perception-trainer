"""
Observer Pattern for test session progress tracking.

Pattern: Observer (Behavioral)
Purpose: Decouple achievement and XP systems from test session completion logic.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List

from django.contrib.auth.models import User
from api.models import TestSession, Achievement, UserAchievement, UserProfile

logger = logging.getLogger(__name__)


@dataclass
class ProgressEvent:
    """
    Event data for progress observers.
    
    Attributes:
        event_type: Type of event ('answer_submitted', 'correct_answer', 
                                  'test_completed', 'level_up')
        session: TestSession instance
        user: User instance
        data: Additional event data
        timestamp: Event timestamp
    """
    event_type: str
    session: TestSession
    user: User
    data: Dict[str, Any]
    timestamp: datetime


class ProgressObserver(ABC):
    """
    Abstract observer for test session progress events.
    
    Pattern: Observer (Behavioral) - Observer Interface
    """
    
    @abstractmethod
    def update(self, event: ProgressEvent):
        """
        Handle progress event.
        
        Args:
            event: ProgressEvent with event data
        """
        pass


class XPObserver(ProgressObserver):
    """
    Observer for awarding experience points.
    
    Pattern: Observer (Behavioral) - Concrete Observer
    """
    
    def update(self, event: ProgressEvent):
        """Award XP based on event type."""
        if event.event_type == 'correct_answer':
            self._award_answer_xp(event)
        elif event.event_type == 'test_completed':
            self._award_completion_xp(event)
        elif event.event_type == 'level_up':
            self._award_level_bonus(event)
    
    def _award_answer_xp(self, event: ProgressEvent):
        """Award XP for correct answer."""
        base_xp = 2
        streak_bonus = event.data.get('streak', 0) * 1
        
        profile, _ = UserProfile.objects.get_or_create(user=event.user)
        profile.add_experience(base_xp + streak_bonus)
        
        logger.info(f"Awarded {base_xp + streak_bonus} XP to user {event.user.id} for correct answer")
    
    def _award_completion_xp(self, event: ProgressEvent):
        """Award XP for test completion."""
        session = event.session
        experience_gained = event.data.get('experience_gained', session.experience_gained)
        
        profile, _ = UserProfile.objects.get_or_create(user=event.user)
        level_up = profile.add_experience(experience_gained)
        
        # Store level up info in event data for other observers
        if level_up:
            event.data['level_up'] = True
            event.data['new_level'] = profile.level
        
        logger.info(f"Awarded {experience_gained} XP to user {event.user.id} for test completion")
    
    def _award_level_bonus(self, event: ProgressEvent):
        """Award bonus XP for level up."""
        bonus_xp = 50
        profile, _ = UserProfile.objects.get_or_create(user=event.user)
        profile.add_experience(bonus_xp)
        
        logger.info(f"Awarded {bonus_xp} bonus XP to user {event.user.id} for level up")


class AchievementObserver(ProgressObserver):
    """
    Observer for checking and awarding achievements.
    
    Pattern: Observer (Behavioral) - Concrete Observer
    """
    
    def update(self, event: ProgressEvent):
        """Check and award achievements based on event type."""
        if event.event_type == 'test_completed':
            self._check_completion_achievements(event)
            self._check_accuracy_achievements(event)
            # Check level achievements if level up occurred
            if event.data.get('level_up'):
                self._check_level_achievements(event)
        elif event.event_type == 'level_up':
            self._check_level_achievements(event)
        elif event.event_type == 'correct_answer':
            self._check_streak_achievements(event)
    
    def _check_completion_achievements(self, event: ProgressEvent):
        """Check achievements for test completion."""
        user = event.user
        completed_count = TestSession.objects.filter(
            user=user,
            is_completed=True
        ).count()
        
        achievements = Achievement.objects.filter(
            achievement_type='tests_completed',
            requirement_value__lte=completed_count
        )
        
        for achievement in achievements:
            UserAchievement.objects.get_or_create(
                user=user,
                achievement=achievement
            )
            logger.info(f"User {user.id} unlocked achievement: {achievement.name}")
    
    def _check_accuracy_achievements(self, event: ProgressEvent):
        """Check achievements for accuracy."""
        session = event.session
        user = event.user
        
        accuracy = session.accuracy_percentage
        
        achievements = Achievement.objects.filter(
            achievement_type='accuracy',
            requirement_value__lte=accuracy
        )
        
        for achievement in achievements:
            UserAchievement.objects.get_or_create(
                user=user,
                achievement=achievement
            )
            logger.info(f"User {user.id} unlocked accuracy achievement: {achievement.name}")
    
    def _check_level_achievements(self, event: ProgressEvent):
        """Check achievements for level milestones."""
        user = event.user
        new_level = event.data.get('new_level', 1)
        
        # Get user profile to check current level
        try:
            profile = UserProfile.objects.get(user=user)
            current_level = profile.level
        except UserProfile.DoesNotExist:
            return
        
        achievements = Achievement.objects.filter(
            achievement_type='level',
            requirement_value=current_level
        )
        
        for achievement in achievements:
            UserAchievement.objects.get_or_create(
                user=user,
                achievement=achievement
            )
            logger.info(f"User {user.id} unlocked level achievement: {achievement.name}")
    
    def _check_streak_achievements(self, event: ProgressEvent):
        """Check achievements for answer streaks."""
        streak = event.data.get('streak', 0)
        
        if streak >= 10:  # Example threshold
            user = event.user
            achievements = Achievement.objects.filter(
                achievement_type='streak',
                requirement_value__lte=streak
            )
            
            for achievement in achievements:
                UserAchievement.objects.get_or_create(
                    user=user,
                    achievement=achievement
                )
                logger.info(f"User {user.id} unlocked streak achievement: {achievement.name}")


class AnalyticsObserver(ProgressObserver):
    """
    Observer for logging analytics data.
    
    Pattern: Observer (Behavioral) - Concrete Observer
    """
    
    def update(self, event: ProgressEvent):
        """Log analytics data for events."""
        analytics_data = {
            'event_type': event.event_type,
            'user_id': event.user.id,
            'session_id': event.session.id if event.session else None,
            'timestamp': event.timestamp.isoformat(),
            **event.data
        }
        
        # In production, this would send to analytics service
        logger.info(f"Analytics: {analytics_data}")


class ProgressSubject:
    """
    Subject for managing observers and notifying them of events.
    
    Pattern: Observer (Behavioral) - Subject
    """
    
    def __init__(self):
        """Initialize subject with empty observer list."""
        self._observers: List[ProgressObserver] = []
    
    def attach(self, observer: ProgressObserver):
        """
        Attach observer to subject.
        
        Args:
            observer: Observer to attach
        """
        if observer not in self._observers:
            self._observers.append(observer)
            logger.debug(f"Attached observer {type(observer).__name__}")
    
    def detach(self, observer: ProgressObserver):
        """
        Detach observer from subject.
        
        Args:
            observer: Observer to detach
        """
        if observer in self._observers:
            self._observers.remove(observer)
            logger.debug(f"Detached observer {type(observer).__name__}")
    
    def notify(self, event: ProgressEvent):
        """
        Notify all observers of event.
        
        Args:
            event: ProgressEvent to notify observers about
        """
        for observer in self._observers:
            try:
                observer.update(event)
            except Exception as e:
                logger.error(f"Error in observer {type(observer).__name__}: {e}")
