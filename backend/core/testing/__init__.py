"""
Testing module for test session management.

Patterns:
- Builder: TestSessionBuilder for constructing complex test sessions
- Observer: ProgressObserver for tracking achievements and XP
"""

from .builder import TestSessionBuilder
from .observers import (
    ProgressObserver,
    XPObserver,
    AchievementObserver,
    AnalyticsObserver,
    ProgressEvent,
    ProgressSubject,
)

__all__ = [
    'TestSessionBuilder',
    'ProgressObserver',
    'XPObserver',
    'AchievementObserver',
    'AnalyticsObserver',
    'ProgressEvent',
    'ProgressSubject',
]
