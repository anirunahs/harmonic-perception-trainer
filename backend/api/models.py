from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    experience_points = models.IntegerField(default=0)
    level = models.IntegerField(default=1)
    
    vocal_range_min_frequency = models.FloatField(null=True, blank=True)
    vocal_range_max_frequency = models.FloatField(null=True, blank=True)
    vocal_range_min_note = models.CharField(max_length=10, null=True, blank=True)
    vocal_range_max_note = models.CharField(max_length=10, null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username} - Level {self.level} ({self.experience_points} XP)"
    
    def add_experience(self, points):
        """Додає досвід та перевіряє підвищення рівня"""
        self.experience_points += points
        new_level = self.calculate_level(self.experience_points)
        
        level_up = new_level > self.level
        self.level = new_level
        self.save()
        
        return level_up
    
    @staticmethod
    def calculate_level(experience_points):
        """Розрахунок рівня на основі досвіду (100 XP за рівень + 50 XP за кожен наступний)"""
        if experience_points < 100:
            return 1
        
        level = 1
        required_xp = 100
        remaining_xp = experience_points
        
        while remaining_xp >= required_xp:
            remaining_xp -= required_xp
            level += 1
            required_xp += 50
            
        return level
    
    @property
    def xp_for_next_level(self):
        """Скільки XP потрібно для наступного рівня"""
        current_level_start_xp = 0
        required_xp = 100
        
        for i in range(1, self.level):
            current_level_start_xp += required_xp
            required_xp += 50
            
        next_level_xp = current_level_start_xp + required_xp
        return next_level_xp - self.experience_points


class TestSession(models.Model):
    TEST_TYPES = [
        ('interval_recognition', 'Розпізнавання інтервалів'),
        ('note_reproduction', 'Відтворення нот'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='test_sessions')
    test_type = models.CharField(max_length=20, choices=TEST_TYPES)
    total_questions = models.IntegerField()
    correct_answers = models.IntegerField(default=0)
    experience_gained = models.IntegerField(default=0)
    accuracy_percentage = models.FloatField(default=0.0)
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    is_completed = models.BooleanField(default=False)
    
    def __str__(self):
        return f"{self.user.username} - {self.get_test_type_display()} ({self.accuracy_percentage}%)"
    
    def calculate_accuracy(self):
        """Розрахунок точності"""
        if self.total_questions > 0:
            self.accuracy_percentage = (self.correct_answers / self.total_questions) * 100
        return self.accuracy_percentage
    
    def calculate_experience(self):
        """Розрахунок досвіду на основі результатів"""
        base_xp = 10
        accuracy_bonus = int(self.accuracy_percentage)
        perfect_bonus = 20 if self.accuracy_percentage == 100 else 0
        
        self.experience_gained = base_xp + accuracy_bonus + perfect_bonus
        return self.experience_gained
    
class TestQuestion(models.Model):
    session = models.ForeignKey(TestSession, on_delete=models.CASCADE, related_name='questions')
    question_number = models.IntegerField()
    
    interval_type = models.CharField(max_length=20, null=True, blank=True)
    base_note = models.CharField(max_length=5, null=True, blank=True)
    target_note = models.CharField(max_length=5, null=True, blank=True)
    harmonic_audio_url = models.CharField(max_length=500, null=True, blank=True)
    melodic_audio_url = models.CharField(max_length=500, null=True, blank=True)
    
    target_frequency = models.FloatField(null=True, blank=True)
    target_note_name = models.CharField(max_length=10, null=True, blank=True)
    reference_audio_url = models.CharField(max_length=500, null=True, blank=True)
    
    user_answer = models.CharField(max_length=50, null=True, blank=True)
    is_correct = models.BooleanField(default=False)
    answered_at = models.DateTimeField(null=True, blank=True)
    
    recorded_frequency = models.FloatField(null=True, blank=True)
    frequency_tolerance = models.FloatField(default=10.0)  # Допустиме відхилення в Гц
    
    def __str__(self):
        return f"Question {self.question_number} - {self.session.get_test_type_display()}"
    
    def check_frequency_answer(self, recorded_freq):
        """Перевіряє правильність відтвореної частоти"""
        if self.target_frequency and recorded_freq:
            difference = abs(self.target_frequency - recorded_freq)
            self.recorded_frequency = recorded_freq
            self.is_correct = difference <= self.frequency_tolerance
            return self.is_correct
        return False


class Achievement(models.Model):
    ACHIEVEMENT_TYPES = [
        ('level', 'Досягнення рівня'),
        ('accuracy', 'Точність'),
        ('streak', 'Серія правильних відповідей'),
        ('tests_completed', 'Кількість тестів'),
    ]
    
    name = models.CharField(max_length=100)
    description = models.TextField()
    achievement_type = models.CharField(max_length=20, choices=ACHIEVEMENT_TYPES)
    requirement_value = models.IntegerField()
    experience_reward = models.IntegerField(default=50)
    icon = models.CharField(max_length=50, default='award')
    
    def __str__(self):
        return self.name


class UserAchievement(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='achievements')
    achievement = models.ForeignKey(Achievement, on_delete=models.CASCADE)
    earned_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ('user', 'achievement')
    
    def __str__(self):
        return f"{self.user.username} - {self.achievement.name}"