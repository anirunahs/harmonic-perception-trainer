from django.contrib.auth.models import User
from rest_framework import serializers
from .models import UserProfile, TestSession, TestQuestion, Achievement, UserAchievement

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["id", "username", "password"]
        extra_kwargs = {"password": {"write_only": True}}

    def create(self, validated_data):
        user = User.objects.create_user(**validated_data)
        return user
    
class UserProfileSerializer(serializers.ModelSerializer):
    xp_for_next_level = serializers.ReadOnlyField()
    
    class Meta:
        model = UserProfile
        fields = [
            'id', 'experience_points', 'level', 'xp_for_next_level',
            'vocal_range_min_frequency', 'vocal_range_max_frequency',
            'vocal_range_min_note', 'vocal_range_max_note',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['experience_points', 'level', 'created_at', 'updated_at']


class AchievementSerializer(serializers.ModelSerializer):
    class Meta:
        model = Achievement
        fields = '__all__'


class UserAchievementSerializer(serializers.ModelSerializer):
    achievement = AchievementSerializer(read_only=True)
    
    class Meta:
        model = UserAchievement
        fields = ['achievement', 'earned_at']


class TestQuestionSerializer(serializers.ModelSerializer):
    class Meta:
        model = TestQuestion
        fields = [
            'id', 'question_number', 'interval_type', 'base_note', 'target_note',
            'harmonic_audio_url', 'melodic_audio_url', 'target_frequency',
            'target_note_name', 'reference_audio_url', 'user_answer', 'is_correct',
            'answered_at', 'recorded_frequency', 'frequency_tolerance'
        ]
        read_only_fields = ['id', 'harmonic_audio_url', 'melodic_audio_url', 'reference_audio_url']


class TestSessionSerializer(serializers.ModelSerializer):
    questions = TestQuestionSerializer(many=True, read_only=True)
    accuracy_percentage = serializers.ReadOnlyField()
    experience_gained = serializers.ReadOnlyField()
    
    class Meta:
        model = TestSession
        fields = [
            'id', 'test_type', 'total_questions', 'correct_answers',
            'experience_gained', 'accuracy_percentage', 'started_at',
            'completed_at', 'is_completed', 'questions'
        ]
        read_only_fields = [
            'id', 'correct_answers', 'experience_gained', 'accuracy_percentage',
            'started_at', 'completed_at'
        ]


class CreateTestSessionSerializer(serializers.Serializer):
    test_type = serializers.ChoiceField(choices=TestSession.TEST_TYPES)
    total_questions = serializers.IntegerField(min_value=1, max_value=50, default=10)
    intervals = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        help_text="Список інтервалів для тесту розпізнавання"
    )
    difficulty = serializers.ChoiceField(
        choices=[('easy', 'Легкий'), ('medium', 'Середній'), ('hard', 'Складний')],
        default='medium',
        required=False
    )
    instrument = serializers.ChoiceField(
        choices=[('piano', 'Фортепіано'), ('guitar', 'Гітара'), ('synth', 'Синтезатор')],
        default='piano',
        required=False,
        help_text="Інструмент для генерації звуку"
    )
    time_limit = serializers.IntegerField(
        min_value=0,
        required=False,
        allow_null=True,
        help_text="Час на тест в секундах (null = без обмеження)"
    )
    enable_hints = serializers.BooleanField(
        default=False,
        required=False,
        help_text="Дозволити підказки"
    )
    auto_next = serializers.BooleanField(
        default=False,
        required=False,
        help_text="Автоматично переходити до наступного питання"
    )
    random_order = serializers.BooleanField(
        default=False,
        required=False,
        help_text="Випадковий порядок питань"
    )
    include_reference_note = serializers.BooleanField(
        default=True,
        required=False,
        help_text="Включати опорну ноту"
    )


class SubmitAnswerSerializer(serializers.Serializer):
    question_id = serializers.IntegerField()
    answer = serializers.CharField(max_length=50, required=False)
    recorded_frequency = serializers.FloatField(required=False)
    
    def validate(self, data):
        if not data.get('answer') and not data.get('recorded_frequency'):
            raise serializers.ValidationError(
                "Потрібно надати або текстову відповідь, або записану частоту"
            )
        return data


class VocalRangeSetupSerializer(serializers.Serializer):
    min_frequency = serializers.FloatField(min_value=50.0, max_value=2000.0)
    max_frequency = serializers.FloatField(min_value=50.0, max_value=2000.0)
    min_note = serializers.CharField(max_length=10)
    max_note = serializers.CharField(max_length=10)
    
    def validate(self, data):
        if data['min_frequency'] >= data['max_frequency']:
            raise serializers.ValidationError(
                "Мінімальна частота повинна бути меншою за максимальну"
            )
        return data