from django.shortcuts import render
from django.contrib.auth.models import User
from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.decorators import authentication_classes, permission_classes
from .models import UserProfile, TestSession, TestQuestion, Achievement, UserAchievement
from .serializers import (UserSerializer, UserProfileSerializer, TestSessionSerializer, 
    CreateTestSessionSerializer, SubmitAnswerSerializer, VocalRangeSetupSerializer,
    AchievementSerializer, UserAchievementSerializer)
from rest_framework.permissions import IsAuthenticated, AllowAny
from django.conf import settings
from django.http import FileResponse, Http404, HttpResponse
from django.views import View
import os
import uuid
import numpy as np
from scipy.io import wavfile
import shutil
import base64
import io
import random
from scipy import signal
from datetime import datetime

# Import audio generation module (Factory Method pattern)
from core.audio import (
    InstrumentFactory,
    NOTE_FREQUENCIES,
    INTERVALS_SEMITONES,
    DEFAULT_SAMPLE_RATE,
    get_note_frequency,
    get_frequency_with_semitone_offset,
    get_target_note,
    save_audio,
    combine_tones_harmonic,
    combine_tones_melodic,
    frequency_to_note,
)


class CreateUserView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [AllowAny]


class GenerateIntervalsView(APIView):
    """
    Generate interval audio files for training.
    
    Uses Factory Method pattern via InstrumentFactory to create
    audio generators for different instruments.
    """
    permission_classes = [IsAuthenticated]
    
    # Supported instruments for audio generation
    SUPPORTED_INSTRUMENTS = ['piano', 'guitar', 'synth']
    DEFAULT_INSTRUMENT = 'piano'
    
    def __init__(self):
        super().__init__()
        self.audio_dir = os.path.join(settings.MEDIA_ROOT, 'training_audio')
        os.makedirs(self.audio_dir, exist_ok=True)
        # Create instrument factory
        self.factory = InstrumentFactory()

    def post(self, request):
        try:
            data = request.data
            intervals = data.get('intervals', [])
            base_note = data.get('base_note', 'C')
            instrument = data.get('instrument', self.DEFAULT_INSTRUMENT)
            duration = data.get('duration', 2.0)
            
            # Validate inputs
            if not intervals:
                return Response(
                    {'error': 'Не вказано інтервали'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            if base_note not in NOTE_FREQUENCIES:
                return Response(
                    {'error': 'Невірна базова нота'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            if instrument not in self.SUPPORTED_INSTRUMENTS:
                return Response(
                    {'error': f'Непідтримуваний інструмент. Доступні: {", ".join(self.SUPPORTED_INSTRUMENTS)}'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Prepare user audio directory
            user_audio_dir = os.path.join(self.audio_dir, str(request.user.id))
            if os.path.exists(user_audio_dir):
                shutil.rmtree(user_audio_dir)
            os.makedirs(user_audio_dir, exist_ok=True)
            
            # Create generator using Factory Method
            generator = self.factory.create_generator(instrument)
            
            generated_intervals = []
            
            for interval_type in intervals:
                if interval_type not in INTERVALS_SEMITONES:
                    continue
                
                semitones = INTERVALS_SEMITONES[interval_type]
                target_note = get_target_note(base_note, semitones)
                
                # Get frequencies using module functions
                base_freq = get_note_frequency(base_note)
                target_freq = get_frequency_with_semitone_offset(base_note, semitones)
                
                # Generate tones using the factory-created generator
                base_tone = generator.generate_tone(base_freq, duration)
                target_tone = generator.generate_tone(target_freq, duration)
                
                # Combine tones for harmonic and melodic intervals
                harmonic_audio = combine_tones_harmonic(base_tone, target_tone)
                melodic_audio = combine_tones_melodic(base_tone, target_tone)
                
                # Generate unique filenames
                interval_id = str(uuid.uuid4())
                harmonic_filename = f"{interval_id}_harmonic.wav"
                melodic_filename = f"{interval_id}_melodic.wav"
                
                # Save audio files
                harmonic_path = os.path.join(user_audio_dir, harmonic_filename)
                melodic_path = os.path.join(user_audio_dir, melodic_filename)
                save_audio(harmonic_audio, harmonic_path)
                save_audio(melodic_audio, melodic_path)
                
                # Build URLs
                harmonic_url = f"/api/training/audio/{request.user.id}/{harmonic_filename}"
                melodic_url = f"/api/training/audio/{request.user.id}/{melodic_filename}"
                
                generated_intervals.append({
                    'id': interval_id,
                    'interval_type': interval_type,
                    'base_note': base_note,
                    'target_note': target_note,
                    'semitones': semitones,
                    'instrument': instrument,
                    'harmonic_url': harmonic_url,
                    'melodic_url': melodic_url
                })
            
            return Response({
                'intervals': generated_intervals,
                'base_note': base_note,
                'instrument': instrument,
                'total_count': len(generated_intervals)
            })
            
        except Exception as e:
            return Response(
                {'error': f'Помилка генерації: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class SecureAudioView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    
    def get(self, request, user_id, filename):
        
        if str(request.user.id) != str(user_id):
            return HttpResponse("Доступ заборонено", status=403)
        
        audio_dir = os.path.join(settings.MEDIA_ROOT, 'training_audio')
        file_path = os.path.join(audio_dir, user_id, filename)
        
        allowed_dir = os.path.join(audio_dir, user_id)
        try:
            real_file_path = os.path.realpath(file_path)
            real_allowed_dir = os.path.realpath(allowed_dir)
            
            if not real_file_path.startswith(real_allowed_dir):
                return HttpResponse("Доступ заборонено", status=403)
                
        except Exception:
            return HttpResponse("Помилка доступу", status=400)
        
        if not os.path.exists(file_path):
            return HttpResponse("Файл не знайдено", status=404)
        
        if not filename.endswith('.wav'):
            return HttpResponse("Непідтримуваний тип файлу", status=400)
        
        try:
            file_size = os.path.getsize(file_path)
            content_type = 'audio/wav'
            
            range_header = request.META.get('HTTP_RANGE')
            
            if range_header:
                range_match = range_header.replace('bytes=', '').split('-')
                start = int(range_match[0]) if range_match[0] else 0
                end = int(range_match[1]) if range_match[1] else file_size - 1
                
                with open(file_path, 'rb') as f:
                    f.seek(start)
                    data = f.read(end - start + 1)
                
                response = HttpResponse(
                    data,
                    status=206,
                    content_type=content_type
                )
                response['Content-Range'] = f'bytes {start}-{end}/{file_size}'
                response['Accept-Ranges'] = 'bytes'
                response['Content-Length'] = str(end - start + 1)
            else:
                response = FileResponse(
                    open(file_path, 'rb'),
                    content_type=content_type,
                    filename=filename
                )
                response['Content-Length'] = str(file_size)
                response['Accept-Ranges'] = 'bytes'
            
            response['Cache-Control'] = 'private, max-age=300'
            response['X-Content-Type-Options'] = 'nosniff'
            
            return response
            
        except Exception as e:
            print(f"Error serving file: {e}")
            return HttpResponse("Помилка при видачі файлу", status=500)


class ClearUserAudioView(APIView):
    permission_classes = [IsAuthenticated]
    
    def delete(self, request):
        """Очищення аудіофайлів користувача"""
        try:
            user_audio_dir = os.path.join(
                settings.MEDIA_ROOT, 
                'training_audio', 
                str(request.user.id)
            )
            
            if os.path.exists(user_audio_dir):
                shutil.rmtree(user_audio_dir)
            
            return Response({'message': 'Файли успішно видалено'})
            
        except Exception as e:
            return Response(
                {'error': f'Помилка видалення файлів: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class UserProfileView(generics.RetrieveUpdateAPIView):
    serializer_class = UserProfileSerializer
    permission_classes = [IsAuthenticated]
    
    def get_object(self):
        profile, created = UserProfile.objects.get_or_create(user=self.request.user)
        return profile


class VocalRangeSetupView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        serializer = VocalRangeSetupSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        profile, created = UserProfile.objects.get_or_create(user=request.user)
        
        profile.vocal_range_min_frequency = data['min_frequency']
        profile.vocal_range_max_frequency = data['max_frequency']
        profile.vocal_range_min_note = data['min_note']
        profile.vocal_range_max_note = data['max_note']
        profile.save()
        
        return Response({'message': 'Вокальний діапазон збережено'})


class TestSessionListView(generics.ListAPIView):
    serializer_class = TestSessionSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return TestSession.objects.filter(
            user=self.request.user,
            is_completed=True
        ).order_by('-completed_at')


class TestSessionDetailView(generics.RetrieveAPIView):
    serializer_class = TestSessionSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return TestSession.objects.filter(user=self.request.user)


class UserAchievementsView(generics.ListAPIView):
    serializer_class = UserAchievementSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return UserAchievement.objects.filter(user=self.request.user).order_by('-earned_at')


class CreateTestSessionView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        serializer = CreateTestSessionSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        test_type = data['test_type']
        total_questions = data['total_questions']
        
        session = TestSession.objects.create(
            user=request.user,
            test_type=test_type,
            total_questions=total_questions
        )
        
        try:
            if test_type == 'interval_recognition':
                self._create_interval_questions(session, data)
            elif test_type == 'note_reproduction':
                self._create_note_questions(session, data)
                
            session_serializer = TestSessionSerializer(session)
            return Response(session_serializer.data, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            session.delete()
            return Response(
                {'error': f'Помилка створення тесту: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _create_interval_questions(self, session, data):
        """Створення питань для тесту розпізнавання інтервалів"""
        available_intervals = data.get('intervals', [
            'minor_second', 'major_second', 'minor_third', 'major_third',
            'perfect_fourth', 'tritone', 'perfect_fifth', 'minor_sixth',
            'major_sixth', 'minor_seventh', 'major_seventh', 'perfect_octave'
        ])
        
        if not available_intervals:
            available_intervals = ['major_third', 'perfect_fourth', 'perfect_fifth']
        
        base_notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        octaves = [3, 4, 5]
        
        created_questions = []
        
        for i in range(session.total_questions):
            interval_type = random.choice(available_intervals)
            base_note = random.choice(base_notes)
            octave = random.choice(octaves)
            base_note_with_octave = f"{base_note}{octave}"
            
            target_note = self._calculate_target_note(base_note, interval_type)
            target_note_with_octave = f"{target_note}{octave}"
            
            question = TestQuestion.objects.create(
                session=session,
                question_number=i + 1,
                interval_type=interval_type,
                base_note=base_note_with_octave,
                target_note=target_note_with_octave,
                harmonic_audio_url="",
                melodic_audio_url=""
            )
            created_questions.append(question)
        
        self._generate_test_audio(session)
        
        return created_questions
    
    def _create_note_questions(self, session, data):
        """Створення питань для тесту відтворення нот"""
        user_profile, created = UserProfile.objects.get_or_create(user=session.user)
        
        if user_profile.vocal_range_min_frequency and user_profile.vocal_range_max_frequency:
            min_freq = user_profile.vocal_range_min_frequency
            max_freq = user_profile.vocal_range_max_frequency
        else:
            min_freq = 130.81  # C3
            max_freq = 523.25  # C5
        
        for i in range(session.total_questions):
            target_frequency = self._generate_random_frequency(min_freq, max_freq)
            target_note = self._frequency_to_note(target_frequency)
            
            TestQuestion.objects.create(
                session=session,
                question_number=i + 1,
                target_frequency=target_frequency,
                target_note_name=target_note,
                reference_audio_url=f"/api/testing/audio/placeholder_note_{i+1}.wav",
                frequency_tolerance=20.0
            )
    
    def _calculate_target_note(self, base_note, interval_type):
        """Calculate target note based on base note and interval type."""
        semitones = INTERVALS_SEMITONES.get(interval_type, 0)
        return get_target_note(base_note, semitones)
    
    def _generate_random_frequency(self, min_freq, max_freq):
        """Генерація випадкової частоти, яка відповідає музичній ноті"""
        frequencies = []
        current_freq = min_freq
        
        while current_freq <= max_freq:
            frequencies.append(current_freq)
            current_freq *= 2**(1/12)
        
        return random.choice(frequencies)
    
    def _frequency_to_note(self, freq):
        """Convert frequency to note name with octave."""
        return frequency_to_note(freq)
    
    def _generate_test_audio(self, session, instrument: str = 'piano'):
        """
        Generate audio files for test questions.
        
        Uses Factory Method pattern for instrument selection.
        
        Args:
            session: TestSession instance
            instrument: Instrument type ('piano', 'guitar', 'synth')
        """
        audio_dir = os.path.join(settings.MEDIA_ROOT, 'testing_audio')
        os.makedirs(audio_dir, exist_ok=True)
        
        user_audio_dir = os.path.join(audio_dir, str(session.user.id))
        if os.path.exists(user_audio_dir):
            shutil.rmtree(user_audio_dir)
        os.makedirs(user_audio_dir, exist_ok=True)
        
        # Create generator using Factory Method
        factory = InstrumentFactory()
        generator = factory.create_generator(instrument)
        
        for question in session.questions.all():
            if session.test_type == 'interval_recognition':
                # Extract note name (without octave) from question
                base_note_name = question.base_note[:-1] if question.base_note[-1].isdigit() else question.base_note
                target_note_name = question.target_note[:-1] if question.target_note[-1].isdigit() else question.target_note
                
                # Get frequencies
                base_freq = get_note_frequency(base_note_name)
                target_freq = get_note_frequency(target_note_name)
                
                # Generate tones using factory-created generator
                base_tone = generator.generate_tone(base_freq)
                target_tone = generator.generate_tone(target_freq)
                
                # Combine tones
                harmonic_audio = combine_tones_harmonic(base_tone, target_tone)
                melodic_audio = combine_tones_melodic(base_tone, target_tone)
                
                # Save audio files
                harmonic_filename = f"question_{question.id}_harmonic.wav"
                melodic_filename = f"question_{question.id}_melodic.wav"
                
                harmonic_path = os.path.join(user_audio_dir, harmonic_filename)
                melodic_path = os.path.join(user_audio_dir, melodic_filename)
                
                save_audio(harmonic_audio, harmonic_path)
                save_audio(melodic_audio, melodic_path)
                
                question.harmonic_audio_url = f"/api/testing/audio/{session.user.id}/{harmonic_filename}"
                question.melodic_audio_url = f"/api/testing/audio/{session.user.id}/{melodic_filename}"
                question.save()


class SubmitAnswerView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        serializer = SubmitAnswerSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        question_id = data['question_id']
        
        try:
            question = TestQuestion.objects.get(
                id=question_id,
                session__user=request.user,
                session__is_completed=False
            )
        except TestQuestion.DoesNotExist:
            return Response(
                {'error': 'Питання не знайдено'}, 
                status=status.HTTP_404_NOT_FOUND
            )
        
        if question.session.test_type == 'interval_recognition':
            question.user_answer = data.get('answer')
            question.is_correct = (question.user_answer == question.interval_type)
        
        elif question.session.test_type == 'note_reproduction':
            recorded_freq = data.get('recorded_frequency')
            question.is_correct = question.check_frequency_answer(recorded_freq)
        
        question.answered_at = datetime.now()
        question.save()
        
        session = question.session
        answered_questions = session.questions.filter(answered_at__isnull=False).count()
        
        if answered_questions == session.total_questions:
            self._complete_session(session)
        
        return Response({
            'is_correct': question.is_correct,
            'session_completed': session.is_completed
        })
    
    def _complete_session(self, session):
        """Завершення сесії тестування"""
        session.correct_answers = session.questions.filter(is_correct=True).count()
        session.calculate_accuracy()
        session.calculate_experience()
        session.completed_at = datetime.now()
        session.is_completed = True
        session.save()
        
        profile, created = UserProfile.objects.get_or_create(user=session.user)
        level_up = profile.add_experience(session.experience_gained)
        
        self._check_achievements(session.user, session, level_up)
    
    def _check_achievements(self, user, session, level_up):
        """Перевірка та нарахування досягнень"""
        profile = user.profile
        
        if level_up:
            level_achievements = Achievement.objects.filter(
                achievement_type='level',
                requirement_value=profile.level
            )
            for achievement in level_achievements:
                UserAchievement.objects.get_or_create(
                    user=user,
                    achievement=achievement
                )
        
        if session.accuracy_percentage >= 90:
            accuracy_achievements = Achievement.objects.filter(
                achievement_type='accuracy',
                requirement_value__lte=session.accuracy_percentage
            )
            for achievement in accuracy_achievements:
                UserAchievement.objects.get_or_create(
                    user=user,
                    achievement=achievement
                )


class GenerateSingleNoteView(APIView):
    """
    Generate a single note audio file.
    
    Uses Factory Method pattern via InstrumentFactory.
    """
    permission_classes = [IsAuthenticated]
    
    SUPPORTED_INSTRUMENTS = ['piano', 'guitar', 'synth']
    DEFAULT_INSTRUMENT = 'piano'
    
    def __init__(self):
        super().__init__()
        self.audio_dir = os.path.join(settings.MEDIA_ROOT, 'testing_audio')
        os.makedirs(self.audio_dir, exist_ok=True)
        self.factory = InstrumentFactory()

    def post(self, request):
        """Generate audio for a single note"""
        try:
            data = request.data
            note = data.get('note', 'A')
            octave = data.get('octave', 4)
            duration = data.get('duration', 3.0)
            instrument = data.get('instrument', self.DEFAULT_INSTRUMENT)
            
            # Validate note
            if note not in NOTE_FREQUENCIES:
                return Response(
                    {'error': 'Невірна нота'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Validate octave
            if not (0 <= octave <= 8):
                return Response(
                    {'error': 'Невірна октава (0-8)'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Validate instrument
            if instrument not in self.SUPPORTED_INSTRUMENTS:
                return Response(
                    {'error': f'Непідтримуваний інструмент. Доступні: {", ".join(self.SUPPORTED_INSTRUMENTS)}'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Prepare user audio directory
            user_audio_dir = os.path.join(self.audio_dir, str(request.user.id))
            if os.path.exists(user_audio_dir):
                shutil.rmtree(user_audio_dir)
            os.makedirs(user_audio_dir, exist_ok=True)
            
            # Get frequency using module function
            frequency = get_note_frequency(note, octave)
            
            # Create generator using Factory Method
            generator = self.factory.create_generator(instrument)
            
            # Generate audio
            audio_data = generator.generate_tone(frequency, duration)
            
            # Save audio file
            note_id = str(uuid.uuid4())
            filename = f"{note_id}_{note}{octave}.wav"
            file_path = os.path.join(user_audio_dir, filename)
            save_audio(audio_data, file_path)
            
            audio_url = f"/api/testing/audio/{request.user.id}/{filename}"
            
            return Response({
                'note_id': note_id,
                'note': note,
                'octave': octave,
                'frequency': frequency,
                'duration': duration,
                'instrument': instrument,
                'audio_url': audio_url,
                'note_name': f"{note}{octave}"
            })
            
        except Exception as e:
            return Response(
                {'error': f'Помилка генерації: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class TestingSecureAudioView(APIView):
    """Безпечна віддача аудіофайлів для тестування"""
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    
    def get(self, request, user_id, filename):
        if str(request.user.id) != str(user_id):
            return HttpResponse("Доступ заборонено", status=403)
        
        audio_dir = os.path.join(settings.MEDIA_ROOT, 'testing_audio')
        file_path = os.path.join(audio_dir, user_id, filename)
        
        allowed_dir = os.path.join(audio_dir, user_id)
        try:
            real_file_path = os.path.realpath(file_path)
            real_allowed_dir = os.path.realpath(allowed_dir)
            
            if not real_file_path.startswith(real_allowed_dir):
                return HttpResponse("Доступ заборонено", status=403)
                
        except Exception:
            return HttpResponse("Помилка доступу", status=400)
        
        if not os.path.exists(file_path):
            return HttpResponse("Файл не знайдено", status=404)
        
        if not filename.endswith('.wav'):
            return HttpResponse("Непідтримуваний тип файлу", status=400)
        
        try:
            file_size = os.path.getsize(file_path)
            content_type = 'audio/wav'
            
            range_header = request.META.get('HTTP_RANGE')
            
            if range_header:
                range_match = range_header.replace('bytes=', '').split('-')
                start = int(range_match[0]) if range_match[0] else 0
                end = int(range_match[1]) if range_match[1] else file_size - 1
                
                with open(file_path, 'rb') as f:
                    f.seek(start)
                    data = f.read(end - start + 1)
                
                response = HttpResponse(
                    data,
                    status=206,
                    content_type=content_type
                )
                response['Content-Range'] = f'bytes {start}-{end}/{file_size}'
                response['Accept-Ranges'] = 'bytes'
                response['Content-Length'] = str(end - start + 1)
            else:
                response = FileResponse(
                    open(file_path, 'rb'),
                    content_type=content_type,
                    filename=filename
                )
                response['Content-Length'] = str(file_size)
                response['Accept-Ranges'] = 'bytes'
            
            response['Cache-Control'] = 'private, max-age=300'
            response['X-Content-Type-Options'] = 'nosniff'
            
            return response
            
        except Exception as e:
            print(f"Error serving testing file: {e}")
            return HttpResponse("Помилка при видачі файлу", status=500)


class ClearTestingAudioView(APIView):
    """Очищення аудіофайлів тестування для користувача"""
    permission_classes = [IsAuthenticated]
    
    def delete(self, request):
        try:
            user_audio_dir = os.path.join(
                settings.MEDIA_ROOT, 
                'testing_audio', 
                str(request.user.id)
            )
            
            if os.path.exists(user_audio_dir):
                shutil.rmtree(user_audio_dir)
            
            return Response({'message': 'Файли тестування успішно видалено'})
            
        except Exception as e:
            return Response(
                {'error': f'Помилка видалення файлів: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )