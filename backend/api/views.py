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

# Import testing module (Builder, Observer patterns, and Service)
from core.testing import (
    TestSessionService,
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
    """
    Create test session using Builder pattern.
    
    Pattern: Builder (Creational)
    Uses TestSessionBuilder for step-by-step construction of test sessions.
    """
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        serializer = CreateTestSessionSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        
        try:
            # Use Service layer for session creation
            # Extract test_type separately to avoid duplicate keyword argument
            test_type = data.pop('test_type')
            session = TestSessionService.create_session(
                user=request.user,
                test_type=test_type,
                **data
            )
            
            session_serializer = TestSessionSerializer(session)
            return Response(session_serializer.data, status=status.HTTP_201_CREATED)
            
        except ValueError as e:
            return Response(
                {'error': f'Помилка валідації: {str(e)}'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return Response(
                {'error': f'Помилка створення тесту: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


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
        
        try:
            # Use Service layer for submitting answer
            result = TestSessionService.submit_answer(
                session=question.session,
                question=question,
                answer=data.get('answer'),
                response_time=data.get('response_time')
            )
            
            return Response({
                'is_correct': result['is_correct'],
                'session_completed': result['session_completed'],
                'answered_count': result['answered_count'],
                'total_questions': result['total_questions']
            })
        except ValueError as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
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