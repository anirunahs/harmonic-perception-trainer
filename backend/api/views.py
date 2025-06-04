from django.shortcuts import render
from django.contrib.auth.models import User
from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.decorators import authentication_classes, permission_classes
from .serializers import UserSerializer
from rest_framework.permissions import IsAuthenticated, AllowAny
from django.conf import settings
from django.http import FileResponse, Http404, HttpResponse
from django.views import View
import os
import uuid
import numpy as np
from scipy.io import wavfile
import shutil

class CreateUserView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [AllowAny]

class GenerateIntervalsView(APIView):
    permission_classes = [IsAuthenticated]
    
    def __init__(self):
        super().__init__()

        self.note_frequencies = {
            'C': 261.63,
            'C#': 277.18,
            'D': 293.66,
            'D#': 311.13,
            'E': 329.63,
            'F': 349.23,
            'F#': 369.99,
            'G': 392.00,
            'G#': 415.30,
            'A': 440.00,
            'A#': 466.16,
            'B': 493.88
        }
        
        self.intervals_semitones = {
            'minor_second': 1,
            'major_second': 2,
            'minor_third': 3,
            'major_third': 4,
            'perfect_fourth': 5,
            'tritone': 6,
            'perfect_fifth': 7,
            'minor_sixth': 8,
            'major_sixth': 9,
            'minor_seventh': 10,
            'major_seventh': 11,
            'perfect_octave': 12
        }
        
        self.audio_dir = os.path.join(settings.MEDIA_ROOT, 'training_audio')
        os.makedirs(self.audio_dir, exist_ok=True)

    def get_note_frequency(self, note, semitone_offset=0):
        """Отримати частоту ноти з урахуванням зміщення в півтонах"""
        base_freq = self.note_frequencies.get(note, 440.0)
        # Кожен півтон - це множення на 2^(1/12)
        return base_freq * (2 ** (semitone_offset / 12))

    def generate_piano_tone(self, frequency, duration=2.0, sample_rate=44100):
        """Генерація тону з тембром, схожим на рояль"""
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        wave = np.sin(2 * np.pi * frequency * t)
        
        harmonics = [
            (2, 0.3),
            (3, 0.2),
            (4, 0.1),
            (5, 0.05),
        ]
        
        for harmonic, amplitude in harmonics:
            wave += amplitude * np.sin(2 * np.pi * frequency * harmonic * t)
        
        attack_time = 0.1
        decay_time = 0.3
        sustain_level = 0.7
        release_time = 0.5
        
        envelope = np.ones_like(t)
        
        attack_samples = int(attack_time * sample_rate)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        decay_samples = int(decay_time * sample_rate)
        if decay_samples > 0 and attack_samples + decay_samples < len(envelope):
            envelope[attack_samples:attack_samples + decay_samples] = np.linspace(1, sustain_level, decay_samples)
        
        sustain_start = attack_samples + decay_samples
        release_start = len(envelope) - int(release_time * sample_rate)
        if sustain_start < release_start:
            envelope[sustain_start:release_start] = sustain_level
        
        release_samples = int(release_time * sample_rate)
        if release_samples > 0:
            envelope[-release_samples:] = np.linspace(sustain_level, 0, release_samples)
        
        wave *= envelope
        
        wave = wave / np.max(np.abs(wave)) * 0.7
        
        return wave

    def save_audio(self, audio_data, filename, sample_rate=44100):
        filepath = os.path.join(self.audio_dir, filename)
        audio_int16 = np.int16(audio_data * 32767)
        
        wavfile.write(filepath, sample_rate, audio_int16)
        return filepath

    def get_target_note(self, base_note, semitones):
        notes = list(self.note_frequencies.keys())
        base_index = notes.index(base_note)
        target_index = (base_index + semitones) % 12
        return notes[target_index]

    def post(self, request):
        try:
            data = request.data
            intervals = data.get('intervals', [])
            base_note = data.get('base_note', 'C')
            
            if not intervals:
                return Response(
                    {'error': 'Не вказано інтервали'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            if base_note not in self.note_frequencies:
                return Response(
                    {'error': 'Невірна базова нота'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            user_audio_dir = os.path.join(self.audio_dir, str(request.user.id))
            if os.path.exists(user_audio_dir):
                shutil.rmtree(user_audio_dir)
            os.makedirs(user_audio_dir, exist_ok=True)
            
            generated_intervals = []
            
            for interval_type in intervals:
                if interval_type not in self.intervals_semitones:
                    continue
                
                semitones = self.intervals_semitones[interval_type]
                target_note = self.get_target_note(base_note, semitones)
                
                base_freq = self.get_note_frequency(base_note)
                target_freq = self.get_note_frequency(base_note, semitones)
                
                base_tone = self.generate_piano_tone(base_freq)
                target_tone = self.generate_piano_tone(target_freq)
                
                harmonic_audio = (base_tone + target_tone) / 2
                melodic_audio = np.concatenate([base_tone, target_tone])
                
                interval_id = str(uuid.uuid4())
                harmonic_filename = f"{interval_id}_harmonic.wav"
                melodic_filename = f"{interval_id}_melodic.wav"
                
                harmonic_path = self.save_audio(
                    harmonic_audio, 
                    os.path.join(str(request.user.id), harmonic_filename)
                )
                melodic_path = self.save_audio(
                    melodic_audio, 
                    os.path.join(str(request.user.id), melodic_filename)
                )
                
                harmonic_url = f"/api/training/audio/{request.user.id}/{harmonic_filename}"
                melodic_url = f"/api/training/audio/{request.user.id}/{melodic_filename}"
                
                generated_intervals.append({
                    'id': interval_id,
                    'interval_type': interval_type,
                    'base_note': base_note,
                    'target_note': target_note,
                    'semitones': semitones,
                    'harmonic_url': harmonic_url,
                    'melodic_url': melodic_url
                })
            
            return Response({
                'intervals': generated_intervals,
                'base_note': base_note,
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