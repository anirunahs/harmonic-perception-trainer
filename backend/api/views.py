from django.shortcuts import render
from django.contrib.auth.models import User
from rest_framework import generics
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import UserSerializer
from rest_framework.permissions import IsAuthenticated, AllowAny
from django.conf import settings
from django.http import FileResponse, Http404
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
        # Частоти нот в герцах (4-та октава)
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
        
        # Визначення інтервалів (півтони)
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
        
        # Створюємо директорію для аудіофайлів
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
        
        # Основна частота
        wave = np.sin(2 * np.pi * frequency * t)
        
        # Додаємо гармоніки для більш реалістичного звучання рояля
        harmonics = [
            (2, 0.3),    # Друга гармоніка
            (3, 0.2),    # Третя гармоніка
            (4, 0.1),    # Четверта гармоніка
            (5, 0.05),   # П'ята гармоніка
        ]
        
        for harmonic, amplitude in harmonics:
            wave += amplitude * np.sin(2 * np.pi * frequency * harmonic * t)
        
        # Додаємо envelope (ADSR) для більш природного звучання
        attack_time = 0.1
        decay_time = 0.3
        sustain_level = 0.7
        release_time = 0.5
        
        envelope = np.ones_like(t)
        
        # Attack
        attack_samples = int(attack_time * sample_rate)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay
        decay_samples = int(decay_time * sample_rate)
        if decay_samples > 0 and attack_samples + decay_samples < len(envelope):
            envelope[attack_samples:attack_samples + decay_samples] = np.linspace(1, sustain_level, decay_samples)
        
        # Sustain
        sustain_start = attack_samples + decay_samples
        release_start = len(envelope) - int(release_time * sample_rate)
        if sustain_start < release_start:
            envelope[sustain_start:release_start] = sustain_level
        
        # Release
        release_samples = int(release_time * sample_rate)
        if release_samples > 0:
            envelope[-release_samples:] = np.linspace(sustain_level, 0, release_samples)
        
        # Застосовуємо envelope
        wave *= envelope
        
        # Нормалізуємо амплітуду
        wave = wave / np.max(np.abs(wave)) * 0.7
        
        return wave

    def save_audio(self, audio_data, filename, sample_rate=44100):
        """Зберегти аудіо дані у файл"""
        filepath = os.path.join(self.audio_dir, filename)
        
        # Конвертуємо у 16-bit integer
        audio_int16 = np.int16(audio_data * 32767)
        
        wavfile.write(filepath, sample_rate, audio_int16)
        return filepath

    def get_target_note(self, base_note, semitones):
        """Отримати ноту через вказану кількість півтонів"""
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
            
            # Очищуємо старі файли користувача
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
                
                # Генеруємо частоти
                base_freq = self.get_note_frequency(base_note)
                target_freq = self.get_note_frequency(base_note, semitones)
                
                # Генеруємо тони
                base_tone = self.generate_piano_tone(base_freq)
                target_tone = self.generate_piano_tone(target_freq)
                
                # Створюємо гармонічний інтервал (одночасно)
                harmonic_audio = (base_tone + target_tone) / 2
                
                # Створюємо мелодичний інтервал (послідовно)
                melodic_audio = np.concatenate([base_tone, target_tone])
                
                # Генеруємо унікальні імена файлів
                interval_id = str(uuid.uuid4())
                harmonic_filename = f"{interval_id}_harmonic.wav"
                melodic_filename = f"{interval_id}_melodic.wav"
                
                # Зберігаємо файли
                harmonic_path = self.save_audio(
                    harmonic_audio, 
                    os.path.join(str(request.user.id), harmonic_filename)
                )
                melodic_path = self.save_audio(
                    melodic_audio, 
                    os.path.join(str(request.user.id), melodic_filename)
                )
                
                # Формуємо URL для доступу до файлів
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

