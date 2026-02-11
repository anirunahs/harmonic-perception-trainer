"""
Experiments API — доступ лише для ролі experimenter.
Експеримент "Обертони інтервалів": семпли фортепіано (Salamander), спектри, аудіо.
"""
import base64
import io
import urllib.request
import numpy as np
from scipy.io import wavfile
from scipy import signal as scipy_signal
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
import librosa

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.audio import (
    DEFAULT_SAMPLE_RATE,
    INTERVALS_SEMITONES,
    get_note_frequency,
    get_frequency_with_semitone_offset,
)

from .permissions import IsExperimenter

SALAMANDER_BASE = "https://tonejs.github.io/audio/salamander/"
SAMPLE_NOTE = "C4"


INTERVAL_NAMES_UK = {
    'minor_second': 'Мала секунда',
    'major_second': 'Велика секунда',
    'minor_third': 'Мала терція',
    'major_third': 'Велика терція',
    'perfect_fourth': 'Чиста кварта',
    'tritone': 'Тритон',
    'perfect_fifth': 'Чиста квінта',
    'minor_sixth': 'Мала секста',
    'major_sixth': 'Велика секста',
    'minor_seventh': 'Мала септима',
    'major_seventh': 'Велика септима',
    'perfect_octave': 'Чиста октава',
}


def _compute_spectrum(wave: np.ndarray, sample_rate: int, max_freq: float = 4000.0, n_bins: int = 512):
    """
    FFT спектр: частоти та величини (амплітуди).
    Повертає списки frequencies, magnitudes обмежені max_freq і зменшеною кількістю бінів.
    """
    n = len(wave)
    if n == 0:
        return [], []
    # Вікно Ханна для зменшення витоків
    window = np.hanning(n)
    windowed = wave * window
    fft_vals = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    magnitudes = np.abs(fft_vals) / n

    # Обрізати по max_freq
    mask = freqs <= max_freq
    freqs = freqs[mask]
    magnitudes = magnitudes[mask]

    # Зменшити кількість точок (усреднення по бінах)
    if len(freqs) <= n_bins:
        return freqs.tolist(), magnitudes.tolist()
    step = len(freqs) // n_bins
    idx = np.arange(0, len(freqs), step)[:n_bins]
    f_reduced = freqs[idx].tolist()
    m_reduced = magnitudes[idx].tolist()
    return f_reduced, m_reduced


def _wav_to_base64(wave: np.ndarray, sample_rate: int) -> str:
    """Перетворити хвилю в WAV base64."""
    wave_int16 = (np.clip(wave, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, wave_int16)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _load_piano_sample(sr: int = DEFAULT_SAMPLE_RATE):
    """Завантажити семпл фортепіано C4 з Salamander (ті самі, що й у фронті)."""
    url = f"{SALAMANDER_BASE}{SAMPLE_NOTE}.mp3"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = resp.read()
    y, _ = librosa.load(io.BytesIO(data), sr=sr, mono=True)
    return y


def _piano_interval_from_sample(semitones: int, duration: float, sr: int = DEFAULT_SAMPLE_RATE):
    """
    Дві ноти з одного семплу: базова C4, верхня — pitch_shift на semitones.
    Повертає (root_wave, upper_wave) обрізані по duration.
    """
    root = _load_piano_sample(sr)
    n_samples = int(duration * sr)
    root = root[:n_samples] if len(root) >= n_samples else np.pad(root, (0, n_samples - len(root)))
    
    upper = librosa.effects.pitch_shift(root, sr=sr, n_steps=semitones)
    upper = upper[:n_samples] if len(upper) >= n_samples else np.pad(upper, (0, n_samples - len(upper)))
    return root, upper


SPECTROGRAM_GRID_INTERVALS = [
    'minor_second', 'major_second', 'minor_third', 'major_third',
    'perfect_fourth', 'tritone', 'perfect_fifth', 'minor_sixth',
    'major_sixth', 'minor_seventh', 'major_seventh', 'perfect_octave',
]


def _render_spectrogram_grid(duration: float = 1.5, sr: int = DEFAULT_SAMPLE_RATE, max_freq_hz: float = 2000.0):
    """
    Малює сітку 3×4 (3 рядки, 4 стовпці) спектрограм для 12 інтервалів. Повертає PNG bytes.
    Велике FFT-вікно (4096) — чіткі горизонтальні лінії обертонів; обмежений діапазон дБ — кращий контраст.
    """
    n_per_row = 4
    n_rows = 3
    fig, axes = plt.subplots(n_rows, n_per_row, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    nperseg = 4096
    noverlap = nperseg // 2
    for idx, interval_key in enumerate(SPECTROGRAM_GRID_INTERVALS):
        if idx >= len(axes):
            break
        semitones = INTERVALS_SEMITONES.get(interval_key, 7)
        root_wave, upper_wave = _piano_interval_from_sample(semitones, duration, sr)
        combined = root_wave + upper_wave
        combined = combined / (np.max(np.abs(combined)) + 1e-8) * 0.8
        f, t, Sxx = scipy_signal.spectrogram(combined, sr, nperseg=nperseg, noverlap=noverlap)
        mask = f <= max_freq_hz
        f, Sxx = f[mask], Sxx[mask, :]
        Sxx_db = 10 * np.log10(Sxx + 1e-12)
        db_max = np.nanmax(Sxx_db)
        db_min = max(db_max - 50, np.nanmin(Sxx_db))
        ax = axes[idx]
        ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis', vmin=db_min, vmax=db_max)
        ax.set_ylim(0, max_freq_hz)
        ax.set_title(INTERVAL_NAMES_UK.get(interval_key, interval_key), fontsize=10)
        ax.set_ylabel('')
        ax.set_xlabel('')
    for ax in axes:
        ax.label_outer()
    plt.suptitle('Спектрограми інтервалів (базова C4 + верхня нота). Горизонтальні смуги — основна частота та обертони.', fontsize=11, y=1.02)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


class OvertonesExperimentView(APIView):
    """
    Експеримент "Обертони інтервалів": семпли фортепіано (Salamander), спектри, аудіо.
    GET/POST: base_note (C), interval (назва або semitones 1–12), duration (сек).
    """
    permission_classes = [IsAuthenticated, IsExperimenter]

    def get(self, request):
        base_note = request.query_params.get('base_note', 'C')
        interval = request.query_params.get('interval', 'perfect_fifth')
        duration = float(request.query_params.get('duration', 2.0))
        return self._compute(base_note, interval, duration)

    def post(self, request):
        data = request.data or {}
        audio_combined_b64 = data.get('audio_combined_base64')
        audio_root_b64 = data.get('audio_root_base64')
        audio_upper_b64 = data.get('audio_upper_base64')
        client_sr = data.get('sample_rate') or DEFAULT_SAMPLE_RATE
        if audio_combined_b64 and audio_root_b64 and audio_upper_b64:
            return self._compute_from_client_audio(
                data.get('interval', 'perfect_fifth'),
                audio_root_b64, audio_upper_b64, audio_combined_b64, client_sr
            )
        base_note = data.get('base_note', 'C')
        interval = data.get('interval', 'perfect_fifth')
        duration = float(data.get('duration', 2.0))
        return self._compute(base_note, interval, duration)

    def _compute_from_client_audio(self, interval: str, root_b64: str, upper_b64: str, combined_b64: str, sr: int):
        """Спектри з аудіо, згенерованого на клієнті (Tone.js + семпли)."""
        def decode_wav(b64):
            raw = base64.b64decode(b64)
            sr_wav, arr = wavfile.read(io.BytesIO(raw))
            if arr.dtype == np.int16:
                arr = arr.astype(np.float32) / 32768.0
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            return arr, sr_wav
        try:
            root_wave, sr_root = decode_wav(root_b64)
            upper_wave, sr_upper = decode_wav(upper_b64)
            combined_wave, sr_comb = decode_wav(combined_b64)
        except Exception as e:
            return Response({'detail': f'Invalid audio data: {e}'}, status=status.HTTP_400_BAD_REQUEST)
        use_sr = int(sr_comb or sr or DEFAULT_SAMPLE_RATE)
        root_freqs, root_mags = _compute_spectrum(root_wave, use_sr)
        upper_freqs, upper_mags = _compute_spectrum(upper_wave, use_sr)
        comb_freqs, comb_mags = _compute_spectrum(combined_wave, use_sr)
        semitones = INTERVALS_SEMITONES.get(interval, 7)
        interval_key = interval if interval in INTERVALS_SEMITONES else 'perfect_fifth'
        for k, v in INTERVALS_SEMITONES.items():
            if v == semitones:
                interval_key = k
                break
        root_freq = get_note_frequency('C', octave=4)
        upper_freq = get_frequency_with_semitone_offset('C', semitones, octave=4)
        return Response({
            'interval_key': interval_key,
            'interval_name_uk': INTERVAL_NAMES_UK.get(interval_key, interval_key),
            'semitones': semitones,
            'base_note': 'C',
            'root_freq_hz': round(root_freq, 2),
            'upper_freq_hz': round(upper_freq, 2),
            'duration_sec': 0,
            'sample_rate': use_sr,
            'combined_spectrum': {'frequencies': comb_freqs, 'magnitudes': comb_mags},
            'root_spectrum': {'frequencies': root_freqs, 'magnitudes': root_mags},
            'upper_spectrum': {'frequencies': upper_freqs, 'magnitudes': upper_mags},
            'audio_base64': None,
        }, status=status.HTTP_200_OK)

    def _compute(self, base_note: str, interval: str, duration: float):
        # Визначити кількість півтонів та ключ інтервалу
        semitones = None
        interval_key = None
        if isinstance(interval, str) and interval.isdigit():
            semitones = int(interval)
        elif isinstance(interval, str) and interval in INTERVALS_SEMITONES:
            semitones = INTERVALS_SEMITONES[interval]
            interval_key = interval
        if interval_key is None and semitones is not None:
            for k, v in INTERVALS_SEMITONES.items():
                if v == semitones:
                    interval_key = k
                    break
        if semitones is None:
            semitones = 7
            interval_key = 'perfect_fifth'
        if interval_key is None:
            interval_key = 'perfect_fifth'

        root_freq = get_note_frequency(base_note, octave=4)
        upper_freq = get_frequency_with_semitone_offset(base_note, semitones, octave=4)

        # Семпли фортепіано Salamander
        root_wave, upper_wave = _piano_interval_from_sample(semitones, duration, sr=DEFAULT_SAMPLE_RATE)
        combined = root_wave + upper_wave
        combined = combined / (np.max(np.abs(combined)) + 1e-8) * 0.8

        root_freqs, root_mags = _compute_spectrum(root_wave, DEFAULT_SAMPLE_RATE)
        upper_freqs, upper_mags = _compute_spectrum(upper_wave, DEFAULT_SAMPLE_RATE)
        comb_freqs, comb_mags = _compute_spectrum(combined, DEFAULT_SAMPLE_RATE)

        interval_name_uk = INTERVAL_NAMES_UK.get(interval_key, interval_key)
        audio_base64 = _wav_to_base64(combined, DEFAULT_SAMPLE_RATE)

        return Response({
            'interval_key': interval_key,
            'interval_name_uk': interval_name_uk,
            'semitones': semitones,
            'base_note': base_note,
            'root_freq_hz': round(root_freq, 2),
            'upper_freq_hz': round(upper_freq, 2),
            'duration_sec': duration,
            'sample_rate': DEFAULT_SAMPLE_RATE,
            'combined_spectrum': {'frequencies': comb_freqs, 'magnitudes': comb_mags},
            'root_spectrum': {'frequencies': root_freqs, 'magnitudes': root_mags},
            'upper_spectrum': {'frequencies': upper_freqs, 'magnitudes': upper_mags},
            'audio_base64': audio_base64,
        }, status=status.HTTP_200_OK)


class SpectrogramGridView(APIView):
    """
    Один знімок: сітка 4×3 спектрограм для 12 інтервалів.
    GET/POST без параметрів. Повертає { image_base64: "..." }.
    """
    permission_classes = [IsAuthenticated, IsExperimenter]

    def get(self, request):
        return self._render()

    def post(self, request):
        return self._render()

    def _render(self):
        try:
            png_bytes = _render_spectrogram_grid(duration=1.5, sr=DEFAULT_SAMPLE_RATE, max_freq_hz=2000.0)
            b64 = base64.b64encode(png_bytes).decode('utf-8')
            return Response({'image_base64': b64}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'detail': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ExperimentsListView(APIView):
    """Список доступних експериментів (для меню)."""
    permission_classes = [IsAuthenticated, IsExperimenter]

    def get(self, request):
        return Response({
            'experiments': [
                {
                    'id': 'overtones',
                    'name_uk': 'Обертони інтервалів',
                    'description_uk': 'Семпли фортепіано (Salamander) та візуалізація спектрів обертонів.',
                    'path': '/experiments/overtones',
                },
            ],
        }, status=status.HTTP_200_OK)
