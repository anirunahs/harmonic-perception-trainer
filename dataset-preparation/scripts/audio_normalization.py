import librosa
import soundfile as sf
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter

class AudioNormalizer:
    def __init__(self, sr=44100, target_lufs=-23.0):
        self.sr = sr
        self.target_lufs = target_lufs
        self.target_rms_db = -18.0
        self.max_peak_db = -3.0
        
        self.processed_count = 0
        self.analysis_data = []
    
    def calculate_lufs(self, audio):
        """Розрахунок LUFS (стандарт гучності)"""
        try:
            import pyloudnorm as pyln
            meter = pyln.Meter(self.sr)
            loudness = meter.integrated_loudness(audio)
            return loudness
        except ImportError:
            return None
    
    def calculate_audio_metrics(self, audio):
        """Розрахунок всіх метрик аудіо"""
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        
        rms_db = 20 * np.log10(rms + 1e-10)
        peak_db = 20 * np.log10(peak + 1e-10)
        
        crest_factor = peak / (rms + 1e-10)
        dynamic_range = peak_db - rms_db
        
        # THD+N наближення
        fundamental_freq = self.estimate_fundamental(audio)
        thd = self.estimate_thd(audio, fundamental_freq) if fundamental_freq else 0
        
        lufs = self.calculate_lufs(audio)
        
        return {
            'rms': rms,
            'peak': peak,
            'rms_db': rms_db,
            'peak_db': peak_db,
            'crest_factor': crest_factor,
            'dynamic_range': dynamic_range,
            'thd': thd,
            'lufs': lufs,
            'duration': len(audio) / self.sr
        }
    
    def estimate_fundamental(self, audio):
        """Оцінка основної частоти"""
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sr)
            pitches = pitches[magnitudes > np.percentile(magnitudes, 85)]
            if len(pitches) > 0:
                return np.median(pitches[pitches > 0])
        except:
            pass
        return None
    
    def estimate_thd(self, audio, fundamental_freq):
        """Оцінка коефіцієнта гармонічних спотворень"""
        if not fundamental_freq:
            return 0
        
        try:
            fft = np.fft.fft(audio)
            freqs = np.fft.fftfreq(len(audio), 1/self.sr)
            
            fundamental_bin = np.argmin(np.abs(freqs - fundamental_freq))
            fundamental_magnitude = np.abs(fft[fundamental_bin])
            
            harmonic_magnitudes = []
            for harmonic in range(2, 6):
                harmonic_freq = fundamental_freq * harmonic
                if harmonic_freq < self.sr / 2:
                    harmonic_bin = np.argmin(np.abs(freqs - harmonic_freq))
                    harmonic_magnitudes.append(np.abs(fft[harmonic_bin]))
            
            if harmonic_magnitudes:
                thd = np.sqrt(sum(h**2 for h in harmonic_magnitudes)) / fundamental_magnitude
                return min(thd, 1.0)
        except:
            pass
        return 0
    
    def apply_preemphasis(self, audio, alpha=0.97):
        """Пре-емфазис для підвищення високих частот"""
        return np.append(audio[0], audio[1:] - alpha * audio[:-1])
    
    def apply_highpass_filter(self, audio, cutoff=80, order=4):
        """High-pass фільтр"""
        nyquist = self.sr / 2
        normal_cutoff = cutoff / nyquist
        
        if normal_cutoff >= 1.0:
            return audio
        
        try:
            b, a = butter(order, normal_cutoff, btype='high', analog=False)
            return filtfilt(b, a, audio)
        except:
            return audio
    
    def apply_lowpass_filter(self, audio, cutoff=8000, order=4):
        """Low-pass фільтр"""
        nyquist = self.sr / 2
        normal_cutoff = cutoff / nyquist
        
        if normal_cutoff >= 1.0:
            return audio
        
        try:
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return filtfilt(b, a, audio)
        except:
            return audio
    
    def apply_spectral_gating(self, audio, gate_threshold=-70):
        """Спектральний гейт для зменшення шуму"""
        frame_length = 2048
        hop_length = 512
        
        stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        gate_mask = magnitude_db > gate_threshold
        gated_magnitude = magnitude * gate_mask
        
        gated_stft = gated_magnitude * np.exp(1j * phase)
        gated_audio = librosa.istft(gated_stft, hop_length=hop_length)
        
        return gated_audio
    
    def rms_normalize(self, audio, target_rms_db=-18):
        """RMS нормалізація"""
        current_rms = np.sqrt(np.mean(audio**2))
        if current_rms == 0:
            return audio
        
        current_rms_db = 20 * np.log10(current_rms)
        gain_db = target_rms_db - current_rms_db
        gain_linear = 10**(gain_db / 20)
        
        normalized = audio * gain_linear
        
        peak = np.max(np.abs(normalized))
        if peak > 0.95:
            normalized = normalized * (0.95 / peak)
        
        return normalized
    
    def lufs_normalize(self, audio, target_lufs=-23):
        """LUFS нормалізація (професійний стандарт)"""
        try:
            import pyloudnorm as pyln
            meter = pyln.Meter(self.sr)
            loudness = meter.integrated_loudness(audio)
            
            if np.isfinite(loudness):
                normalized_audio = pyln.normalize.loudness(audio, loudness, target_lufs)
                
                peak = np.max(np.abs(normalized_audio))
                if peak > 0.95:
                    normalized_audio = normalized_audio * (0.95 / peak)
                
                return normalized_audio
        except ImportError:
            pass
        
        return self.rms_normalize(audio, self.target_rms_db)
    
    def process_recording(self, input_path, output_path, backup=True):
        """Повна обробка одного запису"""
        if backup:
            backup_path = str(input_path).replace('.wav', '_original_backup.wav')
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(input_path, backup_path)
        
        try:
            audio, sr = librosa.load(input_path, sr=self.sr, mono=True)
            original_metrics = self.calculate_audio_metrics(audio)
            
            processed_audio = audio.copy()
            
            processed_audio = processed_audio - np.mean(processed_audio)
            
            processed_audio = self.apply_highpass_filter(processed_audio, cutoff=80)
            processed_audio = self.apply_lowpass_filter(processed_audio, cutoff=8000)
            
            processed_audio = self.apply_spectral_gating(processed_audio, gate_threshold=-60)
            
            processed_audio = self.lufs_normalize(processed_audio, self.target_lufs)
            
            processed_audio = np.tanh(processed_audio * 0.95) * 0.95
            
            final_metrics = self.calculate_audio_metrics(processed_audio)
            
            sf.write(output_path, processed_audio, self.sr, subtype='PCM_24')
            
            processing_data = {
                'file': os.path.basename(input_path),
                'original': original_metrics,
                'processed': final_metrics,
                'gain_applied': final_metrics['rms_db'] - original_metrics['rms_db']
            }
            
            self.analysis_data.append(processing_data)
            self.processed_count += 1
            
            print(f"Оброблено: {os.path.basename(input_path)}")
            print(f"  RMS: {original_metrics['rms_db']:.1f} → {final_metrics['rms_db']:.1f} дБ")
            print(f"  Пік: {original_metrics['peak_db']:.1f} → {final_metrics['peak_db']:.1f} дБ")
            if original_metrics['lufs'] and final_metrics['lufs']:
                print(f"  LUFS: {original_metrics['lufs']:.1f} → {final_metrics['lufs']:.1f}")
            print()
            
            return True
            
        except Exception as e:
            print(f"Помилка обробки {input_path}: {e}")
            return False
    
    def process_directory(self, input_dir, output_dir=None):
        """Обробка всієї папки"""
        if output_dir is None:
            output_dir = input_dir
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        wav_files = [f for f in os.listdir(input_dir) 
                    if f.endswith('.wav') and not f.endswith('_original_backup.wav')]
        
        if not wav_files:
            print("Не знайдено WAV файлів для обробки")
            return False
        
        print(f"НОРМАЛІЗАЦІЯ АУДІО ФАЙЛІВ")
        print(f"Знайдено {len(wav_files)} файлів")
        print(f"Цільові параметри:")
        print(f"  LUFS: {self.target_lufs} дБ")
        print(f"  RMS: {self.target_rms_db} дБ")
        print(f"  Максимальний пік: {self.max_peak_db} дБ")
        print("=" * 50)
        
        for filename in wav_files:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            self.process_recording(input_path, output_path)
        
        self.generate_analysis_report(output_dir)
        
        print(f"Нормалізація завершена. Оброблено {self.processed_count} файлів")
        return True
    
    def generate_analysis_report(self, output_dir):
        """Генерація звіту аналізу"""
        if not self.analysis_data:
            return
        
        import json
        
        original_rms = [d['original']['rms_db'] for d in self.analysis_data]
        processed_rms = [d['processed']['rms_db'] for d in self.analysis_data]
        
        original_peaks = [d['original']['peak_db'] for d in self.analysis_data]
        processed_peaks = [d['processed']['peak_db'] for d in self.analysis_data]
        
        report = {
            'summary': {
                'files_processed': len(self.analysis_data),
                'target_lufs': self.target_lufs,
                'target_rms_db': self.target_rms_db
            },
            'original_stats': {
                'rms_db_mean': np.mean(original_rms),
                'rms_db_std': np.std(original_rms),
                'rms_db_min': np.min(original_rms),
                'rms_db_max': np.max(original_rms),
                'peak_db_mean': np.mean(original_peaks),
                'peak_db_max': np.max(original_peaks)
            },
            'processed_stats': {
                'rms_db_mean': np.mean(processed_rms),
                'rms_db_std': np.std(processed_rms),
                'rms_db_min': np.min(processed_rms),
                'rms_db_max': np.max(processed_rms),
                'peak_db_mean': np.mean(processed_peaks),
                'peak_db_max': np.max(processed_peaks)
            },
            'consistency_improvement': {
                'original_rms_std': np.std(original_rms),
                'processed_rms_std': np.std(processed_rms),
                'improvement_factor': np.std(original_rms) / np.std(processed_rms)
            }
        }
        
        report_path = os.path.join(output_dir, 'normalization_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        print(f"Звіт збережено: {report_path}")
        print(f"Покращення консистентності RMS: {report['consistency_improvement']['improvement_factor']:.2f}x")
    

def normalize_desktop_recordings(input_dir, output_dir=None):
    """Нормалізація записів"""
    normalizer = AudioNormalizer(
        sr=44100,
        target_lufs=-23.0
    )
    
    return normalizer.process_directory(input_dir, output_dir)


if __name__ == "__main__":
    input_directory = "dataset-preparation/raw-recordings"
    output_directory = "dataset-preparation/normalized-recordings"
    
    if os.path.exists(input_directory):
        wav_files = [f for f in os.listdir(input_directory) if f.endswith('.wav')]
        if wav_files:
            print(f"Знайдено {len(wav_files)} файлів для нормалізації")
            success = normalize_desktop_recordings(input_directory, output_directory)
            if success:
                print("\nНормалізація завершена. Файли готові для нарізки сегментів.")
        else:
            print(f"Відсутні WAV файли")
    else:
        print(f"Папка не знайдена")
        Path(input_directory).mkdir(parents=True, exist_ok=True)
        print(f"Створено папку: {input_directory}")