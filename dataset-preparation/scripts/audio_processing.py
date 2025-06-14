import librosa
import soundfile as sf
import numpy as np
import os
from scipy.signal import butter, filtfilt
from pathlib import Path
import shutil

class AudioProcessor:
    def __init__(self, sr=44100, segment_duration=3.0):
        self.sr = sr
        self.segment_duration = segment_duration
        self.segment_samples = int(segment_duration * sr)
        
        self.interval_map = {
            'perf8': 'perfect_octave',
            'min2': 'minor_2nd',
            'maj2': 'major_2nd',
            'min3': 'minor_3rd',
            'maj3': 'major_3rd',
            'perf4': 'perfect_4th',
            'tritone': 'tritone',
            'perf5': 'perfect_5th',
            'min6': 'minor_6th',
            'maj6': 'major_6th',
            'min7': 'minor_7th',
            'maj7': 'major_7th'
        }
        
        self.dynamics_map = {
            'soft': 'piano',
            'hard': 'forte'
        }
    
    def parse_filename(self, filename):
        base_name = filename.replace('.wav', '')
        parts = base_name.split('_')
        
        if len(parts) >= 2:
            interval_code = parts[0]
            dynamics_code = parts[1]
            
            interval_name = self.interval_map.get(interval_code, interval_code)
            dynamics_name = self.dynamics_map.get(dynamics_code, dynamics_code)
            
            return interval_name, dynamics_name
        
        return None, None
    
    def clean_specific_interval_files(self, output_base_dir, interval_name, dynamics):
        """Очистити файли конкретного інтервалу та динаміки"""
        dirs_to_check = [
            os.path.join(output_base_dir, "original"),
            os.path.join(output_base_dir, "up_semitone"),
            os.path.join(output_base_dir, "down_semitone")
        ]
        
        patterns_to_remove = [
            f"{interval_name}_*_{dynamics}.wav",
            f"{interval_name}_*_{dynamics}_up.wav",
            f"{interval_name}_*_{dynamics}_down.wav"
        ]
        
        import glob
        removed_count = 0
        
        for dir_path in dirs_to_check:
            if os.path.exists(dir_path):
                for pattern in patterns_to_remove:
                    files_to_remove = glob.glob(os.path.join(dir_path, pattern))
                    for file_path in files_to_remove:
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                removed_count += 1
                        except PermissionError:
                            print(f"Не можу видалити {file_path}: файл використовується")
                        except Exception as e:
                            print(f"Помилка видалення {file_path}: {e}")
        
        if removed_count > 0:
            print(f"Видалено {removed_count} старих файлів для {interval_name}_{dynamics}")
        
        return removed_count
    
    def clean_output_directories(self, output_base_dir):
        dirs_to_create = [
            os.path.join(output_base_dir, "original"),
            os.path.join(output_base_dir, "up_semitone"),
            os.path.join(output_base_dir, "down_semitone")
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_audio_stereo(self, file_path):
        """Завантажити аудіо файл у стерео з правильним закриттям ресурсів"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sr, mono=False)
            print(f"Завантажено: {os.path.basename(file_path)}, тривалість: {len(audio[0] if audio.ndim > 1 else audio)/sr:.2f}с")
            return audio
        except Exception as e:
            print(f"Помилка завантаження {file_path}: {e}")
            return None
        finally:
            pass
    
    def hard_limiter(self, audio, threshold=0.95):
        limited = np.clip(audio, -threshold, threshold)
        saturation_factor = 0.8
        limited = np.tanh(limited / saturation_factor) * saturation_factor
        return limited
    
    def compressor(self, audio, threshold=-12, ratio=4.0, attack_time=0.003, release_time=0.1):
        attack_samples = int(attack_time * self.sr)
        release_samples = int(release_time * self.sr)
        
        window_size = 1024
        rms = np.sqrt(np.convolve(audio**2, np.ones(window_size)/window_size, mode='same'))
        rms_db = 20 * np.log10(np.maximum(rms, 1e-10))
        
        reduction_db = np.zeros_like(rms_db)
        mask = rms_db > threshold
        reduction_db[mask] = (rms_db[mask] - threshold) * (1 - 1/ratio)
        
        smoothed_reduction = np.zeros_like(reduction_db)
        for i in range(1, len(reduction_db)):
            if reduction_db[i] > smoothed_reduction[i-1]:
                alpha = 1 - np.exp(-1/attack_samples)
            else:
                alpha = 1 - np.exp(-1/release_samples)
            
            smoothed_reduction[i] = (1-alpha) * smoothed_reduction[i-1] + alpha * reduction_db[i]
        
        gain_reduction = 10 ** (-smoothed_reduction / 20)
        compressed = audio * gain_reduction
        
        return compressed
    
    def noise_gate(self, audio, threshold_db=-45, ratio=10):
        window_size = 1024
        rms = np.sqrt(np.convolve(audio**2, np.ones(window_size)/window_size, mode='same'))
        rms_db = 20 * np.log10(np.maximum(rms, 1e-10))
        
        gate_gain = np.ones_like(rms_db)
        mask = rms_db < threshold_db
        gate_gain[mask] = 1.0 / ratio
        
        gate_gain = np.convolve(gate_gain, np.ones(100)/100, mode='same')
        
        return audio * gate_gain
    
    def advanced_noise_reduction(self, audio):
        try:
            import noisereduce as nr
            reduced_noise = nr.reduce_noise(
                y=audio, 
                sr=self.sr,
                stationary=True,
                prop_decrease=1.0
            )
            print("Використано noisereduce для видалення шумів")
            return reduced_noise
        except ImportError:
            print("noisereduce не встановлено, заміна на noise gate")
            return self.noise_gate(audio)
    
    def adaptive_noise_reduction(self, audio):
        try:
            import noisereduce as nr
            
            noise_duration = int(0.5 * self.sr)
            noise_sample = np.concatenate([
                audio[:noise_duration],
                audio[-noise_duration:]
            ])
            
            signal_rms = np.sqrt(np.mean(audio**2))
            noise_rms = np.sqrt(np.mean(noise_sample**2))
            
            if noise_rms == 0:
                return audio
            
            snr_db = 20 * np.log10(signal_rms / noise_rms)
            print(f"Оцінений SNR: {snr_db:.1f} dB")
            
            if snr_db < 10:
                reduced_noise = nr.reduce_noise(
                    y=audio, 
                    sr=self.sr,
                    stationary=False,
                    prop_decrease=0.8,
                    n_std_thresh_stationary=1.5
                )
                print("Інтенсивне видалення шумів (низький SNR)")
            elif snr_db < 20:
                reduced_noise = nr.reduce_noise(
                    y=audio, 
                    sr=self.sr,
                    stationary=True,
                    prop_decrease=0.6
                )
                print("Помірне видалення шумів (середній SNR)")
            else:
                reduced_noise = nr.reduce_noise(
                    y=audio, 
                    sr=self.sr,
                    stationary=True,
                    prop_decrease=0.3
                )
                print("М'яке видалення шумів (високий SNR)")
            
            return reduced_noise
            
        except ImportError:
            print("noisereduce не встановлено, заміна на noise gate")
            return self.noise_gate(audio)
    
    def apply_highpass_filter(self, audio, cutoff_freq=80):
        nyquist = self.sr / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        b, a = butter(4, normalized_cutoff, btype='high')
        
        if audio.ndim > 1:
            filtered = np.array([filtfilt(b, a, channel) for channel in audio])
        else:
            filtered = filtfilt(b, a, audio)
        
        return filtered
    
    def pitch_shift(self, audio, semitones):
        if audio.ndim > 1:
            shifted_channels = []
            for channel in audio:
                shifted = librosa.effects.pitch_shift(
                    channel, sr=self.sr, n_steps=semitones, bins_per_octave=12
                )
                shifted_channels.append(shifted)
            return np.array(shifted_channels)
        else:
            return librosa.effects.pitch_shift(
                audio, sr=self.sr, n_steps=semitones, bins_per_octave=12
            )
    
    def normalize_final(self, audio, target_db=-1.0):
        if audio.ndim > 1:
            peak_amplitude = np.max(np.abs(audio))
        else:
            peak_amplitude = np.max(np.abs(audio))
        
        if peak_amplitude == 0:
            return audio
        
        target_amplitude = 10 ** (target_db / 20)
        normalized = audio * (target_amplitude / peak_amplitude)
        
        return normalized
    
    def rms_normalize(self, audio, target_rms_db=-20.0):
        rms = np.sqrt(np.mean(audio**2))
        
        if rms == 0:
            return audio
        
        current_rms_db = 20 * np.log10(rms)
        gain_db = target_rms_db - current_rms_db
        gain_linear = 10 ** (gain_db / 20)
        
        normalized = audio * gain_linear
        
        if np.max(np.abs(normalized)) > 0.95:
            peak_reduction = 0.95 / np.max(np.abs(normalized))
            normalized *= peak_reduction
            print(f"RMS нормалізація з peak limiting: {current_rms_db:.1f} → {target_rms_db} dB RMS")
        else:
            print(f"RMS нормалізація: {current_rms_db:.1f} → {target_rms_db} dB RMS")
        
        return normalized
    
    def adaptive_normalize(self, audio):
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        
        if peak == 0:
            return audio
        
        crest_factor = peak / rms if rms > 0 else 1
        crest_factor_db = 20 * np.log10(crest_factor)
        
        print(f"Crest factor: {crest_factor_db:.1f} dB")
        
        if crest_factor_db > 15:
            print("Використовую RMS нормалізацію (високий crest factor)")
            return self.rms_normalize(audio, target_rms_db=-18.0)
        else:
            print("Використовую peak нормалізацію (низький crest factor)")
            return self.normalize_final(audio, target_db=-1.0)
    
    def convert_to_mono(self, audio):
        if audio.ndim > 1:
            mono = np.mean(audio, axis=0)
            print("Конвертовано зі стерео в моно")
            return mono
        return audio
    
    def detect_onsets(self, audio, hop_length=512, min_interval=1.5):
        onset_frames = librosa.onset.onset_detect(
            y=audio,
            sr=self.sr,
            hop_length=hop_length,
            backtrack=True,
            units='time',
            pre_max=20,
            post_max=20,
            pre_avg=100,
            post_avg=100,
            delta=0.2,
            wait=10
        )
        
        filtered_onsets = []
        for onset in onset_frames:
            if not filtered_onsets or onset - filtered_onsets[-1] >= min_interval:
                filtered_onsets.append(onset)
        
        print(f"Знайдено {len(filtered_onsets)} onset'ів")
        return np.array(filtered_onsets)
    
    def cut_segments(self, audio, onsets, interval_name, dynamics, output_dir):
        """Нарізати аудіо на сегменти з правильним збереженням файлів"""
        segments = []
        
        # Переконуємося що папка існує
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for i, onset in enumerate(onsets):
            start_sample = int(onset * self.sr)
            end_sample = start_sample + self.segment_samples
            
            if end_sample <= len(audio):
                segment = audio[start_sample:end_sample]
                
                filename = f"{interval_name}_{i:03d}_{dynamics}.wav"
                filepath = os.path.join(output_dir, filename)
                
                try:
                    # Зберігаємо файл з явним закриттям
                    sf.write(filepath, segment, self.sr)
                    
                    # Перевіряємо що файл дійсно створений
                    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                        segments.append((segment, filename))
                        print(f"Збережено: {filename}")
                    else:
                        print(f"Помилка збереження: {filename}")
                        
                except Exception as e:
                    print(f"Помилка при збереженні {filename}: {e}")
                    continue
        
        return segments
    
    def process_single_recording(self, input_file, output_base_dir):
        """Повна обробка одного запису з правильним управлінням ресурсами"""
        print(f"\n=== Обробка {os.path.basename(input_file)} ===")
        
        filename = os.path.basename(input_file)
        interval_name, dynamics = self.parse_filename(filename)
        
        if not interval_name or not dynamics:
            print(f"Неможливо розпарсити: {filename}")
            return 0
        
        print(f"Інтервал: {interval_name}, Динаміка: {dynamics}")
        
        # Перевіряємо що вхідний файл існує та доступний
        if not os.path.exists(input_file):
            print(f"Файл не існує: {input_file}")
            return 0
        
        try:
            self.clean_output_directories(output_base_dir)
            removed_count = self.clean_specific_interval_files(output_base_dir, interval_name, dynamics)
            
            # Завантажуємо аудіо
            audio = self.load_audio_stereo(input_file)
            if audio is None:
                return 0
            
            print("Застосування хард лімітера...")
            if audio.ndim > 1:
                audio = np.array([self.hard_limiter(channel) for channel in audio])
            else:
                audio = self.hard_limiter(audio)
            
            print("Застосування компресора...")
            if audio.ndim > 1:
                audio = np.array([self.compressor(channel) for channel in audio])
            else:
                audio = self.compressor(audio)
            
            print("Покращене видалення шуму...")
            if audio.ndim > 1:
                audio = np.array([self.adaptive_noise_reduction(channel) for channel in audio])
            else:
                audio = self.adaptive_noise_reduction(audio)
            
            print("High-pass фільтр...")
            audio = self.apply_highpass_filter(audio)
            
            print("Створення транспозицій...")
            audio_up = self.pitch_shift(audio, 1.0)
            audio_down = self.pitch_shift(audio, -1.0)
            
            print("Адаптивна нормалізація гучності...")
            audio = self.adaptive_normalize(audio)
            audio_up = self.adaptive_normalize(audio_up)
            audio_down = self.adaptive_normalize(audio_down)
            
            print("Конвертація в моно...")
            audio_mono = self.convert_to_mono(audio)
            audio_up_mono = self.convert_to_mono(audio_up)
            audio_down_mono = self.convert_to_mono(audio_down)
            
            # Звільняємо пам'ять від стерео версій
            del audio, audio_up, audio_down
            
            onsets = self.detect_onsets(audio_mono)
            
            if len(onsets) < 10:
                print(f"Мало onset'ів ({len(onsets)}). Перевірте запис.")
            
            original_dir = os.path.join(output_base_dir, "original")
            up_dir = os.path.join(output_base_dir, "up_semitone") 
            down_dir = os.path.join(output_base_dir, "down_semitone")
            
            print("Нарізка оригінальних сегментів...")
            original_segments = self.cut_segments(
                audio_mono, onsets, interval_name, dynamics, original_dir
            )
            
            print("Нарізка +1 півтон...")
            up_segments = self.cut_segments(
                audio_up_mono, onsets, interval_name, f"{dynamics}_up", up_dir
            )
            
            print("Нарізка -1 півтон...")
            down_segments = self.cut_segments(
                audio_down_mono, onsets, interval_name, f"{dynamics}_down", down_dir
            )
            
            del audio_mono, audio_up_mono, audio_down_mono
            
            total_segments = len(original_segments) + len(up_segments) + len(down_segments)
            print(f"Створено {total_segments} сегментів для {interval_name}_{dynamics}")
            
            import time
            time.sleep(0.1)
            
            return total_segments
            
        except Exception as e:
            print(f"Помилка обробки {filename}: {e}")
            return 0
        finally:
            import gc
            gc.collect()

def clear_entire_dataset(output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Повністю очищено датасет: {output_dir}")
        return True
    return False

def process_all_recordings(input_dir, output_dir, clear_all=False):
    processor = AudioProcessor()
    
    if clear_all:
        print("ПОВНЕ ОЧИЩЕННЯ ДАТАСЕТУ")
        print("=" * 50)
        clear_entire_dataset(output_dir)
        print("СТВОРЕННЯ НОВОГО ДАТАСЕТУ")
    else:
        print("ПОПОВНЕННЯ ДАТАСЕТУ")
    
    print("=" * 50)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    wav_files.sort()
    
    if not wav_files:
        print("Не знайдено wav файлів!")
        return
    
    print(f"Знайдено {len(wav_files)} файлів для обробки:")
    for file in wav_files:
        interval, dynamics = processor.parse_filename(file)
        if interval and dynamics:
            print(f"  {file} → {interval} ({dynamics})")
        else:
            print(f"  {file} → неправильний формат")
    
    print("\nПочинаємо обробку...")
    
    total_processed = 0
    successful_files = 0
    
    for filename in wav_files:
        input_file = os.path.join(input_dir, filename)
        
        try:
            segments_count = processor.process_single_recording(input_file, output_dir)
            if segments_count > 0:
                total_processed += segments_count
                successful_files += 1
        except Exception as e:
            print(f"Помилка при обробці {filename}: {e}")
    
    print(f"\nОБРОБКА ЗАВЕРШЕНА!")
    print(f"Успішно оброблено: {successful_files}/{len(wav_files)} файлів")
    print(f"Загалом створено: {total_processed} сегментів")
    print(f"Датасет поповнено новими даними")

def show_expected_filenames():
    intervals = ['perf8', 'min2', 'maj2', 'min3', 'maj3', 'perf4', 'tritone', 'perf5', 'min6', 'maj6', 'min7', 'maj7']
    dynamics = ['soft', 'hard']
    
    print("Очікувані назви файлів:")
    print("=" * 40)
    
    for interval in intervals:
        for dynamic in dynamics:
            filename = f"{interval}_{dynamic}.wav"
            processor = AudioProcessor()
            interval_name, dynamics_name = processor.parse_filename(filename)
            print(f"{filename:18} → {interval_name} ({dynamics_name})")
    
    print(f"\nЗагалом: {len(intervals) * len(dynamics)} файлів")

if __name__ == "__main__":
    show_expected_filenames()
    
    input_directory = "dataset-preparation/raw-recordings"
    output_directory = "dataset-preparation/processed-segments"
    
    if os.path.exists(input_directory):
        process_all_recordings(input_directory, output_directory)
    else:
        print(f"Папка {input_directory} не існує!")
        print("Створіть структуру і помістіть wav файли")