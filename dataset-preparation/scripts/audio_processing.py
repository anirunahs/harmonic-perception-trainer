import librosa
import soundfile as sf
import numpy as np
import os
from scipy.signal import butter, filtfilt
from pathlib import Path
import shutil

class AudioProcessor:
    def __init__(self, sr=44100, segment_duration=2.0):
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
    
    def preprocess_audio_for_training(self, audio):
        """Попередня обробка для датасету"""
        audio = audio - np.mean(audio)
        
        audio = np.clip(audio, -0.98, 0.98)
        
        saturation_factor = 0.7
        audio = np.tanh(audio / saturation_factor) * saturation_factor
        
        audio = np.clip(audio, -0.95, 0.95)
        
        audio = self.apply_compressor(audio, threshold=-15, ratio=6.0)
        
        audio = self.apply_highpass_filter(audio, cutoff_freq=80)
        
        audio = self.apply_lowpass_filter(audio, cutoff_freq=8000)
        
        audio = self.normalize_audio(audio, target_db=-1.0)
        
        return audio
    
    def apply_lowpass_filter(self, audio, cutoff_freq=8000):
        """Low-pass фільтр для зрізання високочастотних артефактів"""
        nyquist = self.sr / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        if normalized_cutoff >= 1.0:
            return audio
        
        b, a = butter(4, normalized_cutoff, btype='low')
        
        if audio.ndim > 1:
            filtered = np.array([filtfilt(b, a, channel) for channel in audio])
        else:
            filtered = filtfilt(b, a, audio)
        
        return filtered
    
    def apply_compressor(self, audio, threshold=-12, ratio=4.0, attack_time=0.003, release_time=0.1):
        """Компресор для вирівнювання динаміки"""
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
        return audio * gain_reduction
    
    def apply_highpass_filter(self, audio, cutoff_freq=80):
        """High-pass фільтр для видалення низьких частот"""
        nyquist = self.sr / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        b, a = butter(4, normalized_cutoff, btype='high')
        
        if audio.ndim > 1:
            filtered = np.array([filtfilt(b, a, channel) for channel in audio])
        else:
            filtered = filtfilt(b, a, audio)
        
        return filtered
    
    def normalize_audio(self, audio, target_db=-1.0):
        """Нормалізація аудіо до заданого рівня"""
        if audio.ndim > 1:
            peak_amplitude = np.max(np.abs(audio))
        else:
            peak_amplitude = np.max(np.abs(audio))
        
        if peak_amplitude == 0:
            return audio
        
        target_amplitude = 10 ** (target_db / 20)
        normalized = audio * (target_amplitude / peak_amplitude)
        
        return normalized
    
    def pitch_shift(self, audio, semitones):
        """Транспозиція"""
        if semitones == 0:
            return audio
        
        try:
            if audio.ndim > 1:
                shifted_channels = []
                for channel in audio:
                    shifted = librosa.effects.pitch_shift(
                        channel, sr=self.sr, n_steps=semitones, 
                        bins_per_octave=12, res_type='scipy'
                    )
                    shifted_channels.append(shifted)
                return np.array(shifted_channels)
            else:
                return librosa.effects.pitch_shift(
                    audio, sr=self.sr, n_steps=semitones, 
                    bins_per_octave=12, res_type='scipy'
                )
        except Exception as e:
            print(f"Помилка транспозиції на {semitones} піввтонів: {e}")
            return audio
    
    def convert_to_mono(self, audio):
        """Конвертування стерео в моно"""
        if audio.ndim > 1:
            mono = np.mean(audio, axis=0)
            return mono
        return audio
    
    def detect_onsets(self, audio, min_interval=1.2):
        """Виявлення початків"""
        try:
            onset_frames = librosa.onset.onset_detect(
                y=audio,
                sr=self.sr,
                hop_length=512,
                backtrack=True,
                units='time',
                pre_max=30,
                post_max=30,
                pre_avg=150,
                post_avg=150,
                delta=0.3,
                wait=15
            )
            
            filtered_onsets = []
            for onset in onset_frames:
                if not filtered_onsets or onset - filtered_onsets[-1] >= min_interval:
                    filtered_onsets.append(onset)
            
            return np.array(filtered_onsets)
        except Exception as e:
            print(f"Помилка onset detection: {e}")
            duration = len(audio) / self.sr
            num_segments = max(int(duration / 2), 30)
            return np.linspace(1, duration - self.segment_duration - 1, num_segments)
    
    def cut_segments(self, audio, onsets, interval_name, dynamics, output_dir):
        """Нарізка аудіо на сегменти з організацією по папках інтервалів"""
        interval_dir = os.path.join(output_dir, interval_name)
        dynamics_dir = os.path.join(interval_dir, dynamics)
        Path(dynamics_dir).mkdir(parents=True, exist_ok=True)
        
        segments = []
        for i, onset in enumerate(onsets):
            start_sample = int(onset * self.sr)
            end_sample = start_sample + self.segment_samples
            
            if end_sample <= len(audio):
                segment = audio[start_sample:end_sample]
                
                filename = f"{interval_name}_{i:03d}_{dynamics}.wav"
                filepath = os.path.join(dynamics_dir, filename)
                
                try:
                    sf.write(filepath, segment, self.sr)
                    
                    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                        segments.append((segment, filename))
                        print(f"Збережено: {interval_name}/{dynamics}/{filename}")
                    else:
                        print(f"Помилка збереження: {filename}")
                        
                except Exception as e:
                    print(f"Помилка при збереженні {filename}: {e}")
                    continue
        
        return segments
    
    def clean_specific_interval_files(self, output_base_dir, interval_name, dynamics):
        """Очистити файли конкретного інтервалу та динаміки"""
        dirs_to_check = [
            os.path.join(output_base_dir, "original", interval_name, dynamics),
            os.path.join(output_base_dir, "original", interval_name, f"{dynamics}_up"),
            os.path.join(output_base_dir, "original", interval_name, f"{dynamics}_down"),
            os.path.join(output_base_dir, "up_semitone", interval_name, dynamics),
            os.path.join(output_base_dir, "up_semitone", interval_name, f"{dynamics}_up"),
            os.path.join(output_base_dir, "down_semitone", interval_name, dynamics),
            os.path.join(output_base_dir, "down_semitone", interval_name, f"{dynamics}_down")
        ]
        
        import glob
        import time
        removed_count = 0
        
        for dir_path in dirs_to_check:
            if os.path.exists(dir_path):
                wav_files = glob.glob(os.path.join(dir_path, "*.wav"))
                for file_path in wav_files:
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                removed_count += 1
                                break
                        except PermissionError:
                            if attempt < max_retries - 1:
                                print(f"Файл зайнятий, спроба {attempt + 1}/{max_retries}: {os.path.basename(file_path)}")
                                time.sleep(0.5)
                            else:
                                print(f"Не можу видалити {file_path}: файл використовується іншим процесом")
                        except Exception as e:
                            print(f"Помилка видалення {file_path}: {e}")
                            break
        
        if removed_count > 0:
            print(f"Видалено {removed_count} старих файлів для {interval_name}_{dynamics}")
        
        return removed_count
    
    def clean_output_directories(self, output_base_dir):
        """Створення структури папок"""
        dirs_to_create = [
            os.path.join(output_base_dir, "original"),
            os.path.join(output_base_dir, "up_semitone"),
            os.path.join(output_base_dir, "down_semitone")
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_audio_stereo(self, file_path):
        """Завантаження аудіо файлу"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sr, mono=False)
            print(f"Завантажено: {os.path.basename(file_path)}, тривалість: {len(audio[0] if audio.ndim > 1 else audio)/sr:.2f}с")
            return audio
        except Exception as e:
            print(f"Помилка завантаження {file_path}: {e}")
            return None
    
    def process_single_recording(self, input_file, output_base_dir):
        """Повна обробка одного запису"""
        print(f"\n=== Обробка {os.path.basename(input_file)} ===")
        
        filename = os.path.basename(input_file)
        interval_name, dynamics = self.parse_filename(filename)
        
        if not interval_name or not dynamics:
            print(f"Неможливо розпарсити: {filename}")
            return 0
        
        print(f"Інтервал: {interval_name}, Динаміка: {dynamics}")
        
        if not os.path.exists(input_file):
            print(f"Файл не існує: {input_file}")
            return 0
        
        try:
            self.clean_output_directories(output_base_dir)
            self.clean_specific_interval_files(output_base_dir, interval_name, dynamics)
            
            # Завантажуємо аудіо
            audio = self.load_audio_stereo(input_file)
            if audio is None:
                return 0
            
            print("Попередня обробка аудіо...")
            if audio.ndim > 1:
                audio = np.array([self.preprocess_audio_for_training(channel) for channel in audio])
            else:
                audio = self.preprocess_audio_for_training(audio)
            
            print("Створення транспозицій...")
            audio_up = self.pitch_shift(audio, 1.0)
            audio_down = self.pitch_shift(audio, -1.0)
            
            print("Конвертація в моно...")
            audio_mono = self.convert_to_mono(audio)
            audio_up_mono = self.convert_to_mono(audio_up)
            audio_down_mono = self.convert_to_mono(audio_down)
            
            del audio, audio_up, audio_down
            
            onsets = self.detect_onsets(audio_mono)
            print(f"Знайдено {len(onsets)} onset'ів")
            
            if len(onsets) < 10:
                print(f"Мало onset'ів ({len(onsets)}). Можливо, потрібно налаштувати параметри.")
                        
            print("Нарізка сегментів...")
            original_segments = self.cut_segments(
                audio_mono, onsets, interval_name, dynamics, 
                os.path.join(output_base_dir, "original")
            )
            
            up_segments = self.cut_segments(
                audio_up_mono, onsets, interval_name, f"{dynamics}_up",
                os.path.join(output_base_dir, "up_semitone")
            )
            
            down_segments = self.cut_segments(
                audio_down_mono, onsets, interval_name, f"{dynamics}_down",
                os.path.join(output_base_dir, "down_semitone")
            )
            
            del audio_mono, audio_up_mono, audio_down_mono
            
            total_segments = len(original_segments) + len(up_segments) + len(down_segments)
            print(f"Створено {total_segments} сегментів для {interval_name}_{dynamics}")
            
            return total_segments
            
        except Exception as e:
            print(f"Помилка обробки {filename}: {e}")
            return 0
        finally:
            import gc
            gc.collect()


def safe_clear_dataset(output_dir):
    """Очищення"""
    if not os.path.exists(output_dir):
        return True
    
    removed_count = 0
    failed_count = 0
    
    for root, dirs, files in os.walk(output_dir, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                removed_count += 1
            except PermissionError:
                print(f"Пропускаємо зайнятий файл: {os.path.basename(file_path)}")
                failed_count += 1
            except Exception as e:
                print(f"Помилка видалення {file_path}: {e}")
                failed_count += 1
        
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
            except:
                pass
    
    print(f"Видалено {removed_count} файлів, пропущено {failed_count} зайнятих файлів")
    return failed_count == 0


def clear_entire_dataset(output_dir):
    """Повне очищення датасету з повторними спробами"""
    if os.path.exists(output_dir):
        import time
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                shutil.rmtree(output_dir)
                print(f"Повністю очищено датасет: {output_dir}")
                return True
            except PermissionError as e:
                if attempt < max_retries - 1:
                    print(f"Датасет зайнятий, спроба {attempt + 1}/{max_retries}...")
                    time.sleep(1.0)
                else:
                    print(f"Не можу очистити датасет повністю")
                    print("Використовую безпечне очищення...")
                    return safe_clear_dataset(output_dir)
            except Exception as e:
                print(f"Помилка очищення датасету: {e}")
                return False
    return False

def process_all_recordings(input_dir, output_dir, clear_all=False):
    """Обробка всіх записів"""
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
    
    print(f"\nОБРОБКА ЗАВЕРШЕНА")
    print(f"Успішно оброблено: {successful_files}/{len(wav_files)} файлів")
    print(f"Загалом створено: {total_processed} сегментів")
    print(f"Датасет готовий для тренування CNN")

def show_expected_filenames():
    """Показати очікувані назви файлів"""
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
        process_all_recordings(input_directory, output_directory, clear_all=True)
    else:
        print(f"Папка {input_directory} не існує")
        print("Створіть структуру і помістіть wav файли")
        Path(input_directory).mkdir(parents=True, exist_ok=True)
        print(f"Папка {input_directory} створена")