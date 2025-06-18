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
            'maj7': 'major_7th',
            'perf8': 'perfect_8th'
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
    
    def minimal_preprocessing(self, audio):
        """Мінімальний препроцесинг для нормалізованого аудіо"""
        audio = audio - np.mean(audio)
        
        audio = np.clip(audio, -0.98, 0.98)
        
        audio = np.tanh(audio * 0.95) * 0.95
        
        return audio
    
    def pitch_shift(self, audio, semitones):
        """Транспозиція"""
        if semitones == 0:
            return audio
        
        try:
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
            return np.mean(audio, axis=0)
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
            
            return np.array(filtered_onsets)
        except Exception as e:
            print(f"Помилка onset detection: {e}")
            duration = len(audio) / self.sr
            num_segments = max(int(duration / 2), 30)
            return np.linspace(1, duration - self.segment_duration - 1, num_segments)
    
    def cut_segments(self, audio, onsets, interval_name, dynamics, output_dir):
        """Нарізка аудіо на сегменти"""
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
                        if i % 10 == 0:
                            print(f"  Збережено: {interval_name}/{dynamics}/{filename}")
                    else:
                        print(f"  Помилка збереження: {filename}")
                        
                except Exception as e:
                    print(f"  Помилка при збереженні {filename}: {e}")
                    continue
        
        return segments
    
    def clean_specific_interval_files(self, output_base_dir, interval_name, dynamics):
        """Очистити файли конкретного інтервалу"""
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
        removed_count = 0
        
        for dir_path in dirs_to_check:
            if os.path.exists(dir_path):
                wav_files = glob.glob(os.path.join(dir_path, "*.wav"))
                for file_path in wav_files:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            removed_count += 1
                    except Exception as e:
                        print(f"Помилка видалення {file_path}: {e}")
        
        if removed_count > 0:
            print(f"  Видалено {removed_count} старих файлів для {interval_name}_{dynamics}")
        
        return removed_count
    
    def create_output_structure(self, output_base_dir):
        """Створення структури папок"""
        dirs_to_create = [
            os.path.join(output_base_dir, "original"),
            os.path.join(output_base_dir, "up_semitone"),
            os.path.join(output_base_dir, "down_semitone")
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_normalized_audio(self, file_path):
        """Завантаження нормалізованого аудіо"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sr, mono=True)
            print(f"  Завантажено: {os.path.basename(file_path)}, тривалість: {len(audio)/sr:.2f}с")
            return audio
        except Exception as e:
            print(f"  Помилка завантаження {file_path}: {e}")
            return None
    
    def process_single_normalized_recording(self, input_file, output_base_dir):
        """Обробка одного нормалізованого запису"""
        print(f"\nОбробка {os.path.basename(input_file)}")
        
        filename = os.path.basename(input_file)
        interval_name, dynamics = self.parse_filename(filename)
        
        if not interval_name or not dynamics:
            print(f"  Неможливо розпарсити: {filename}")
            return 0
        
        print(f"  Інтервал: {interval_name}, Динаміка: {dynamics}")
        
        if not os.path.exists(input_file):
            print(f"  Файл не існує: {input_file}")
            return 0
        
        try:
            self.create_output_structure(output_base_dir)
            self.clean_specific_interval_files(output_base_dir, interval_name, dynamics)
            
            audio = self.load_normalized_audio(input_file)
            if audio is None:
                return 0
            
            print(f"  Мінімальний препроцесинг...")
            audio = self.minimal_preprocessing(audio)
            
            print(f"  Створення транспозицій...")
            audio_up = self.pitch_shift(audio, 1.0)
            audio_down = self.pitch_shift(audio, -1.0)
            
            onsets = self.detect_onsets(audio)
            print(f"  Знайдено {len(onsets)} onset'ів")
            
            if len(onsets) < 10:
                print(f"  УВАГА: Мало onset'ів ({len(onsets)})")
                        
            print(f"  Нарізка сегментів...")
            original_segments = self.cut_segments(
                audio, onsets, interval_name, dynamics, 
                os.path.join(output_base_dir, "original")
            )
            
            up_segments = self.cut_segments(
                audio_up, onsets, interval_name, f"{dynamics}_up",
                os.path.join(output_base_dir, "up_semitone")
            )
            
            down_segments = self.cut_segments(
                audio_down, onsets, interval_name, f"{dynamics}_down",
                os.path.join(output_base_dir, "down_semitone")
            )
            
            total_segments = len(original_segments) + len(up_segments) + len(down_segments)
            print(f"  Результат: {total_segments} сегментів")
            
            return total_segments
            
        except Exception as e:
            print(f"  Помилка обробки {filename}: {e}")
            return 0
        finally:
            import gc
            gc.collect()


def clear_dataset_safely(output_dir):
    """Очищення датасету"""
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
    
    print(f"Видалено {removed_count} файлів, пропущено {failed_count} файлів")
    return failed_count == 0


def process_normalized_recordings(input_dir, output_dir, clear_existing=False):
    """Обробка нормалізованих записів"""
    processor = AudioProcessor()
    
    if clear_existing:
        print("ОЧИЩЕННЯ ІСНУЮЧОГО ДАТАСЕТУ")
        print("=" * 50)
        if os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
                print(f"Очищено: {output_dir}")
            except Exception as e:
                print(f"Помилка очищення: {e}")
                clear_dataset_safely(output_dir)
    
    print("ОБРОБКА НОРМАЛІЗОВАНИХ ЗАПИСІВ")
    print("=" * 50)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    wav_files = [f for f in os.listdir(input_dir) 
                if f.endswith('.wav') and not f.endswith('_original_backup.wav')]
    wav_files.sort()
    
    if not wav_files:
        print("Не знайдено нормалізованих wav файлів!")
        return
    
    print(f"Знайдено {len(wav_files)} нормалізованих файлів:")
    for file in wav_files:
        interval, dynamics = processor.parse_filename(file)
        if interval and dynamics:
            print(f"  {file} → {interval} ({dynamics})")
        else:
            print(f"  {file} → неправильний формат")
    
    print("\nПочинаємо обробку нормалізованих файлів...")
    
    total_processed = 0
    successful_files = 0
    
    for filename in wav_files:
        input_file = os.path.join(input_dir, filename)
        
        try:
            segments_count = processor.process_single_normalized_recording(input_file, output_dir)
            if segments_count > 0:
                total_processed += segments_count
                successful_files += 1
        except Exception as e:
            print(f"Критична помилка при обробці {filename}: {e}")
    
    print(f"\nОБРОБКА ЗАВЕРШЕНА")
    print(f"=" * 50)
    print(f"Успішно оброблено: {successful_files}/{len(wav_files)} файлів")
    print(f"Загалом створено: {total_processed} сегментів")
    print(f"Середньо сегментів на файл: {total_processed/max(successful_files,1):.1f}")
    
    if total_processed >= 3000:
        print(f"Датасет готовий для генерації спектрограм")
    else:
        print(f"УВАГА: Мало сегментів для надійного тренування")


if __name__ == "__main__": 
    normalized_directory = "dataset-preparation/normalized-recordings"
    output_directory = "dataset-preparation/processed-segments"
    
    if os.path.exists(normalized_directory):
        wav_files = [f for f in os.listdir(normalized_directory) 
                    if f.endswith('.wav') and not f.endswith('_original_backup.wav')]
        if wav_files:
            print(f"\nЗнайдено {len(wav_files)} нормалізованих файлів")
            process_normalized_recordings(normalized_directory, output_directory, clear_existing=True)
            
        else:
            print(f"У папці {normalized_directory} відсутні WAV файли")
    else:
        print(f"\nПапка не знайдена: {normalized_directory}")