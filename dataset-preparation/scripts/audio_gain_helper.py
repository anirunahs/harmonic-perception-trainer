import librosa
import soundfile as sf
import numpy as np
import os
from pathlib import Path
import json

class AudioGainProcessor:
    def __init__(self, sr=44100):
        self.sr = sr
        self.processed_files = 0
        self.skipped_files = 0
        self.stats = {
            'files_processed': 0,
            'files_skipped': 0,
            'average_gain_applied': 0,
            'files_by_gain': {}
        }
    
    def analyze_audio_levels(self, audio):
        """Аналіз рівнів аудіо"""
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        rms_db = 20 * np.log10(rms + 1e-10)
        peak_db = 20 * np.log10(peak + 1e-10)
        
        dynamic_range = peak_db - rms_db
        
        return {
            'rms': rms,
            'peak': peak,
            'rms_db': rms_db,
            'peak_db': peak_db,
            'dynamic_range': dynamic_range
        }
    
    def calculate_optimal_gain(self, audio, target_rms_db=-18, max_peak_db=-3):
        """Розрахунок оптимального gain"""
        levels = self.analyze_audio_levels(audio)
        
        # Розрахунок gain на основі RMS
        rms_gain_db = target_rms_db - levels['rms_db']
        
        # Розрахунок максимального gain на основі піку
        max_gain_db = max_peak_db - levels['peak_db']
        
        # Вибираємо менший gain щоб не було кліпування
        optimal_gain_db = min(rms_gain_db, max_gain_db)
        
        # Обмежуємо gain (не більше +20дБ, не менше -6дБ)
        optimal_gain_db = np.clip(optimal_gain_db, -6, 20)
        
        return optimal_gain_db, levels
    
    def apply_gain(self, audio, gain_db):
        """Застосування gain до аудіо"""
        if gain_db == 0:
            return audio
        
        gain_linear = 10 ** (gain_db / 20)
        gained_audio = audio * gain_linear
        
        # Soft limiting
        if np.max(np.abs(gained_audio)) > 0.95:
            gained_audio = np.tanh(gained_audio * 0.95) * 0.95
        
        return gained_audio
    
    def smart_gain_correction(self, audio, aggressive=False):
        """Розумна корекція gain з різними стратегіями"""
        if aggressive:
            # Інтенсивна корекція для дуже тихих файлів
            target_rms_db = -15
            max_peak_db = -1
        else:
            # М'яка корекція
            target_rms_db = -18
            max_peak_db = -3
        
        gain_db, original_levels = self.calculate_optimal_gain(
            audio, target_rms_db, max_peak_db
        )
        
        if abs(gain_db) < 1.0:  # Якщо gain менше 1дБ, не застосовуємо
            return audio, 0, original_levels
        
        corrected_audio = self.apply_gain(audio, gain_db)
        final_levels = self.analyze_audio_levels(corrected_audio)
        
        return corrected_audio, gain_db, original_levels, final_levels
    
    def process_file(self, input_path, output_path=None, backup=True):
        """Обробка одного файлу"""
        if output_path is None:
            output_path = input_path
        
        try:
            # Завантаження
            audio, sr = librosa.load(input_path, sr=self.sr)
            original_levels = self.analyze_audio_levels(audio)
            
            # Перевірка
            if original_levels['rms_db'] > -25:  # Якщо RMS вище -25дБ, корекція не потрібна
                print(f"Пропускаємо {os.path.basename(input_path)} (RMS: {original_levels['rms_db']:.1f}дБ)")
                self.skipped_files += 1
                return False
            
            # Вибір стратегії
            aggressive = original_levels['rms_db'] < -40
            
            # Застосування корекції
            corrected_audio, gain_applied, orig_levels, final_levels = self.smart_gain_correction(
                audio, aggressive
            )
            
            if abs(gain_applied) < 1.0:
                print(f"Пропускаємо {os.path.basename(input_path)} (мінімальна корекція)")
                self.skipped_files += 1
                return False
            
            if backup and output_path == input_path:
                backup_path = input_path.replace('.wav', '_original.wav')
                if not os.path.exists(backup_path):
                    import shutil
                    shutil.copy2(input_path, backup_path)
            
            # Збереження
            sf.write(output_path, corrected_audio, self.sr)
            
            print(f"Оброблено {os.path.basename(input_path)}:")
            print(f"  RMS: {orig_levels['rms_db']:.1f}дБ → {final_levels['rms_db']:.1f}дБ")
            print(f"  Пік: {orig_levels['peak_db']:.1f}дБ → {final_levels['peak_db']:.1f}дБ")
            print(f"  Gain: +{gain_applied:.1f}дБ")
            
            self.processed_files += 1
            
            # Статистика
            gain_range = f"{int(gain_applied//5)*5}-{int(gain_applied//5)*5+5}дБ"
            if gain_range not in self.stats['files_by_gain']:
                self.stats['files_by_gain'][gain_range] = 0
            self.stats['files_by_gain'][gain_range] += 1
            
            return True
            
        except Exception as e:
            print(f"Помилка обробки {input_path}: {e}")
            return False
    
    def process_directory(self, input_dir, output_dir=None, recursive=True):
        """Обробка всієї папки"""
        if output_dir is None:
            output_dir = input_dir
        
        print("КОРЕКЦІЯ ГУЧНОСТІ АУДІОФАЙЛІВ")
        print("=" * 50)
        
        wav_files = []
        if recursive:
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.endswith('.wav') and not file.endswith('_original.wav'):
                        wav_files.append(os.path.join(root, file))
        else:
            wav_files = [
                os.path.join(input_dir, f) 
                for f in os.listdir(input_dir) 
                if f.endswith('.wav') and not f.endswith('_original.wav')
            ]
        
        if not wav_files:
            print("Не знайдено WAV файлів для обробки")
            return False
        
        print(f"Знайдено {len(wav_files)} файлів")
        
        total_gain = 0
        
        for i, input_path in enumerate(wav_files, 1):
            print(f"\n[{i}/{len(wav_files)}]", end=" ")
            
            if output_dir != input_dir:
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
            else:
                output_path = input_path
            
            success = self.process_file(input_path, output_path)
        
        # Фінальна статистика
        self.stats['files_processed'] = self.processed_files
        self.stats['files_skipped'] = self.skipped_files
        
        print(f"\n{'='*50}")
        print("КОРЕКЦІЯ ЗАВЕРШЕНА")
        print(f"{'='*50}")
        print(f"Оброблено файлів: {self.processed_files}")
        print(f"Пропущено файлів: {self.skipped_files}")
        
        if self.processed_files > 0:
            print(f"\nРозподіл за gain:")
            for gain_range, count in sorted(self.stats['files_by_gain'].items()):
                print(f"  {gain_range}: {count} файлів")
        
        return self.processed_files > 0
    
    def analyze_directory_levels(self, directory):
        """Аналіз рівнів всіх файлів в папці"""
        print("АНАЛІЗ РІВНІВ АУДІОФАЙЛІВ")
        print("=" * 50)
        
        wav_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.wav') and not file.endswith('_original.wav'):
                    wav_files.append(os.path.join(root, file))
        
        if not wav_files:
            print("Не знайдено WAV файлів")
            return
        
        levels_data = []
        quiet_files = []
        loud_files = []
        
        print(f"Аналізуємо {len(wav_files)} файлів...\n")
        
        for file_path in wav_files:
            try:
                audio, sr = librosa.load(file_path, sr=self.sr)
                levels = self.analyze_audio_levels(audio)
                levels['filename'] = os.path.basename(file_path)
                levels_data.append(levels)
                
                # Категоризація
                if levels['rms_db'] < -30:
                    quiet_files.append((levels['filename'], levels['rms_db']))
                elif levels['rms_db'] > -12:
                    loud_files.append((levels['filename'], levels['rms_db']))
                
            except Exception as e:
                print(f"Помилка аналізу {file_path}: {e}")
        
        # Статистика
        rms_values = [l['rms_db'] for l in levels_data]
        peak_values = [l['peak_db'] for l in levels_data]
        
        print(f"ЗАГАЛЬНА СТАТИСТИКА:")
        print(f"RMS рівні:")
        print(f"  Мінімум: {min(rms_values):.1f} дБ")
        print(f"  Максимум: {max(rms_values):.1f} дБ")
        print(f"  Середнє: {np.mean(rms_values):.1f} дБ")
        print(f"  Медіана: {np.median(rms_values):.1f} дБ")
        
        print(f"\nПік рівні:")
        print(f"  Мінімум: {min(peak_values):.1f} дБ")
        print(f"  Максимум: {max(peak_values):.1f} дБ")
        print(f"  Середнє: {np.mean(peak_values):.1f} дБ")
        
        # Проблемні файли
        if quiet_files:
            print(f"\nТИХІ ФАЙЛИ (RMS < -30дБ): {len(quiet_files)}")
            for filename, rms in sorted(quiet_files, key=lambda x: x[1])[:10]:
                print(f"  {filename}: {rms:.1f} дБ")
            if len(quiet_files) > 10:
                print(f"  ... та ще {len(quiet_files)-10} файлів")
        
        if loud_files:
            print(f"\nГУЧНІ ФАЙЛИ (RMS > -12дБ): {len(loud_files)}")
            for filename, rms in sorted(loud_files, key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {filename}: {rms:.1f} дБ")
        
        print(f"\nРЕКОМЕНДАЦІЇ:")
        needs_gain = len([l for l in levels_data if l['rms_db'] < -25])
        if needs_gain > 0:
            print(f"- {needs_gain} файлів потребують підсилення gain")
            print("- Запустіть process_directory() для автоматичної корекції")
        else:
            print("- Всі файли мають прийнятні рівні гучності")
        
        return levels_data


def boost_quiet_files(input_directory, output_directory=None, analyze_first=True):
    """Головна функція для підсилення тихих файлів"""
    processor = AudioGainProcessor()
    
    if analyze_first:
        print("Спочатку аналізуємо рівні...")
        processor.analyze_directory_levels(input_directory)
        print("\n" + "="*50)
    
    success = processor.process_directory(input_directory, output_directory)
    
    if success:
        print("\nГучність файлів скорегована!")
        print("Оригінали збережені з суфіксом '_original.wav'")
    else:
        print("\nНе знайдено файлів для корекції")
    
    return success


if __name__ == "__main__":

    input_dir = "dataset-preparation/processed-segments"
    processor = AudioGainProcessor()
    processor.analyze_directory_levels(input_dir)