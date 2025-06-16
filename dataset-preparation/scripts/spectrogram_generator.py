import numpy as np
import librosa
import cv2
import os
from pathlib import Path
import json
from glob import glob

class OptimizedSpectrogramGenerator:
    def __init__(self, target_shape=(1025, 97)):
        self.target_shape = target_shape
        self.sr = 22050
        
        self.interval_mapping = {
            'minor_2nd': 'minor_2nd',
            'major_2nd': 'major_2nd', 
            'minor_3rd': 'minor_3rd',
            'major_3rd': 'major_3rd',
            'perfect_4th': 'perfect_4th',
            'tritone': 'tritone',
            'perfect_5th': 'perfect_5th',
            'minor_6th': 'minor_6th',
            'major_6th': 'major_6th',
            'minor_7th': 'minor_7th',
            'major_7th': 'major_7th',
            'perfect_octave': 'perfect_8th'
        }
        
        self.stats = {
            'total_processed': 0,
            'by_interval': {},
            'errors': 0
        }
    
    def audio_to_spectrogram(self, audio_path):
        """Конвертація аудіо в спектрограму (метод з прикладу)"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            D = librosa.stft(y)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            
            ImageAudio = (S_db * 255).astype(np.uint8)
            
            resized_image = cv2.resize(
                ImageAudio, 
                (self.target_shape[1], self.target_shape[0]),
                interpolation=cv2.INTER_AREA
            )
            
            return resized_image
            
        except Exception as e:
            print(f"Помилка обробки {audio_path}: {e}")
            return None
    
    def save_spectrogram_simple(self, spectrogram_image, output_path):
        """Просте збереження спектрограми"""
        try:
            success = cv2.imwrite(output_path, spectrogram_image)
            return success
        except Exception as e:
            print(f"Помилка збереження {output_path}: {e}")
            return False
    
    def process_single_audio(self, audio_path, interval_type, output_dir):
        """Обробка одного аудіофайлу"""
        try:
            spectrogram = self.audio_to_spectrogram(audio_path)
            if spectrogram is None:
                return False
            
            interval_dir = os.path.join(output_dir, interval_type)
            Path(interval_dir).mkdir(parents=True, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            parent_dirs = os.path.normpath(audio_path).split(os.sep)
            
            variant = "original"
            if "up_semitone" in parent_dirs:
                variant = "up"
            elif "down_semitone" in parent_dirs:
                variant = "down"
            
            dynamics = "unknown"
            if "piano" in base_name.lower():
                dynamics = "piano"
            elif "forte" in base_name.lower():
                dynamics = "forte"
            
            output_filename = f"{interval_type}_{variant}_{dynamics}_{base_name}.png"
            output_path = os.path.join(interval_dir, output_filename)
            
            success = self.save_spectrogram_simple(spectrogram, output_path)
            
            if success:
                self.stats['total_processed'] += 1
                if interval_type not in self.stats['by_interval']:
                    self.stats['by_interval'][interval_type] = 0
                self.stats['by_interval'][interval_type] += 1
                return True
            else:
                self.stats['errors'] += 1
                return False
                
        except Exception as e:
            print(f"Помилка обробки файлу {audio_path}: {e}")
            self.stats['errors'] += 1
            return False
    
    def process_dataset_structure(self, input_base_dir, output_dir):
        """Обробка структури датасету"""
        print("ГЕНЕРАЦІЯ СПЕКТРОГРАМ (ОПТИМІЗОВАНИЙ МЕТОД)")
        print("=" * 60)
        print(f"Розміри спектрограм: {self.target_shape[0]}×{self.target_shape[1]} (висота×ширина)")
        print(f"Sample rate: {self.sr} Hz")
        print(f"Метод: STFT → amplitude_to_db → resize")
        
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        total_files = 0
        processed_files = 0
        
        variants = ['original', 'up_semitone', 'down_semitone']
        
        for variant in variants:
            variant_path = os.path.join(input_base_dir, variant)
            if not os.path.exists(variant_path):
                print(f"Варіант {variant} не знайдено")
                continue
            
            print(f"\nОбробка варіанту: {variant}")
            
            for interval_folder in os.listdir(variant_path):
                interval_path = os.path.join(variant_path, interval_folder)
                if not os.path.isdir(interval_path):
                    continue
                
                if interval_folder not in self.interval_mapping:
                    print(f"  Пропуск невідомого інтервалу: {interval_folder}")
                    continue
                
                mapped_interval = self.interval_mapping[interval_folder]
                
                wav_pattern = os.path.join(interval_path, "**", "*.wav")
                wav_files = glob(wav_pattern, recursive=True)
                
                interval_processed = 0
                
                print(f"  {interval_folder} → {mapped_interval}: знайдено {len(wav_files)} файлів")
                
                for wav_file in wav_files:
                    total_files += 1
                    
                    success = self.process_single_audio(wav_file, mapped_interval, output_dir)
                    if success:
                        processed_files += 1
                        interval_processed += 1
                    
                    if total_files % 200 == 0:
                        print(f"    Прогрес: {processed_files}/{total_files}")
                
                print(f"    Результат: {interval_processed} спектрограм")
        
        print(f"\n{'='*60}")
        print("РЕЗУЛЬТАТИ ГЕНЕРАЦІЇ")
        print(f"{'='*60}")
        print(f"Всього знайдено файлів: {total_files}")
        print(f"Успішно оброблено: {processed_files}")
        print(f"Помилок: {self.stats['errors']}")
        
        if processed_files > 0:
            success_rate = (processed_files / total_files) * 100
            print(f"Успішність: {success_rate:.1f}%")
            
            print(f"\nРозподіл по інтервалах:")
            for interval, count in sorted(self.stats['by_interval'].items()):
                print(f"  {interval:12}: {count:4} спектрограм")
            
            counts = list(self.stats['by_interval'].values())
            if counts:
                min_count = min(counts)
                max_count = max(counts)
                balance_ratio = min_count / max_count if max_count > 0 else 0
                print(f"\nБалансованість датасету: {balance_ratio:.2f}")
                if balance_ratio < 0.7:
                    print("УВАГА: Датасет незбалансований! Деякі класи мають мало прикладів.")
                else:
                    print("Датасет добре збалансований")
        
        self._save_generation_stats(output_dir)
        
        return processed_files > 0
    
    def _save_generation_stats(self, output_dir):
        """Збереження детальної статистики"""
        stats_path = os.path.join(output_dir, 'generation_stats.json')
        
        try:
            stats_data = {
                'generation_method': 'STFT + amplitude_to_db + resize',
                'target_shape': f"{self.target_shape[0]}×{self.target_shape[1]} (height×width)",
                'sample_rate': self.sr,
                'total_processed': self.stats['total_processed'],
                'errors': self.stats['errors'],
                'by_interval': self.stats['by_interval'],
                'interval_mapping': self.interval_mapping,
                'balance_analysis': self._analyze_balance()
            }
            
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, indent=2, ensure_ascii=False)
                
            print(f"\nСтатистика збережена: {stats_path}")
            
        except Exception as e:
            print(f"Помилка збереження статистики: {e}")
    
    def _analyze_balance(self):
        """Аналіз збалансованості датасету"""
        if not self.stats['by_interval']:
            return {}
        
        counts = list(self.stats['by_interval'].values())
        return {
            'min_samples': min(counts),
            'max_samples': max(counts),
            'mean_samples': np.mean(counts),
            'std_samples': np.std(counts),
            'balance_ratio': min(counts) / max(counts) if max(counts) > 0 else 0
        }


def generate_spectrograms_optimized(input_dir, output_dir, shape=(1025, 97)):
    """Генерація спектрограм оптимізованим методом"""
    
    print(f"Запуск оптимізованої генерації спектрограм")
    print(f"Вхідна папка: {input_dir}")
    print(f"Вихідна папка: {output_dir}")
    print(f"Розміри зображень: {shape[0]}×{shape[1]} (висота×ширина)")
    
    generator = OptimizedSpectrogramGenerator(target_shape=shape)
    success = generator.process_dataset_structure(input_dir, output_dir)
    
    if success:
        print(f"\nСпектрограми успішно згенеровано!")
        print(f"Результат: {output_dir}")
        print("Готово для тренування CNN!")
    else:
        print("\nПомилка генерації спектрограм")
    
    return success

if __name__ == "__main__":
    input_directory = "dataset-preparation/processed-segments"
    output_directory = "dataset-preparation/spectrograms"
    
    SPECTROGRAM_SHAPE = (1025, 97)
    
    if os.path.exists(input_directory):
        generate_spectrograms_optimized(
            input_directory, 
            output_directory,
            shape=SPECTROGRAM_SHAPE
        )
        
    else:
        print(f"Папка не знайдена: {input_directory}")
        print("Спочатку запустіть audio_processing.py")