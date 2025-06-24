import numpy as np
import librosa
import os
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetParameterAnalyzer:
    """Аналізатор параметрів датасету для створення еталонних значень"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.reference_params = {}
        self.audio_stats = defaultdict(list)
        self.spectral_stats = defaultdict(list)
        self.feature_stats = defaultdict(list)
        
    def analyze_dataset_audio_parameters(self):
        """Аналіз аудіо параметрів всього датасету"""
        print("АНАЛІЗ ПАРАМЕТРІВ ДАТАСЕТУ")
        
        dataset_dir = Path(self.dataset_path)
        audio_files = []
        
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith('.wav'):
                    audio_files.append(os.path.join(root, file))
        
        print(f"Знайдено {len(audio_files)} аудіо файлів")
        
        if len(audio_files) == 0:
            print("Не знайдено аудіо файлів у датасеті!")
            return None
        
        processed_count = 0
        for i, file_path in enumerate(audio_files):
            if i % 100 == 0:
                print(f"Проаналізовано: {i}/{len(audio_files)}")
            
            try:
                self._analyze_single_file(file_path)
                processed_count += 1
            except Exception as e:
                logger.warning(f"Помилка аналізу {file_path}: {e}")
        
        print(f"Успішно проаналізовано: {processed_count}/{len(audio_files)} файлів")
        
        self._calculate_reference_parameters()
        
        self._save_reference_parameters()
        
        return self.reference_params
    
    def _analyze_single_file(self, file_path: str):
        """Аналіз одного аудіо файлу"""
        audio, sr = librosa.load(file_path, sr=44100, mono=True)
        
        if len(audio) == 0:
            return
        
        # 1. Базові аудіо параметри
        duration = len(audio) / sr
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        
        if rms > 0:
            rms_db = 20 * np.log10(rms)
            peak_db = 20 * np.log10(peak)
            dynamic_range = peak_db - rms_db
        else:
            rms_db = -100
            peak_db = -100
            dynamic_range = 0
        
        self.audio_stats['duration'].append(duration)
        self.audio_stats['rms'].append(rms)
        self.audio_stats['rms_db'].append(rms_db)
        self.audio_stats['peak'].append(peak)
        self.audio_stats['peak_db'].append(peak_db)
        self.audio_stats['dynamic_range'].append(dynamic_range)
        
        # 2. Спектральні параметри
        try:
            # Спектральний центроїд
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            mean_centroid = np.mean(spectral_centroid)
            
            # Спектральна ширина
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            mean_bandwidth = np.mean(spectral_bandwidth)
            
            # Спектральний rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)[0]
            mean_rolloff = np.mean(spectral_rolloff)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            mean_zcr = np.mean(zcr)
            
            self.spectral_stats['centroid'].append(mean_centroid)
            self.spectral_stats['bandwidth'].append(mean_bandwidth)
            self.spectral_stats['rolloff'].append(mean_rolloff)
            self.spectral_stats['zcr'].append(mean_zcr)
            
        except Exception as e:
            logger.warning(f"Помилка спектрального аналізу: {e}")
        
        # 3. FFT параметри
        try:
            fft_params = self._analyze_fft_characteristics(audio)
            for key, value in fft_params.items():
                self.feature_stats[key].append(value)
        except Exception as e:
            logger.warning(f"Помилка FFT аналізу: {e}")
    
    def _analyze_fft_characteristics(self, audio: np.ndarray):
        """Аналіз FFT характеристик"""
        fft_size = 2048
        if len(audio) < fft_size:
            audio = np.pad(audio, (0, fft_size - len(audio)))
        elif len(audio) > fft_size:
            start = (len(audio) - fft_size) // 2
            audio = audio[start:start + fft_size]
        
        windowed_audio = audio * np.hanning(len(audio))
        fft = np.fft.fft(windowed_audio, n=fft_size)
        magnitude = np.abs(fft[:fft_size // 2 + 1])
        
        freqs = np.fft.fftfreq(fft_size, 1/44100)[:fft_size // 2 + 1]
        min_idx = np.argmax(freqs >= 80)
        max_idx = np.argmax(freqs >= 4000)
        if max_idx == 0:
            max_idx = len(freqs)
        
        magnitude = magnitude[min_idx:max_idx]
        
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        return {
            'fft_mean': np.mean(magnitude),
            'fft_std': np.std(magnitude),
            'fft_max': np.max(magnitude),
            'fft_min': np.min(magnitude),
            'fft_db_mean': np.mean(magnitude_db),
            'fft_db_std': np.std(magnitude_db),
            'fft_db_range': np.max(magnitude_db) - np.min(magnitude_db),
            'fft_energy': np.sum(magnitude**2),
            'fft_spectral_flatness': np.exp(np.mean(np.log(magnitude + 1e-10))) / (np.mean(magnitude) + 1e-10)
        }
    
    def _calculate_reference_parameters(self):
        """Розрахунок еталонних параметрів"""
        print("\nРОЗРАХУНОК ЕТАЛОННИХ ПАРАМЕТРІВ")
        
        self.reference_params = {
            'audio_parameters': {},
            'spectral_parameters': {},
            'fft_parameters': {},
            'quality_thresholds': {}
        }
        
        # 1. Аудіо параметри
        for param, values in self.audio_stats.items():
            if values:
                self.reference_params['audio_parameters'][param] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'percentile_25': float(np.percentile(values, 25)),
                    'percentile_75': float(np.percentile(values, 75)),
                    'target_range': [
                        float(np.percentile(values, 10)),
                        float(np.percentile(values, 90))
                    ]
                }
        
        # 2. Спектральні параметри
        for param, values in self.spectral_stats.items():
            if values:
                self.reference_params['spectral_parameters'][param] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'target_range': [
                        float(np.percentile(values, 10)),
                        float(np.percentile(values, 90))
                    ]
                }
        
        # 3. FFT параметри
        for param, values in self.feature_stats.items():
            if values:
                self.reference_params['fft_parameters'][param] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'target_range': [
                        float(np.percentile(values, 10)),
                        float(np.percentile(values, 90))
                    ]
                }
        
        # 4. Пороги якості
        rms_db_values = self.audio_stats.get('rms_db', [])
        if rms_db_values:
            self.reference_params['quality_thresholds'] = {
                'min_acceptable_rms_db': float(np.percentile(rms_db_values, 5)),
                'target_rms_db': float(np.median(rms_db_values)),
                'max_acceptable_rms_db': float(np.percentile(rms_db_values, 95)),
                'ideal_rms_range': [
                    float(np.percentile(rms_db_values, 25)),
                    float(np.percentile(rms_db_values, 75))
                ]
            }
        
        print("Ключові еталонні параметри:")
        if 'rms_db' in self.reference_params['audio_parameters']:
            rms_params = self.reference_params['audio_parameters']['rms_db']
            print(f"  RMS рівень: {rms_params['mean']:.1f} ± {rms_params['std']:.1f} дБ")
            print(f"  RMS діапазон: [{rms_params['target_range'][0]:.1f}, {rms_params['target_range'][1]:.1f}] дБ")
        
        if 'dynamic_range' in self.reference_params['audio_parameters']:
            dr_params = self.reference_params['audio_parameters']['dynamic_range']
            print(f"  Динамічний діапазон: {dr_params['mean']:.1f} ± {dr_params['std']:.1f} дБ")
        
        if 'centroid' in self.reference_params['spectral_parameters']:
            centroid_params = self.reference_params['spectral_parameters']['centroid']
            print(f"  Спектральний центроїд: {centroid_params['mean']:.0f} ± {centroid_params['std']:.0f} Гц")
    
    def _save_reference_parameters(self):
        """Збереження еталонних параметрів"""
        output_file = "dataset_reference_parameters.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.reference_params, f, indent=2, ensure_ascii=False)
            
            print(f"\nЕталонні параметри збережено: {output_file}")
            
            simplified = {
                'target_rms_db': self.reference_params['audio_parameters']['rms_db']['mean'],
                'rms_range': self.reference_params['audio_parameters']['rms_db']['target_range'],
                'target_dynamic_range': self.reference_params['audio_parameters']['dynamic_range']['mean'],
                'target_spectral_centroid': self.reference_params['spectral_parameters']['centroid']['mean'],
                'quality_thresholds': self.reference_params['quality_thresholds']
            }
            
            simplified_file = "dataset_reference_simplified.json"
            with open(simplified_file, 'w', encoding='utf-8') as f:
                json.dump(simplified, f, indent=2, ensure_ascii=False)
            
            print(f"Спрощені параметри збережено: {simplified_file}")
            
        except Exception as e:
            logger.error(f"Помилка збереження параметрів: {e}")
    
    def create_calibration_report(self):
        """Створення звіту для калібрування"""
        if not self.reference_params:
            print("Відсутній аналіз датасету")
            return
        
        report = []
        report.append("ЗВІТ ПО ЕТАЛОННИХ ПАРАМЕТРАХ ДАТАСЕТУ")
        
        report.append("\n1. АУДІО ПАРАМЕТРИ:")
        for param, stats in self.reference_params['audio_parameters'].items():
            report.append(f"   {param}:")
            report.append(f"     Середнє: {stats['mean']:.3f}")
            report.append(f"     Цільовий діапазон: [{stats['target_range'][0]:.3f}, {stats['target_range'][1]:.3f}]")
        
        report.append("\n2. СПЕКТРАЛЬНІ ПАРАМЕТРИ:")
        for param, stats in self.reference_params['spectral_parameters'].items():
            report.append(f"   {param}:")
            report.append(f"     Середнє: {stats['mean']:.3f}")
            report.append(f"     Цільовий діапазон: [{stats['target_range'][0]:.3f}, {stats['target_range'][1]:.3f}]")
        
        report.append("\n3. ПОРОГИ ЯКОСТІ:")
        qt = self.reference_params['quality_thresholds']
        report.append(f"   Мінімальний прийнятний RMS: {qt['min_acceptable_rms_db']:.1f} дБ")
        report.append(f"   Цільовий RMS: {qt['target_rms_db']:.1f} дБ")
        report.append(f"   Максимальний прийнятний RMS: {qt['max_acceptable_rms_db']:.1f} дБ")
        report.append(f"   Ідеальний RMS діапазон: [{qt['ideal_rms_range'][0]:.1f}, {qt['ideal_rms_range'][1]:.1f}] дБ")
        
        report_text = "\n".join(report)
        
        with open("dataset_calibration_report.txt", 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nЗвіт збережено: dataset_calibration_report.txt")


def analyze_dataset_parameters(dataset_path: str):
    """Основна функція аналізу"""
    analyzer = DatasetParameterAnalyzer(dataset_path)
    
    reference_params = analyzer.analyze_dataset_audio_parameters()
    
    if reference_params:
        analyzer.create_calibration_report()
        
        print("АНАЛІЗ ЗАВЕРШЕНО")
        
        return reference_params
    else:
        print("Помилка аналізу датасету")
        return None


if __name__ == "__main__":
    dataset_path = "dataset-preparation/processed-segments"
    
    if os.path.exists(dataset_path):
        print(f"Аналіз датасету: {dataset_path}")
        analyze_dataset_parameters(dataset_path)
    else:
        print(f"Датасет не знайдено: {dataset_path}")
