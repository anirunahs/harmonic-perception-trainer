import numpy as np
import librosa
import os
import json
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class FFTDatasetGenerator:
    def __init__(self, fft_size=2048, sample_rate=22050):
        self.fft_size = fft_size
        self.sample_rate = sample_rate
        self.freq_bins = fft_size // 2 + 1
        
        self.min_freq = 80
        self.max_freq = 4000
        self.freq_range = self._calculate_freq_range()
        
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
            'perfect_8th': 'perfect_8th'
        }
        
        self.stats = {
            'total_processed': 0,
            'by_interval': {},
            'feature_stats': {},
            'errors': 0
        }
    
    def _calculate_freq_range(self):
        """Розрахунок індексів частотного діапазону"""
        freqs = np.fft.fftfreq(self.fft_size, 1/self.sample_rate)[:self.freq_bins]
        min_idx = np.argmax(freqs >= self.min_freq)
        max_idx = np.argmax(freqs >= self.max_freq)
        if max_idx == 0:
            max_idx = len(freqs)
        return min_idx, max_idx
    
    def extract_harmonic_features(self, audio):
        """Виділення гармонічних ознак з аудіо"""
        if len(audio) < self.fft_size:
            audio = np.pad(audio, (0, self.fft_size - len(audio)))
        elif len(audio) > self.fft_size:
            start = (len(audio) - self.fft_size) // 2
            audio = audio[start:start + self.fft_size]
        
        windowed_audio = audio * np.hanning(len(audio))
        
        fft = np.fft.fft(windowed_audio, n=self.fft_size)
        magnitude = np.abs(fft[:self.freq_bins])
        
        min_idx, max_idx = self.freq_range
        magnitude = magnitude[min_idx:max_idx]
        
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        magnitude_normalized = (magnitude_db - np.min(magnitude_db)) / (np.max(magnitude_db) - np.min(magnitude_db) + 1e-10)
        
        additional_features = self._extract_additional_features(magnitude, audio)
        
        all_features = np.concatenate([magnitude_normalized, additional_features])
        
        return all_features
    
    def _extract_additional_features(self, magnitude, audio):
        """Додаткові музичні ознаки"""
        features = []
        
        freqs = np.linspace(self.min_freq, self.max_freq, len(magnitude))
        
        spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)
        features.append(spectral_centroid / self.max_freq)
        
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / (np.sum(magnitude) + 1e-10))
        features.append(spectral_bandwidth / self.max_freq)
        
        cumsum = np.cumsum(magnitude)
        rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
        if len(rolloff_idx) > 0:
            rolloff_freq = freqs[rolloff_idx[0]]
            features.append(rolloff_freq / self.max_freq)
        else:
            features.append(1.0)
        
        rms = np.sqrt(np.mean(audio ** 2))
        features.append(rms)
        
        zcr = np.mean(np.abs(np.diff(np.sign(audio))))
        features.append(zcr)
        
        peak_indices = self._find_spectral_peaks(magnitude)
        peak_features = self._calculate_peak_features(peak_indices, freqs, magnitude)
        features.extend(peak_features)
        
        return np.array(features)
    
    def _find_spectral_peaks(self, magnitude, prominence=0.1):
        """Знаходження піків в спектрі"""
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(magnitude, prominence=prominence * np.max(magnitude))
            return peaks[:10]
        except ImportError:
            peaks = []
            for i in range(1, len(magnitude) - 1):
                if magnitude[i] > magnitude[i-1] and magnitude[i] > magnitude[i+1]:
                    if magnitude[i] > prominence * np.max(magnitude):
                        peaks.append(i)
                if len(peaks) >= 10:
                    break
            return peaks
    
    def _calculate_peak_features(self, peak_indices, freqs, magnitude):
        """Розрахунок ознак піків"""
        features = []
        
        if len(peak_indices) == 0:
            return [0.0] * 15
        
        fundamental_freq = freqs[peak_indices[0]] if len(peak_indices) > 0 else 1.0
        features.append(fundamental_freq / self.max_freq)
        
        for i in range(1, min(5, len(peak_indices))):
            ratio = freqs[peak_indices[i]] / fundamental_freq if fundamental_freq > 0 else 0.0
            features.append(ratio)
        
        while len(features) < 6:
            features.append(0.0)
        
        max_magnitude = np.max(magnitude)
        for i in range(min(5, len(peak_indices))):
            amplitude = magnitude[peak_indices[i]] / max_magnitude
            features.append(amplitude)
        
        while len(features) < 11:
            features.append(0.0)
        
        for i in range(1, min(5, len(peak_indices))):
            interval = (freqs[peak_indices[i]] - freqs[peak_indices[i-1]]) / self.max_freq
            features.append(interval)
        
        while len(features) < 15:
            features.append(0.0)
        
        return features[:15]
    
    def process_audio_segments(self, segments_dir, output_dir):
        """Обробка аудіосегментів та створення FFT датасету"""
        
        print("СТВОРЕННЯ FFT ДАТАСЕТУ")
        print("=" * 50)
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        X = []
        y = []
        file_info = []
        
        variants = ['original', 'up_semitone', 'down_semitone']
        
        for variant in variants:
            variant_path = os.path.join(segments_dir, variant)
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
                
                processed_count = 0
                
                for dynamics_folder in os.listdir(interval_path):
                    dynamics_path = os.path.join(interval_path, dynamics_folder)
                    if not os.path.isdir(dynamics_path):
                        continue
                    
                    wav_files = [f for f in os.listdir(dynamics_path) if f.endswith('.wav')]
                    
                    for wav_file in wav_files:
                        audio_path = os.path.join(dynamics_path, wav_file)
                        
                        try:
                            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                            features = self.extract_harmonic_features(audio)
                            
                            X.append(features)
                            y.append(mapped_interval)
                            file_info.append({
                                'file': wav_file,
                                'interval': mapped_interval,
                                'variant': variant,
                                'dynamics': dynamics_folder
                            })
                            
                            processed_count += 1
                            self.stats['total_processed'] += 1
                            
                        except Exception as e:
                            print(f"    Помилка обробки {wav_file}: {e}")
                            self.stats['errors'] += 1
                
                self.stats['by_interval'][mapped_interval] = self.stats['by_interval'].get(mapped_interval, 0) + processed_count
                print(f"  {interval_folder} → {mapped_interval}: {processed_count} файлів")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n{'='*50}")
        print("РЕЗУЛЬТАТ")
        print(f"{'='*50}")
        print(f"Загалом оброблено: {len(X)} зразків")
        print(f"Розмірність ознак: {X.shape[1]}")
        print(f"Помилок: {self.stats['errors']}")
        
        print(f"\nРозподіл по інтервалах:")
        for interval, count in sorted(self.stats['by_interval'].items()):
            percentage = (count / len(X)) * 100 if len(X) > 0 else 0
            print(f"  {interval:15}: {count:4} зразків ({percentage:5.1f}%)")
        
        self._save_dataset(X, y, file_info, output_dir)
        
        return X, y, file_info
    
    def _save_dataset(self, X, y, file_info, output_dir):
        """Збереження датасету"""
        
        dataset = {
            'features': X,
            'labels': y,
            'file_info': file_info,
            'feature_names': self._get_feature_names(),
            'class_names': sorted(list(set(y))),
            'metadata': {
                'fft_size': self.fft_size,
                'sample_rate': self.sample_rate,
                'freq_range': self.freq_range,
                'min_freq': self.min_freq,
                'max_freq': self.max_freq,
                'total_samples': len(X),
                'feature_dim': X.shape[1]
            }
        }
                
        pickle_path = os.path.join(output_dir, 'fft_dataset.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        npz_path = os.path.join(output_dir, 'fft_dataset.npz')
        np.savez_compressed(npz_path, features=X, labels=y)
        
        metadata_path = os.path.join(output_dir, 'fft_dataset_metadata.json')
        json_metadata = {
            'class_names': dataset['class_names'],
            'feature_names': dataset['feature_names'],
            'metadata': dataset['metadata'],
            'stats': self.stats
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(json_metadata, f, indent=2, ensure_ascii=False, default=self._json_serializer)
        
        self._save_csv_version(X, y, file_info, output_dir)
        
        print(f"\nДатасет збережено:")
        print(f"  Pickle: {pickle_path}")
        print(f"  NPZ: {npz_path}")
        print(f"  Metadata: {metadata_path}")
        
        self._create_train_test_split(X, y, output_dir)
    
    def _get_feature_names(self):
        """Отримання назв ознак"""
        names = []
        
        min_idx, max_idx = self.freq_range
        freq_step = (self.max_freq - self.min_freq) / (max_idx - min_idx)
        for i in range(max_idx - min_idx):
            freq = self.min_freq + i * freq_step
            names.append(f'fft_mag_{freq:.1f}Hz')
        
        additional_names = [
            'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
            'rms_energy', 'zero_crossing_rate', 'fundamental_freq'
        ]
        
        for i in range(1, 5):
            additional_names.append(f'harmonic_ratio_{i}')
        
        for i in range(5):
            additional_names.append(f'peak_amplitude_{i}')
        
        for i in range(4):
            additional_names.append(f'peak_interval_{i}')
        
        names.extend(additional_names)
        return names
    
    def _save_csv_version(self, X, y, file_info, output_dir):
        """Збереження CSV версії для аналізу"""
        import pandas as pd
        
        feature_names = self._get_feature_names()
        df = pd.DataFrame(X, columns=feature_names)
        df['interval'] = y
        
        for i, info in enumerate(file_info):
            df.loc[i, 'file'] = info['file']
            df.loc[i, 'variant'] = info['variant']
            df.loc[i, 'dynamics'] = info['dynamics']
        
        csv_path = os.path.join(output_dir, 'fft_dataset.csv')
        df.to_csv(csv_path, index=False)
        print(f"  CSV: {csv_path}")
    
    def _create_train_test_split(self, X, y, output_dir):
        """Створення train/test розбиття"""
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        splits = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
        split_path = os.path.join(output_dir, 'fft_dataset_splits.npz')
        np.savez_compressed(split_path, **splits)
        
        print(f"  Train/Test split: {split_path}")
        print(f"    Train: {len(X_train)} зразків")
        print(f"    Test: {len(X_test)} зразків")
    
    def _json_serializer(self, obj):
        """Серіалізатор для JSON"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def visualize_features(self, output_dir, max_samples_per_class=50):
        """Візуалізація ознак"""
        
        pickle_path = os.path.join(output_dir, 'fft_dataset.pkl')
        if not os.path.exists(pickle_path):
            print("Датасет не знайдено для візуалізації")
            return
        
        with open(pickle_path, 'rb') as f:
            dataset = pickle.load(f)
        
        X = dataset['features']
        y = dataset['labels']
        class_names = dataset['class_names']
        
        # Візуалізація середніх спектрів по класах
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, class_name in enumerate(class_names[:12]):
            if i >= len(axes):
                break
                
            class_indices = np.where(y == class_name)[0]
            class_samples = X[class_indices[:max_samples_per_class]]
            
            mean_spectrum = np.mean(class_samples, axis=0)
            
            fft_part = mean_spectrum[:-20]
            
            axes[i].plot(fft_part)
            axes[i].set_title(f'{class_name}\n({len(class_indices)} зразків)')
            axes[i].set_xlabel('Частотний бін')
            axes[i].set_ylabel('Magnitude (norm)')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'class_spectra_visualization.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Візуалізація: {plot_path}")


def create_fft_dataset(segments_dir, output_dir):
    """Основна функція створення FFT датасету"""
    
    generator = FFTDatasetGenerator()
    
    X, y, file_info = generator.process_audio_segments(segments_dir, output_dir)
    
    if len(X) > 0:
        generator.visualize_features(output_dir)
        print(f"\nДатасет готовий")
        return True
    else:
        print("Не вдалося створити датасет")
        return False


if __name__ == "__main__":
    segments_directory = "dataset-preparation/processed-segments"
    output_directory = "dataset-preparation/fft-dataset"
    
    if os.path.exists(segments_directory):
        success = create_fft_dataset(segments_directory, output_directory)
    else:
        print(f"Папка аудіосегментів не знайдена: {segments_directory}")