import numpy as np
import librosa
import tensorflow as tf
import joblib
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class FFTModelTester:
    def __init__(self, model_path, scaler_path, dataset_dir):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.dataset_dir = dataset_dir
        
        self.model = tf.keras.models.load_model(model_path)
        print(f"Модель завантажена: {model_path}")
        
        self.scaler = joblib.load(scaler_path)
        print(f"Scaler завантажено: {scaler_path}")
        
        self.fft_size = 2048
        self.sample_rate = 22050
        self.min_freq = 80
        self.max_freq = 4000
        self.freq_range = self._calculate_freq_range()
        
        self.class_names = self._load_class_names_from_dataset()
        print(f"Класи з датасету: {self.class_names}")
    
    def _load_class_names_from_dataset(self):
        """Завантаження класів з датасету"""
        import pickle
        
        dataset_path = os.path.join(self.dataset_dir, 'fft_dataset.pkl')
        
        if os.path.exists(dataset_path):
            try:
                with open(dataset_path, 'rb') as f:
                    dataset = pickle.load(f)
                
                labels = dataset['labels']
                unique_labels = sorted(list(set(labels)))
                
                print(f"Завантажено класи з датасету: {unique_labels}")
                return unique_labels
                
            except Exception as e:
                print(f"Помилка завантаження класів з датасету: {e}")
        
        return [
            'major_2nd', 'major_3rd', 'major_6th', 'major_7th',
            'minor_2nd', 'minor_3rd', 'minor_6th', 'minor_7th',
            'perfect_4th', 'perfect_5th', 'perfect_8th', 'tritone'
        ]
    
    def _calculate_freq_range(self):
        """Розрахунок індексів частотного діапазону"""
        freq_bins = self.fft_size // 2 + 1
        freqs = np.fft.fftfreq(self.fft_size, 1/self.sample_rate)[:freq_bins]
        min_idx = np.argmax(freqs >= self.min_freq)
        max_idx = np.argmax(freqs >= self.max_freq)
        if max_idx == 0:
            max_idx = len(freqs)
        return min_idx, max_idx
    
    def extract_harmonic_features(self, audio):
        """Виділення гармонічних ознак"""
        if len(audio) < self.fft_size:
            audio = np.pad(audio, (0, self.fft_size - len(audio)))
        elif len(audio) > self.fft_size:
            start = (len(audio) - self.fft_size) // 2
            audio = audio[start:start + self.fft_size]
        
        windowed_audio = audio * np.hanning(len(audio))
        
        fft = np.fft.fft(windowed_audio, n=self.fft_size)
        magnitude = np.abs(fft[:self.fft_size // 2 + 1])
        
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

    def test_validation_split(self):
        """Тестування на валідаційному наборі з правильними класами"""
        import pickle
        
        splits_path = os.path.join(self.dataset_dir, 'fft_dataset.npz')
        
        if os.path.exists(splits_path):
            print("ТЕСТУВАННЯ НА ГОТОВОМУ ТЕСТОВОМУ НАБОРІ")
            print("=" * 50)
            
            try:
                data = np.load(splits_path)
                print(f"Ключі в архіві: {list(data.keys())}")
                
                if 'X_test' in data.keys():
                    X_test = data['X_test']
                    y_test = data['y_test']
                else:
                    print("Ключ 'X_test' не знайдено. Створюємо тестовий набір з основного датасету...")
                    return self._create_and_test_split()
                
            except Exception as e:
                print(f"Помилка завантаження splits: {e}")
                return self._create_and_test_split()
        else:
            print("Файл splits не знайдено. Створюємо тестовий набір...")
            return self._create_and_test_split()
        
        X_test_scaled = self.scaler.transform(X_test)
        
        predictions = self.model.predict(X_test_scaled, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        y_test_names = [self.class_names[i] for i in y_test]
        predicted_names = [self.class_names[i] for i in predicted_classes]
        
        accuracy = np.mean(predicted_classes == y_test)
        print(f"\nТочність на тестовому наборі: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\nЗвіт по класах:")
        print(classification_report(y_test_names, predicted_names))
        
        self._plot_confusion_matrix(y_test_names, predicted_names)
        
        return accuracy
    
    def _create_and_test_split(self):
        """Створення тестового набору з основного датасету"""
        import pickle
        from sklearn.model_selection import train_test_split
        
        dataset_path = os.path.join(self.dataset_dir, 'fft_dataset.pkl')
        
        if not os.path.exists(dataset_path):
            print(f"Основний датасет не знайдено: {dataset_path}")
            return None
        
        print("СТВОРЕННЯ ТЕСТОВОГО НАБОРУ З ОСНОВНОГО ДАТАСЕТУ")
        print("=" * 50)
        
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        X = dataset['features']
        y_labels = dataset['labels']
        
        y = np.array([self.class_names.index(label) for label in y_labels])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Тестовий набір: {len(X_test)} зразків")
        
        X_test_scaled = self.scaler.transform(X_test)
        
        predictions = self.model.predict(X_test_scaled, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        y_test_names = [self.class_names[i] for i in y_test]
        predicted_names = [self.class_names[i] for i in predicted_classes]
        
        accuracy = np.mean(predicted_classes == y_test)
        print(f"\nТочність на тестовому наборі: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\nЗвіт по класах:")
        print(classification_report(y_test_names, predicted_names))
        
        self._plot_confusion_matrix(y_test_names, predicted_names)
        
        return accuracy
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Побудова матриці помилок"""
        cm = confusion_matrix(y_true, y_pred, labels=self.class_names)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Матриця помилок')
        plt.xlabel('Прогноз')
        plt.ylabel('Справжній клас')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Матрицю помилок збережено: confusion_matrix.png")
    
    def debug_class_mapping(self):
        """Дебаг відповідності класів"""
        print("ДЕБАГ ВІДПОВІДНОСТІ КЛАСІВ")
        print("=" * 40)
        
        print("Порядок класів у моделі:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {i}: {class_name}")
        
        try:
            import pickle
            dataset_path = os.path.join(self.dataset_dir, 'fft_dataset.pkl')
            
            if os.path.exists(dataset_path):
                with open(dataset_path, 'rb') as f:
                    dataset = pickle.load(f)
                
                X = dataset['features'][:5]
                y_labels = dataset['labels'][:5]
                
                X_scaled = self.scaler.transform(X)
                predictions = self.model.predict(X_scaled, verbose=0)
                
                print(f"\nПриклади прогнозів:")
                for i in range(len(X)):
                    true_class = y_labels[i]
                    pred_index = np.argmax(predictions[i])
                    pred_class = self.class_names[pred_index]
                    confidence = predictions[i][pred_index]
                    
                    print(f"  Справжній: {true_class}, Прогноз: {pred_class}, Впевненість: {confidence:.3f}")
            else:
                print("Основний датасет не знайдено")
                
        except Exception as e:
            print(f"Помилка при дебагу: {e}")


def main():
    """Основна функція тестування"""
    
    model_path = "ml_models/fft_interval_model.keras"
    scaler_path = "ml_models/fft_scaler.joblib"
    dataset_dir = "../dataset-preparation/fft-dataset"
    
    if not os.path.exists(model_path):
        print(f"Модель не знайдена: {model_path}")
        return
    
    if not os.path.exists(scaler_path):
        print(f"Scaler не знайдено: {scaler_path}")
        return
    
    if not os.path.exists(dataset_dir):
        print(f"Датасет не знайдено: {dataset_dir}")
        return
    
    tester = FFTModelTester(model_path, scaler_path, dataset_dir)
    
    print("МОДЕЛЬ ЗАВАНТАЖЕНА")
    print(f"Кількість класів: {len(tester.class_names)}")
    
    tester.debug_class_mapping()
    
    tester.test_validation_split()
    

if __name__ == "__main__":
    main()