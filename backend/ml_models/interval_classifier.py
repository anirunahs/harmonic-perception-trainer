import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

class FFTIntervalClassifier:
    def __init__(self, fft_size=2048, sample_rate=22050):
        self.fft_size = fft_size
        self.sample_rate = sample_rate
        self.freq_bins = fft_size // 2 + 1
        self.model = None
        self.scaler = StandardScaler()
        
        self.class_names = []
        self.class_to_index = {}
        self.num_classes = 0
        
        self.min_freq = 80
        self.max_freq = 4000
        self.freq_range = self._calculate_freq_range()
    
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
        peak_ratios = self._calculate_peak_ratios(peak_indices, freqs)
        features.extend(peak_ratios)
        
        return np.array(features)
    
    def _find_spectral_peaks(self, magnitude, prominence=0.1):
        """Знаходження піків в спектрі"""
        from scipy.signal import find_peaks
        
        peaks, _ = find_peaks(magnitude, prominence=prominence * np.max(magnitude))
        return peaks[:10]
    
    def _calculate_peak_ratios(self, peak_indices, freqs):
        """Розрахунок співвідношень між піками"""
        if len(peak_indices) < 2:
            return [0.0] * 10
        
        ratios = []
        base_freq = freqs[peak_indices[0]] if len(peak_indices) > 0 else 1.0
        
        for i, peak_idx in enumerate(peak_indices[:5]):
            ratio = freqs[peak_idx] / base_freq if base_freq > 0 else 0.0
            ratios.append(ratio)
        
        while len(ratios) < 10:
            ratios.append(0.0)
        
        return ratios[:10]
    
    def prepare_dataset_from_pickle(self, dataset_dir):
        """Завантаження датасету"""
        import pickle
        
        pickle_path = os.path.join(dataset_dir, 'fft_dataset.pkl')
        
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"Датасет не знайдено: {pickle_path}")
        
        print("ЗАВАНТАЖЕННЯ ДАТАСЕТУ")
        
        with open(pickle_path, 'rb') as f:
            dataset = pickle.load(f)
        
        X = dataset['features']
        y_labels = dataset['labels']
        
        self.class_names = sorted(list(set(y_labels)))
        self.num_classes = len(self.class_names)
        self.class_to_index = {name: idx for idx, name in enumerate(self.class_names)}
        
        y = np.array([self.class_to_index[label] for label in y_labels])
        
        print(f"Завантажено {len(X)} зразків")
        print(f"Розмірність ознак: {X.shape[1]}")
        print(f"Класи ({self.num_classes}):")
        
        for i, class_name in enumerate(self.class_names):
            count = np.sum(y == i)
            print(f"  {class_name}: {count} зразків")
        
        return X, y
    
    def create_model(self, input_dim):
        """Створення моделі"""
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            
            layers.BatchNormalization(),
            
            layers.Dense(512, kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            
            layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.4),
            
            layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            
            layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
        )
        
        self.model = model
        return model
    
    def train(self, dataset_dir, epochs=100, batch_size=32, validation_split=0.2, save_path='fft_interval_model.keras'):
        """Тренування моделі"""
        X, y = self.prepare_dataset_from_pickle(dataset_dir)
        
        X_scaled = self.scaler.fit_transform(X)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        print(f"\nТренувальний набір: {len(X_train)} зразків")
        print(f"Валідаційний набір: {len(X_val)} зразків")
        
        self.create_model(X.shape[1])
        
        print(f"Параметрів моделі: {self.model.count_params():,}")
        
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                save_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            )
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        try:
            self.model = keras.models.load_model(save_path)
            print(f"\nТренування завершено. Найкраща модель: {save_path}")
        except Exception as e:
            print(f"Помилка завантаження моделі: {e}")
        
        return self.history
    
    def evaluate(self, segments_dir):
        """Оцінка моделі"""
        X, y = self.prepare_dataset_from_segments(segments_dir)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        predicted_classes = np.argmax(predictions, axis=1)
        
        accuracy = np.mean(predicted_classes == y)
        
        print(f"\nТочність на тестових даних: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return accuracy
    
    def predict_interval_from_audio(self, audio_path):
        """Прогнозування інтервалу з аудіофайлу"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            features = self.extract_harmonic_features(audio)
            features_scaled = self.scaler.transform([features])
            
            predictions = self.model.predict(features_scaled, verbose=0)
            predicted_probs = predictions[0]
            
            predicted_index = np.argmax(predicted_probs)
            confidence = float(predicted_probs[predicted_index])
            predicted_interval = self.class_names[predicted_index]
            
            top3_indices = np.argsort(predicted_probs)[-3:][::-1]
            top3_predictions = [
                {
                    "interval": self.class_names[i],
                    "confidence": float(predicted_probs[i])
                }
                for i in top3_indices
            ]
            
            return {
                "interval": predicted_interval,
                "confidence": confidence,
                "top3_predictions": top3_predictions
            }
            
        except Exception as e:
            return {"error": f"Помилка прогнозування: {str(e)}"}
    
    def save_model_and_scaler(self, model_path, scaler_path):
        """Збереження моделі та scaler"""
        if self.model:
            self.model.save(model_path)
            
            import joblib
            joblib.dump(self.scaler, scaler_path)
            
            metadata = {
                'class_names': self.class_names,
                'class_to_index': self.class_to_index,
                'num_classes': int(self.num_classes),
                'fft_size': int(self.fft_size),
                'sample_rate': int(self.sample_rate),
                'freq_range': [int(self.freq_range[0]), int(self.freq_range[1])]
            }
            
            metadata_path = model_path.replace('.keras', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Модель збережена: {model_path}")
            print(f"Scaler збережено: {scaler_path}")
            print(f"Метадані збережені: {metadata_path}")


def train_fft_classifier(dataset_dir, model_path, scaler_path, epochs=100):
    """Основна функція для тренування"""
    print("ЗАПУСК ТРЕНУВАННЯ")
    
    classifier = FFTIntervalClassifier()
    
    history = classifier.train(
        dataset_dir=dataset_dir,
        epochs=epochs,
        batch_size=32,
        save_path=model_path
    )
    
    classifier.save_model_and_scaler(model_path, scaler_path)
    
    return classifier, history


if __name__ == "__main__":
    dataset_directory = "../dataset-preparation/fft-dataset"
    model_path = "fft_interval_model.keras"
    scaler_path = "fft_scaler.joblib"
    
    if os.path.exists(dataset_directory):
        pickle_file = os.path.join(dataset_directory, 'fft_dataset.pkl')
        if os.path.exists(pickle_file):
            classifier, history = train_fft_classifier(
                dataset_directory, 
                model_path, 
                scaler_path,
                epochs=100
            )
            
            print("Модель готова для використання")
        else:
            print(f"Датасет не знайдено: {pickle_file}")
    else:
        print(f"Папка датасету не знайдена: {dataset_directory}")