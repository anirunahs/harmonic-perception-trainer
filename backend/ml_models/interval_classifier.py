import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import numpy as np
import os
import cv2
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class IntervalClassifier:
    def __init__(self, input_shape=(1025, 97, 1), num_classes=12):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
        self.class_names = [
            'minor_2nd', 'major_2nd', 'minor_3rd', 'major_3rd', 'perfect_4th',
            'tritone', 'perfect_5th', 'minor_6th', 'major_6th', 'minor_7th',
            'major_7th', 'perfect_8th'
        ]
        
        self.class_to_index = {name: idx for idx, name in enumerate(self.class_names)}
        self.index_to_class = {idx: name for idx, name in enumerate(self.class_names)}
    
    def create_model(self):
        """Створення CNN моделі"""
        model = keras.models.Sequential([
            keras.layers.Input(shape=self.input_shape),
            
            keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.2),
            
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.3),
            
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.4),
            
            keras.layers.GlobalAveragePooling2D(),
            
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),
            
            keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        optimizer = keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        self.model = model
        return model
        
    def prepare_data_generators(self, data_dir, validation_split=0.2, batch_size=16):
        """Підготовка генераторів даних з аугментацією"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=8,
            width_shift_range=0.08,
            height_shift_range=0.08,
            shear_range=0.05,
            zoom_range=0.08,
            brightness_range=[0.9, 1.1],
            horizontal_flip=False,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        validation_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='grayscale',
            subset='training',
            shuffle=True
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            data_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='grayscale',
            subset='validation',
            shuffle=False
        )
        
        return train_generator, validation_generator
    
    def train(self, data_dir, epochs=80, batch_size=32, validation_split=0.2, save_path='interval_model.h5'):
        """Тренування моделі"""
        print("ТРЕНУВАННЯ...")
        
        self.create_model()
        
        train_gen, val_gen = self.prepare_data_generators(
            data_dir, validation_split, batch_size
        )
        
        print(f"Параметрів моделі: {self.model.count_params():,}")
        print(f"Знайдено класи: {list(train_gen.class_indices.keys())}")
        print(f"Тренувальних зразків: {train_gen.samples}")
        print(f"Валідаційних зразків: {val_gen.samples}")
        print(f"Розмір батчу: {batch_size}")
        print(f"Кількість епох: {epochs}")
        
        callbacks_list = [
            ModelCheckpoint(
                save_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1,
                save_format='h5'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=10,
                min_lr=1e-8,
                verbose=1,
                cooldown=5
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            )
        ]
        
        steps_per_epoch = int(np.ceil(train_gen.samples / batch_size))
        validation_steps = int(np.ceil(val_gen.samples / batch_size))
        
        print(f"Кроків на епоху: {steps_per_epoch}")
        print(f"Валідаційних кроків: {validation_steps}")
        
        self.history = self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.model.load_weights(save_path)
        print(f"\nТренування завершено. Найкраща модель збережена: {save_path}")
        
        return self.history
    
    def evaluate_model(self, data_dir, batch_size=16):
        """Оцінка моделі"""
        print("ОЦІНКА МОДЕЛІ")
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
            data_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='grayscale',
            shuffle=False
        )
        
        results = self.model.evaluate(test_generator, verbose=1)
        
        print("Генерація прогнозів...")
        predictions = self.model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        
        accuracy = results[1]
        top3_accuracy = results[2] if len(results) > 2 else None
        
        print(f"\nРЕЗУЛЬТАТИ ОЦІНКИ:")
        print(f"Точність: {accuracy:.4f} ({accuracy*100:.2f}%)")
        if top3_accuracy:
            print(f"Top-3 точність: {top3_accuracy:.4f} ({top3_accuracy*100:.2f}%)")
        
        print("\nЗвіт по класах:")
        report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=self.class_names,
            output_dict=True
        )
        print(classification_report(true_classes, predicted_classes, target_names=self.class_names))
        
        return {
            'accuracy': accuracy,
            'top3_accuracy': top3_accuracy,
            'classification_report': report,
            'predictions': predictions,
            'true_classes': true_classes,
            'predicted_classes': predicted_classes
        }
    
    def predict_interval_from_audio(self, audio_path, confidence_threshold=0.6):
        """Прогнозування інтервалу з аудіофайлу"""
        try:
            spectrogram = self._audio_to_spectrogram(audio_path)
            
            if spectrogram is None:
                return {"error": "Помилка обробки аудіо"}
            
            input_data = np.expand_dims(spectrogram, axis=0)
            input_data = input_data / 255.0
            
            predictions = self.model.predict(input_data, verbose=0)
            predicted_probs = predictions[0]
            
            predicted_index = np.argmax(predicted_probs)
            confidence = float(predicted_probs[predicted_index])
            predicted_interval = self.class_names[predicted_index]
            
            if confidence < confidence_threshold:
                return {
                    "interval": "не розпізнано інтервал",
                    "confidence": confidence,
                    "all_predictions": {
                        self.class_names[i]: float(predicted_probs[i]) 
                        for i in range(len(self.class_names))
                    }
                }
            
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
                "top3_predictions": top3_predictions,
                "all_predictions": {
                    self.class_names[i]: float(predicted_probs[i]) 
                    for i in range(len(self.class_names))
                }
            }
            
        except Exception as e:
            return {"error": f"Помилка прогнозування: {str(e)}"}
    
    def _audio_to_spectrogram(self, audio_path):
        """Конвертація аудіо в спектрограму"""
        try:
            import librosa
            
            y, sr = librosa.load(audio_path, sr=22050)
            
            D = librosa.stft(y)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            
            ImageAudio = (S_db * 255).astype(np.uint8)
            
            resized_image = cv2.resize(
                ImageAudio, 
                (self.input_shape[1], self.input_shape[0]),
                interpolation=cv2.INTER_AREA
            )
            
            if len(resized_image.shape) == 2:
                resized_image = np.expand_dims(resized_image, axis=-1)
            
            return resized_image
            
        except Exception as e:
            print(f"Помилка конвертації аудіо: {e}")
            return None
    
    def save_model(self, filepath):
        """Збереження моделі та метаданих"""
        if self.model is None:
            print("Модель не створена")
            return False
        
        try:
            self.model.save(filepath)
            
            metadata = {
                'class_names': self.class_names,
                'class_to_index': self.class_to_index,
                'input_shape': self.input_shape,
                'num_classes': self.num_classes,
                'model_type': 'interval_classifier_cnn',
                'training_method': 'STFT_spectrogram',
                'architecture': 'Compact CNN (3 conv blocks)',
                'parameters': int(self.model.count_params())
            }
            
            metadata_path = filepath.replace('.h5', '_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"Модель збережена: {filepath}")
            print(f"Метадані збережені: {metadata_path}")
            return True
            
        except Exception as e:
            print(f"Помилка збереження: {e}")
            return False
    
    def load_model(self, filepath):
        """Завантаження моделі та метаданих"""
        try:
            self.model = keras.models.load_model(filepath)
            
            metadata_path = filepath.replace('.h5', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                self.class_names = metadata.get('class_names', self.class_names)
                self.class_to_index = metadata.get('class_to_index', self.class_to_index)
                self.input_shape = tuple(metadata.get('input_shape', self.input_shape))
                self.num_classes = metadata.get('num_classes', self.num_classes)
                
                self.index_to_class = {idx: name for idx, name in enumerate(self.class_names)}
            
            print(f"Модель завантажена: {filepath}")
            return True
            
        except Exception as e:
            print(f"Помилка завантаження: {e}")
            return False


def train_interval_classifier(spectrograms_dir, model_save_path, epochs=80):
    """Основна функція для тренування"""
    print("ЗАПУСК ТРЕНУВАННЯ")
    
    if not os.path.exists(spectrograms_dir):
        print(f"Папка зі спектрограмами не знайдена: {spectrograms_dir}")
        return False
    
    total_files = 0
    for root, dirs, files in os.walk(spectrograms_dir):
        total_files += len([f for f in files if f.endswith('.png')])
    
    print(f"Знайдено {total_files} спектрограм")
    
    if total_files < 1000:
        print("УВАГА: Мало даних для тренування")
        
    classifier = IntervalClassifier(input_shape=(1025, 97, 1))
    
    history = classifier.train(
        data_dir=spectrograms_dir,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        save_path=model_save_path
    )
    
    classifier.save_model(model_save_path)
    
    print(f"\nТренування завершено")
    print(f"Модель збережена: {model_save_path}")
    
    return True


if __name__ == "__main__":
    spectrograms_directory = "../dataset-preparation/spectrograms"
    model_output_path = "interval_model.h5"
    
    if os.path.exists(spectrograms_directory):
        success = train_interval_classifier(
            spectrograms_directory, 
            model_output_path, 
            epochs=80
        )
        
        if success:
            print("Модель готова для використання")
        else:
            print("Помилка тренування")
    else:
        print(f"Відсутні спектрограми")