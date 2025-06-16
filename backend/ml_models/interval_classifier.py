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