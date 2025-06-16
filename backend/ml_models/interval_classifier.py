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
        """Створення глибокої CNN моделі"""
        model = keras.models.Sequential([
            keras.layers.Input(shape=self.input_shape),
            
            keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.25),
            
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.25),
            
            keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.3),
            
            keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.3),
            
            keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same'),
            keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.4),
            
            keras.layers.GlobalAveragePooling2D(),
            
            keras.layers.Dense(2048, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            
            keras.layers.Dense(512, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        optimizer = keras.optimizers.Adam(
            learning_rate=0.0001,
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