import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

shape = (1025, 97)
batchSize = 32

imageGenerator = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

train_dataset = imageGenerator.flow_from_directory(directory="../../MICC",
                                                   batch_size=batchSize,
                                                   target_size=shape,
                                                   subset="training",
                                                   color_mode="grayscale",
                                                   class_mode="binary")

validation_dataset = imageGenerator.flow_from_directory(directory="../../MICC",
                                                        batch_size=batchSize,
                                                        target_size=shape,
                                                        subset="validation",
                                                        color_mode="grayscale",
                                                        class_mode="binary")