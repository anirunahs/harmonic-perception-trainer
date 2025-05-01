import numpy as np
import librosa
import cv2
import os
from glob import glob

# Major files
Path_for_Major_Spectrogram="../../MICC/Major"
isExist = os.path.exists(Path_for_Major_Spectrogram)
if not isExist:
    os.makedirs(Path_for_Major_Spectrogram)
    print("The new Major folder created!")

MajorAudioFiles = glob("../../musical-instrument-chord-classification/Major/*.wav")

for file in MajorAudioFiles:
    y, sr = librosa.load(file)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    ImageAudio = (S_db * 255).astype(np.uint8)

    idx = file.rfind("\\")
    filename = file[idx+1:]

    cv2.imwrite(Path_for_Major_Spectrogram+"/"+filename+".png", ImageAudio)
    print(filename)

