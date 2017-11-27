import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram

def extract_feature(file_name):
    print(file_name)
    X, sample_rate = librosa.load(file_name)

    #compute spectral centroid
    cent = np.mean(librosa.feature.spectral_centroid(y=X, sr=sample_rate).T,axis=0)
    print("Spectral Centroid:")
    print(cent)

    #compute spectral contrast
    S = np.abs(librosa.stft(X))
    contrast = np.mean(librosa.feature.spectral_contrast(S=S, sr=sample_rate).T,axis=0)
    print("Spectral contrast:")
    print(contrast)

    #compute RMS energy time series
    rmsTimeSeries = np.mean(librosa.feature.rmse(y=X).T,axis=0)
    print("RMS time series:")
    print(rmsTimeSeries)

    #compute RMS energy spectrogram
    S, phase = librosa.magphase(librosa.stft(X))
    rmsSpectrogram = np.mean(librosa.feature.rmse(S=S).T,axis=0)
    print("RMS energy spectrogram:")
    print(rmsSpectrogram)

    return cent,contrast,rmsTimeSeries,rmsSpectrogram

def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    features, labels = np.empty((0,10)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            print("file found:")
            cent, contrast, rmsTimeSeries, rmsSpectrogram = extract_feature(fn)
            ext_features = np.hstack([cent,contrast,rmsTimeSeries,rmsSpectrogram])
            features = np.vstack([features,ext_features])
            labels = np.append(labels, fn.split('-')[1].split('.')[0])
    return np.array(features), np.array(labels, dtype = np.int)

def condenseLabels(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode
