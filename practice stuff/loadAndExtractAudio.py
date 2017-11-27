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

    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast

def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    features, labels = np.empty((0,187)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            print("file found:")
            mfccs, chroma, mel, contrast = extract_feature(fn)
            ext_features = np.hstack([mfccs,chroma,mel,contrast])
            print(ext_features.size)
            #print(np.vstack([features,ext_features]))
            features = np.vstack([features,ext_features])
            labels = np.append(labels, fn.split('-')[1].split('.')[0])
    return np.array(features), np.array(labels, dtype = np.int)
