import glob
import librosa
import tensorflow as tf
import numpy as np


snare_names = (glob.glob("C:/Users/lap55/Documents/Tech Stuff/402practice/train_snares/*.wav"))
# print(snare_names)

snare_length = len(snare_names)
# print(snare_length)


listOfSnareCentroids = []
listOfSnareContrasts = []
listOfSnareRMSTimeSeries = []
listOfSnareRMSSpectrogram = []
listOfCents = np.array([])


def getSpectralCentroidsSnares():
    for i in range(snare_length):
        y, sr = librosa.load(snare_names[i])
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        newCentroid = [snare_names[i], y, sr, cent]
        listOfCents.append(cent)
        listOfSnareCentroids.append(newCentroid)
    return(listOfSnareCentroids)

def getSpectralContrastSnares():
    for i in range(snare_length):
        y, sr = librosa.load(snare_names[i])
        S = np.abs(librosa.stft(y))
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        newContrast = [snare_names[i], y, sr, contrast]
        listOfSnareContrasts.append(newContrast)
    return(listOfSnareContrasts)

def getRMSEnergyTimeSeriesSnares():
    for i in range(snare_length):
        y, sr = librosa.load(snare_names[i])
        rmsTimeSeries = librosa.feature.rmse(y=y)
        newRMSTimeSeries = [snare_names[i], y, sr, rmsTimeSeries]
        listOfSnareRMSTimeSeries.append(newRMSTimeSeries)
    return(listOfSnareRMSTimeSeries)

def getRMSEnergySpectrogramSnares():
    for i in range(snare_length):
        y, sr = librosa.load(snare_names[i])
        S, phase = librosa.magphase(librosa.stft(y))
        rmsSpectrogram = librosa.feature.rmse(S=S)
        newRMSSpectrogram = [snare_names[i], y, sr, rmsSpectrogram]
        listOfSnareRMSSpectrogram.append(newRMSSpectrogram)
    return(listOfSnareRMSSpectrogram)


# print(getSpectralCentroidsSnares())
# print(getSpectralContrastSnares())
# print(getRMSEnergyTimeSeriesSnares())
# print(getRMSEnergySpectrogramSnares())
