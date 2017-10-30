import glob
import librosa
import numpy as np

kick_names = (glob.glob("C:/Users/lap55/Documents/Tech Stuff/402practice/train_kicks/*.wav"))
# print(kick_names)

kick_length = len(kick_names)
# print(length)


listOfKickCentroids = []
listOfKickContrasts = []
listOfKickRMSTimeSeries = []
listOfKickRMSSpectrogram = []

def getSpectralCentroidKicks():
    for i in range(kick_length):
        y, sr = librosa.load(kick_names[i])
        newCentroid = [kick_names[i], y, sr]
        listOfKickCentroids.append(newCentroid)
    return listOfKickCentroids

def getSpectralContrastKicks():
    for i in range(kick_length):
        y, sr = librosa.load(kick_names[i])
        S = np.abs(librosa.stft(y))
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        newContrast = [kick_names[i], y, sr, contrast]
        listOfKickContrasts.append(newContrast)
    return(listOfKickContrasts)

def getRMSEnergyTimeSeriesKicks():
    for i in range(kick_length):
        y, sr = librosa.load(kick_names[i])
        rmsTimeSeries = librosa.feature.rmse(y=y)
        newRMSTimeSeries = [kick_names[i], y, sr, rmsTimeSeries]
        listOfKickRMSTimeSeries.append(newRMSTimeSeries)
    return(listOfKickRMSTimeSeries)

def getRMSEnergySpectrogramKicks():
    for i in range(kick_length):
        y, sr = librosa.load(kick_names[i])
        S, phase = librosa.magphase(librosa.stft(y))
        rmsSpectrogram = librosa.feature.rmse(S=S)
        newRMSSpectrogram = [kick_names[i], y, sr, rmsSpectrogram]
        listOfKickRMSSpectrogram.append(newRMSSpectrogram)
    return(listOfKickRMSSpectrogram)

# print(getSpectralCentroidKicks())
# print(getSpectralContrastKicks())
# print(getRMSEnergySpectrogramKicks())
# print(getRMSEnergyTimeSeriesKicks())

