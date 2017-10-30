import glob
import librosa
import numpy as np


clap_names = (glob.glob("C:/Users/lap55/Documents/Tech Stuff/402practice/train_claps/*.wav"))
# print(clap_names)

clap_length = len(clap_names)
# print(clap_length)


listOfClapCentroids = []
listOfClapContrasts = []
listOfClapRMSTimeSeries = []
listOfClapRMSSpectrogram = []

def getSpectralCentroidClaps():
    for i in range(clap_length):
        y, sr = librosa.load(clap_names[i])
        newCentroid = [clap_names[i], y, sr]
        listOfClapCentroids.append(newCentroid)
    return listOfClapCentroids

def getSpectralContrastClaps():
    for i in range(clap_length):
        y, sr = librosa.load(clap_names[i])
        S = np.abs(librosa.stft(y))
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        newContrast = [clap_names[i], y, sr, contrast]
        listOfClapContrasts.append(newContrast)
    return listOfClapContrasts

def getRMSEnergyTimeSeriesClaps():
    for i in range(clap_length):
        y, sr = librosa.load(clap_names[i])
        rmsTimeSeries = librosa.feature.rmse(y=y)
        newRMSTimeSeries = [clap_names[i], y, sr, rmsTimeSeries]
        listOfClapRMSTimeSeries.append(newRMSTimeSeries)
    return(listOfClapRMSTimeSeries)

def getRMSEnergySpectrogramClaps():
    for i in range(clap_length):
        y, sr = librosa.load(clap_names[i])
        S, phase = librosa.magphase(librosa.stft(y))
        rmsSpectrogram = librosa.feature.rmse(S=S)
        newRMSSpectrogram = [clap_names[i], y, sr, rmsSpectrogram]
        listOfClapRMSSpectrogram.append(newRMSSpectrogram)
    return(listOfClapRMSSpectrogram)

# print(getSpectralCentroidClaps())
# print(getSpectralContrastClaps())
# print(getRMSEnergySpectrogramClaps())
# print(getRMSEnergyTimeSeriesClaps())



