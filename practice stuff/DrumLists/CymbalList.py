import glob
import librosa
import numpy as np


cymbal_names = (glob.glob("C:/Users/lap55/Documents/Tech Stuff/402practice/train_cymbals/*.wav"))
# print(cymbal_names)

cymbal_length = len(cymbal_names)
# print(cymbal_length)


listOfCymbalCentroids = []
listOfCymbalContrasts = []
listOfCymbalRMSTimeSeries = []
listOfCymbalRMSSpectrogram = []

def getSpectralCentroidCymbals():
    for i in range(cymbal_length):
        y, sr = librosa.load(cymbal_names[i])
        newCentroid = [cymbal_names[i], y, sr]
        listOfCymbalCentroids.append(newCentroid)
    return listOfCymbalCentroids

def getSpectralContrastCymbals():
    for i in range(cymbal_length):
        y, sr = librosa.load(cymbal_names[i])
        S = np.abs(librosa.stft(y))
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        newContrast = [cymbal_names[i], y, sr, contrast]
        listOfCymbalContrasts.append(newContrast)
    return(listOfCymbalContrasts)

def getRMSEnergyTimeSeriesCymbals():
    for i in range(cymbal_length):
        y, sr = librosa.load(cymbal_names[i])
        rmsTimeSeries = librosa.feature.rmse(y=y)
        newRMSTimeSeries = [cymbal_names[i], y, sr, rmsTimeSeries]
        listOfCymbalRMSTimeSeries.append(newRMSTimeSeries)
    return (listOfCymbalRMSTimeSeries)

def getRMSEnergySpectrogramCymbals():
    for i in range(cymbal_length):
        y, sr = librosa.load(cymbal_names[i])
        S, phase = librosa.magphase(librosa.stft(y))
        rmsSpectrogram = librosa.feature.rmse(S=S)
        newRMSSpectrogram = [cymbal_names[i], y, sr, rmsSpectrogram]
        listOfCymbalRMSSpectrogram.append(newRMSSpectrogram)
    return (listOfCymbalRMSSpectrogram)

# print(getSpectralCentroidCymbals())
# print(getSpectralContrastCymbals())
# print(getRMSEnergySpectrogramCymbals())
# print(getRMSEnergyTimeSeriesCymbals())