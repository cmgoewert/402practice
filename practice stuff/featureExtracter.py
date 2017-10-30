import librosa
import numpy as np

def getSpectralCentroid(files):
    results = []
    for i in range(len(files)):
        y, sr = librosa.load(files[i])
        newCentroid = [files[i], y, sr]
        results.append(newCentroid)
    return results

def getSpectralContrast(files):
    results = []
    for i in range(len(files)):
        y, sr = librosa.load(files[i])
        S = np.abs(librosa.stft(y))
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        newContrast = [files[i], y, sr, contrast]
        results.append(newContrast)
    return results

def getRMSEnergyTimeSeriesClaps(files):
    results = []
    for i in range(len(files)):
        y, sr = librosa.load(files[i])
        rmsTimeSeries = librosa.feature.rmse(y=y)
        newRMSTimeSeries = [files[i], y, sr, rmsTimeSeries]
        results.append(newRMSTimeSeries)
    return results

def getRMSEnergySpectrogramClaps(files):
    results = []
    for i in range(len(files)):
        y, sr = librosa.load(files[i])
        S, phase = librosa.magphase(librosa.stft(y))
        rmsSpectrogram = librosa.feature.rmse(S=S)
        newRMSSpectrogram = [files[i], y, sr, rmsSpectrogram]
        results.append(newRMSSpectrogram)
    return(results)