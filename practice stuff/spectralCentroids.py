import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display




y, sr = librosa.load('Bass-Drum-1.wav')
cent = librosa.feature.spectral_centroid(y=y, sr=sr)

S, phase = librosa.magphase(librosa.stft(y=y))
librosa.feature.spectral_centroid(S=S)

y, sr = librosa.load('Bass-Drum-1.wav')
if_gram, D = librosa.ifgram(y)
librosa.feature.spectral_centroid(S=np.abs(D), freq=if_gram)

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.semilogy(cent.T, label='Spectral centroid')
plt.ylabel('Hz')
plt.xticks([])
plt.xlim([0, cent.shape[-1]])
plt.legend()
plt.subplot(2, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time')
plt.title('log Power spectrogram')
plt.tight_layout()