import glob
import librosa


cymbal_names = (glob.glob("C:/Users/lap55/Documents/Tech Stuff/402practice/train_cymbals/*.wav"))
print(cymbal_names)

cymbal_length = len(cymbal_names)
print(cymbal_length)


listOfCymbalCentroids = []

for i in range(cymbal_length):
    y, sr = librosa.load(cymbal_names[i])
    newCentroid = [cymbal_names[i], y, sr]
    listOfCymbalCentroids.append(newCentroid)
    # print(cymbal_names[i])

centroidList = ''.join(str(e) for e in listOfCymbalCentroids)
print(centroidList)

