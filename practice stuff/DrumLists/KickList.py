import glob
import librosa


kick_names = (glob.glob("C:/Users/lap55/Documents/Tech Stuff/402practice/train_kicks/*.wav"))
# print(kick_names)

kick_length = len(kick_names)
# print(length)


listOfKickCentroids = []

for i in range(length):
    y, sr = librosa.load(kick_names[i])
    newCentroid = [kick_names[i], y, sr]
    listOfKickCentroids.append(newCentroid)
    # print(listOfCentroids)

centroidList = ''.join(str(e) for e in listOfKickCentroids)
print(centroidList)

