import glob
import librosa


clap_names = (glob.glob("C:/Users/lap55/Documents/Tech Stuff/402practice/train_claps/*.wav"))
print(clap_names)

clap_length = len(clap_names)
print(clap_length)


listOfClapCentroids = []

for i in range(clap_length):
    y, sr = librosa.load(clap_names[i])
    newCentroid = [clap_names[i], y, sr]
    listOfClapCentroids.append(newCentroid)
    # print(clap_names[i])

centroidList = ''.join(str(e) for e in listOfClapCentroids)
print(centroidList)

