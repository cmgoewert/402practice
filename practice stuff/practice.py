import wave, struct, librosa

print("Starting analysis...")

#this part uses librosa
audio, sampleRate = librosa.load('Snare-1.wav');
print(audio)
print(sampleRate)

#this part uses the wav module

audioFile = wave.open('Snare-1.wav', 'r')
numberOutput = open("output.csv", "w")
numberOutput.write("amplitude" + "\n")

length = audioFile.getnframes()
print(length)
for i in range(0,length):
    waveData = audioFile.readframes(1)
    #print(str(waveData))
    #numberOutput.write(str(waveData) + "\n")
    data = struct.unpack("<h", waveData)
    numberOutput.write(str(data[0]) + "\n")
    print(int(data[0]))

numberOutput.close()