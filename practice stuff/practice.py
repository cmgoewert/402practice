import wave, struct

print("hello box munchers")

audioFile = wave.open('Snare-1.wav', 'r')
numberOutput = open("output.csv", "w")
numberOutput.write("amplitude" + "\n")

length = audioFile.getnframes()
print(length)
for i in range(0,length):
    waveData = audioFile.readframes(1)
    data = struct.unpack("<h", waveData)
    numberOutput.write(str(data[0]) + "\n")

numberOutput.close()