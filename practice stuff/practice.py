import wave, struct, librosa

print("Starting analysis...")


#this part uses librosa stuff
audio, sampleRate = librosa.load('Snare-1.wav');
#print(audio)
#print(sampleRate)



tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sampleRate)
#print('Estimated tempo: {:.2f} beats per minute'.format(tempo))
beat_times = librosa.frames_to_time(beat_frames, sr=sampleRate)
#print(beat_times)

y_orig, sr_orig = librosa.load('Splashin Everywhere (Master).wav', sr=None)
print(len(y_orig), sr_orig)

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
    #print(int(data[0]))

numberOutput.close()