import matplotlib.pyplot as plt
from scipy.io import wavfile
from pydub import AudioSegment
from os import path
import os

for file in os.listdir("/home/danteguti12/Documents/pyprojects/Musicplot/mp3"):
	if file.endswith(".mp3"):
	
		src = "/home/danteguti12/Documents/pyprojects/Musicplot/mp3/" + file
		dst = "/home/danteguti12/Documents/pyprojects/Musicplot/wav/" + os.path.splitext(file)[0] + '.wav'
		sound = AudioSegment.from_mp3(src)
		sound.export(dst, format="wav")

		samplingFreq, signalData = wavfile.read(dst)
		signalData = signalData[:,0]
		plt.specgram(signalData, Fs=samplingFreq)
		plt.savefig("/home/danteguti12/Documents/pyprojects/Musicplot/graphs/" + os.path.splitext(file)[0] + '.png')
		os.remove(dst)
