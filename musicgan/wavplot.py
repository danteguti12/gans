import matplotlib.pyplot as plt
from scipy.io import wavfile
from pydub import AudioSegment
from os import path
import os

file_dir = "/mydirectory"
temp_dir = "/temp"
plots_dir = "/plots"

for file in os.listdir(file_dir):
	if file.endswith(".mp3"):
	
		src = file_dir + file
		dst = temp_dir + os.path.splitext(file)[0] + '.wav'
		sound = AudioSegment.from_mp3(src)
		sound.export(dst, format="wav")

		samplingFreq, signalData = wavfile.read(dst)
		signalData = signalData[:,0]
		plt.specgram(signalData, Fs=samplingFreq)
		plt.savefig(plots_dir + os.path.splitext(file)[0] + '.png')
		os.remove(dst)
