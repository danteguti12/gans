import matplotlib.pyplot as plt
from scipy.io import wavfile
from pydub import AudioSegment
from os import path
import os

#this script locates all your MP3 files in a specified directory
#the temp directory will store temporary wave files generated to produce the final plot

file_dir = "/mydirectory" #mp3 files directory
temp_dir = "/temp" #temporary wave file directory
plots_dir = "/plots" #plots directory

for file in os.listdir(file_dir):
	if file.endswith(".mp3"):

		src = file_dir + file
		dst = temp_dir + os.path.splitext(file)[0] + '.wav'
		sound = AudioSegment.from_mp3(src) 
		sound.export(dst, format="wav") #convert to wave file

		samplingFreq, signalData = wavfile.read(dst) #read wave file
		signalData = signalData[:,0] #select 0 for left channel 
		plt.specgram(signalData, Fs=samplingFreq) #plot spectrogram
		plt.savefig(plots_dir + os.path.splitext(file)[0] + '.png') #save file
		os.remove(dst) #remove temporary wave file
