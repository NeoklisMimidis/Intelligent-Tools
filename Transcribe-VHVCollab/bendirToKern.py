import os
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

import librosa
import scipy

from librosa import onset
from librosa import feature
from librosa import beat
import librosa, librosa.display
from scipy import signal

import argparse


### Access wav file and 3 variables
#filename = "/home/kalohr/virtualEnvs/bendir/BNDR_SAMPLE_KM184.wav"
# filename = "/var/www/html/vhv/bendir.wav"
#top_number = nominator
#bottom_number = denominator 
#approximate_tempo = bendir_tempo


def bendir_to_score(filename, top_number = '4', bottom_number = '4', approximate_tempo = 60):


  #READING FILE
  bendir, fs = sf.read(filename)
  krn = open("/home/kalohr/virtualEnvs/bendir/bendir_server.krn", "w") # Path of the FINAL .krn file


  #CLEANING THE AUDIO FILE AND PREPARING THE NOTATION
  bendir, index = librosa.effects.trim(bendir)
  time_signature = int(top_number) / int(bottom_number)


  #WRITING THE STAVE
  krn.write("**kern\n")
  krn.write("*Ibdrum\n")

  #FINDING ONSETS
  o_env = librosa.onset.onset_strength(y=bendir, sr=fs)
  peak_frames = librosa.util.peak_pick(o_env, pre_max=7, post_max=7, pre_avg=7, post_avg=7, delta=5, wait=10)
  peak_samples = librosa.frames_to_samples(peak_frames, hop_length=512, n_fft=None)


  #ESTIMATING TEMPO
  tempo = librosa.feature.rhythm.tempo(y=bendir, onset_envelope = o_env, start_bpm = approximate_tempo, sr=fs)
  tempo_int = np.round(tempo)

  krn.write(f"*MM{str(int(tempo))}\n")
  krn.write("*stria3\n")
  krn.write("*clefX\n")
  a = '*M'+top_number+'/'+bottom_number
  krn.write(a)
  krn.write("\n")
  krn.write("!!!OMD: [quarter]=")
  krn.write(str(int(tempo)))
  krn.write("\n")
  

  #FINDING BEATS
  _, beats = librosa.beat.beat_track(y=bendir, onset_envelope=o_env, sr=fs,start_bpm = tempo)
  beat_samples = librosa.frames_to_samples(beats, hop_length=512, n_fft=None)

  

  #TO COUNT THE METRES
  metre = 1


  #FINDING THE SOUNDS
  prevsound = 'aR/'
  sound=[]
  for i in range(len(peak_samples)-1):
    note = bendir[(peak_samples[i]-1700):(peak_samples[i+1]-1700)]
    length = len(note)

    #WINDOWING
    length = int(length/4)
    zeros = np.zeros(length*3)
    window = signal.windows.blackman(length)
    window = np.append(window, zeros)
    note = note*window


    #COMPUTING DENSITY
    f, S = scipy.signal.periodogram(note, fs, scaling='density')
    S = S/np.max(S)


    #FIND THE NOTES
    peaks0_250, _ = scipy.signal.find_peaks(S[0:250], height = 0.1)
    peaks250_500, _ = scipy.signal.find_peaks(S[250:500], height = 0.1)
    peaks500_750, _ = scipy.signal.find_peaks(S[550:750], height = 0.1)
    peaks750_1000, _ = scipy.signal.find_peaks(S[750:1000], height = 0.1)
    allpeaks = len(peaks0_250)+ len(peaks250_500)+len(peaks500_750)+len(peaks750_1000)
    if allpeaks == 0:
      allpeaks = 1
    perpeaks1 = len(peaks0_250)/ allpeaks
    perpeaks2 = len(peaks250_500)/ allpeaks
    perpeaks3 = len(peaks500_750)/allpeaks
    perpeaks4 = len(peaks750_1000)/allpeaks
    perpeaks1_2 = perpeaks1+perpeaks2
    perpeaks3_4 = perpeaks3+perpeaks4
    perpeaks2_3 = perpeaks2+perpeaks3
    perpeaks1_4 = perpeaks1+perpeaks4


    #SELECTING KE, TE
    if perpeaks2_3 >= perpeaks1_4:
      magnitude_spectrum = np.abs(np.fft.fft(note))
      band_magnitude_spectrum = magnitude_spectrum[600:1000]
      geometric_mean = np.exp(np.mean(np.log(band_magnitude_spectrum + 1e-10)))
      arithmetic_mean = np.mean(band_magnitude_spectrum)
      flatness600_1000 = geometric_mean**2 / (arithmetic_mean**2 + 1e-10)
      if flatness600_1000<0.45:
        sound.append("ccR") #te
      else:
        sound.append("bR") #ke


    #SELECTING DUM, PA, PE
    elif perpeaks1 > 0.75:
      magnitude_spectrum = np.abs(np.fft.fft(note))
      geometric_mean = np.exp(np.mean(np.log(magnitude_spectrum + 1e-10)))
      arithmetic_mean = np.mean(magnitude_spectrum)
      flatness = geometric_mean**2 / (arithmetic_mean**2 + 1e-10)
      if allpeaks >6:
        if flatness>29 and flatness<=34:
          sound.append("aR") #pe
          prevsound = 'aR'
        elif prevsound == 'aR':
          sound.append("gR") #pa
          prevsound = 'gR' #pe
        else:
          sound.append("aR") #pe
          prevsound = 'aR'
      else:
        sound.append("dR") #dum

    elif perpeaks1_2 >= perpeaks3_4:
      sound.append("ccR") #te


    #SELECTING SNAP
    elif perpeaks3_4 > perpeaks1_2:
      sound.append("ddR") #snap


  #FINDING THE VALUES
  beat = 0
  metre = 1

  #CHECKING HOW MANY ONSETS EVERY BEAT HAVE
  for i in range(len(beat_samples)-1):

    onsets = [0,0,0,0]
    onset_number = [0,0,0,0]
    beat_start = beat_samples[i]-3500
    beat_end = beat_samples[i+1]-3500
    quarter_time = beat_end - beat_start
    for y in range(len(peak_samples)):
      if peak_samples[y] > beat_start and peak_samples[y] < beat_start+(quarter_time/4):
        onsets[0] = 1
        onset_number[0] = y
      if peak_samples[y] > beat_start + (quarter_time/4) and peak_samples[y] < beat_start+(quarter_time/2):
        onsets[1] = 1
        onset_number[1] = y
      if peak_samples[y] > beat_start + (quarter_time/2) and peak_samples[y] < beat_end - (quarter_time/4):
        onsets[2] = 1
        onset_number[2] = y
      if peak_samples[y] > beat_end - (quarter_time/4) and peak_samples[y] < beat_end:
        onsets[3] = 1
        onset_number[3] = y


    #CHECKING FOR TRIPLETS
    if onsets == [1,1,1,0]:
      triplets = False
      if peak_samples[onset_number[1]] > beat_start+ 3000 + (quarter_time/3) and peak_samples[onset_number[2]]> beat_end+3000 - (quarter_time/3):
        triplets = True


    #WRITING THE STAVE
    if onsets == [0,0,0,0] :
      krn.write("4r")
      krn.write("\n")
    elif onsets == [1,0,0,0]:
      krn.write ("4"+sound[onset_number[0]])
      krn.write("\n")
    elif onsets == [0,1,0,0]:
      krn.write("16r")
      krn.write("\n")
      krn.write("8."+sound[onset_number[1]])
      krn.write("\n")
    elif onsets == [0,0,1,0]:
      krn.write("8r")
      krn.write("\n")
      krn.write("8"+sound[onset_number[2]])
      krn.write("\n")
    elif onsets == [0,0,0,1]:
      krn.write('8.r')
      krn.write("\n")
      krn.write("16"+sound[onset_number[3]])
      krn.write("\n")
    elif onsets == [1,0,1,0]:
      krn.write("8"+sound[onset_number[0]])
      krn.write("\n")
      krn.write("8"+sound[onset_number[2]])
      krn.write("\n")
    elif onsets == [1,1,0,0]:
      krn.write("16"+sound[onset_number[0]])
      krn.write("\n")
      krn.write("8."+sound[onset_number[1]])
      krn.write("\n")
    elif onsets == [1,0,0,1]:
      krn.write("8."+sound[onset_number[0]])
      krn.write("\n")
      krn.write("16"+sound[onset_number[3]])
      krn.write("\n")
    elif onsets == [0,1,0,1]:
      krn.write("16r")
      krn.write("\n")
      krn.write("8"+sound[onset_number[1]])
      krn.write("\n")
      krn.write("16"+sound[onset_number[3]])
      krn.write("\n")
    elif onsets ==[0,0,1,1]:
      krn.write('8r')
      krn.write("\n")
      krn.write("16"+sound[onset_number[2]])
      krn.write("\n")
      krn.write("16"+sound[onset_number[3]])
      krn.write("\n")
    elif onsets == [1,1,1,0] and triplets == False:
      krn.write("16"+sound[onset_number[0]])
      krn.write("\n")
      krn.write("16"+sound[onset_number[1]])
      krn.write("\n")
      krn.write("8"+sound[onset_number[2]])
      krn.write("\n")
    elif onsets == [1,1,1,0] and triplets == True:
      krn.write("12"+sound[onset_number[0]])
      krn.write("\n")
      krn.write("12"+sound[onset_number[1]])
      krn.write("\n")
      krn.write("12"+sound[onset_number[2]])
      krn.write("\n")
    elif onsets == [1,1,0,1]:
      krn.write("16"+sound[onset_number[0]])
      krn.write("\n")
      krn.write("8"+sound[onset_number[1]])
      krn.write("\n")
      krn.write("16"+sound[onset_number[3]])
      krn.write("\n")
    elif onsets == [1,0,1,1]:
      krn.write("8"+sound[onset_number[0]])
      krn.write("\n")
      krn.write("16"+sound[onset_number[2]])
      krn.write("\n")
      krn.write("16"+sound[onset_number[3]])
      krn.write("\n")
    elif onsets == [0,1,1,1]:
      krn.write('16r')
      krn.write("\n")
      krn.write("16"+sound[onset_number[1]])
      krn.write("\n")
      krn.write("16"+sound[onset_number[2]])
      krn.write("\n")
      krn.write("16"+sound[onset_number[3]])
      krn.write("\n")
    elif onsets == [1,1,1,1]:
      krn.write("16"+sound[onset_number[0]])
      krn.write("\n")
      krn.write("16"+sound[onset_number[1]])
      krn.write("\n")
      krn.write("16"+sound[onset_number[2]])
      krn.write("\n")
      krn.write("16"+sound[onset_number[3]])
      krn.write("\n")
    beat +=1

    #CHANGING METRE
    if beat == 4:
      krn.write("="+str(metre))
      krn.write("\n")
      metre+=1
      beat = 0


  #WRITING THE END OF THE STAVE
  krn.write("==\n")
  krn.write("*-\n")
  krn.write("!!!filter: autobeam")

  krn.close()
  print(krn.read)
  
  
# bendir_to_score('BNDR_SAMPLE_KM184.wav', '4','4', 60)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Converts a bendir file to a musical score.")

    # Required argument: filename
    parser.add_argument("filename", type=str, help="Path to the bendir file to be processed.")

    # Optional arguments
    parser.add_argument("--top_number", type=str, default='4', help="Top number of the time signature. Default is 4.")
    parser.add_argument("--bottom_number", type=str, default='4', help="Bottom number of the time signature. Default is 4.")
    parser.add_argument("--approximate_tempo", type=int, default=60, help="Approximate tempo in beats per minute. Default is 60.")

    args = parser.parse_args()

    # Call the function with parsed arguments
    bendir_to_score(args.filename, args.top_number, args.bottom_number, args.approximate_tempo)