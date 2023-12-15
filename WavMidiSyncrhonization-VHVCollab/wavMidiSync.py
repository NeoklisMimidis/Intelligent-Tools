# coding: utf-8
# test
import librosa
import os
from midi2audio import FluidSynth
import numpy as np
# test
#os.system("/usr/bin/ffmpeg -y -i /var/www/html/vhv/syncWav.webm -qscale 0  /var/www/html/vhv/syncWav.wav")

#### Access midi and wav files
midifile = "/var/www/html/vhv/syncMidi2.mid"
file2 = "/var/www/html/vhv/syncWav.wav"

#### Load recorded wav file
x_2, fs = librosa.load(file2,sr=None)


#### Create wav from midi
fsynth = FluidSynth(sample_rate=fs)
fsynth.midi_to_audio(midifile, "/var/www/html/vhv/tmp_file1.wav")
file1 = "/var/www/html/vhv/tmp_file1.wav"

#### Load created wav file
x_1, fs = librosa.load(file1,sr=None)

#### DEBUG ############################################################
#f_x1 = open("/var/www/html/vhv/debug/x1.csv", "w");
#np.savetxt(f_x1, x_1, delimiter=" ") ####
#f_x2 = open("/var/www/html/vhv/debug/x2.csv", "w");
#np.savetxt(f_x2, x_2, delimiter=" ") #### 


#### Extract Chroma Features
hop_length = 1024
x_1_chroma = librosa.feature.chroma_cqt(y=x_1, sr=fs, hop_length=hop_length)
x_2_chroma = librosa.feature.chroma_cqt(y=x_2, sr=fs, hop_length=hop_length)


#### DEBUG ############################################################
#f_chr1 = open("/var/www/html/vhv/debug/x1_chroma.csv", "w");
#np.savetxt(f_chr1, x_1_chroma, delimiter=" ") ####
#f_chr2 = open("/var/www/html/vhv/debug/x2_chroma.csv", "w");
#np.savetxt(f_chr2, x_2_chroma, delimiter=" ") #### 


#### Align Chroma Sequences
D, wp = librosa.sequence.dtw(X=x_1_chroma, Y=x_2_chroma, metric='cosine')
wp_1 = wp


#### DEBUG ############################################################
#f_wp = open("/var/www/html/vhv/debug/wp_created.csv", "w");
#np.savetxt(f_wp, wp_1, delimiter=" ") ####

#### Discarding path in the beginning of recording

diff1=np.zeros(len(wp),dtype=int)
i = len(wp)-1
while i >= 1:
    diff1[len(wp)-1-i] = wp[i-1,0]-wp[i,0]
    i = i - 1

forward_chk = 60
repetitions = 40
k = 0
result = False
while result == False:
    while k < len(wp)-1:
        if diff1[k] > 0:
            count = 0
            m = k
            while m < k+forward_chk:
                if diff1[m] == 0:
                    count = count+1
                m = m + 1
            #print ("k=",k,"and count=",count)
            if count <= repetitions:
                start_at = len(wp)-1-k
                k=len(wp)-1
        k=k+1
    result = True

wp = wp[0:start_at]



#### Discarding path at the end of recording
diff2=np.zeros(len(wp),dtype=int)
i = len(wp)-1
while i >= 1:
    diff2[len(wp)-1-i] = wp[i-1,0]-wp[i,0]
    i = i - 1



backward_chk = 60
repetitions = 20
p = len(diff2)-1
result = False
while result == False:
    while p > backward_chk:
        if diff2[p] > 0:
            count = 0
            n = p
            while n > p - backward_chk:
                if diff2[n] == 0:
                    count = count+1
                n = n - 1
            #print ("p=",p,"and count=",count)
            if count <= repetitions:
                end_at = len(wp)-p
                p=backward_chk
        p=p-1
    result = True

wp = wp[end_at:len(wp)]
wp[len(wp)-1,0]=0

wp_s = librosa.frames_to_time(wp, sr=fs, hop_length=hop_length)

#f = open("/var/www/html/vhv/D_array.csv", "w");
#np.savetxt(f, D, delimiter=" ")

#f = open("/var/www/html/vhv/wp_array.csv", "w");
#np.savetxt(f, wp, delimiter=" ")

f = open("/var/www/html/vhv/wp_s_array.csv", "w");
np.savetxt(f, wp_s, delimiter=" ")
