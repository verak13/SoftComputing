import threading
import wave
import numpy as np
import pyaudio
import sqlite3
import scipy.io
from scipy.io import wavfile
import random
import os

import hashlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (generate_binary_structure, iterate_structure, binary_erosion)
from operator import itemgetter


IDX_FREQ_I = 0
IDX_TIME_J = 1

# Sampling rate, related to the Nyquist conditions, which affects
# the range frequencies we can detect.
DEFAULT_FS = 44100

# Size of the FFT window, affects frequency granularity
DEFAULT_WINDOW_SIZE = 4096

# Ratio by which each sequential window overlaps the last and the
# next window. Higher overlap will allow a higher granularity of offset
# matching, but potentially more fingerprints.
DEFAULT_OVERLAP_RATIO = 0.5

# Degree to which a fingerprint can be paired with its neighbors --
# higher will cause more fingerprints, but potentially better accuracy.
DEFAULT_FAN_VALUE = 15

# Minimum amplitude in spectrogram in order to be considered a peak.
# This can be raised to reduce number of fingerprints, but can negatively
# affect accuracy.
DEFAULT_AMP_MIN = 10

# Number of cells around an amplitude peak in the spectrogram in order
# for Dejavu to consider it a spectral peak. Higher values mean less
# fingerprints and faster matching, but can potentially affect accuracy.
PEAK_NEIGHBORHOOD_SIZE = 20

# Thresholds on how close or far fingerprints can be in time in order
# to be paired as a fingerprint. If your max is too low, higher values of
# DEFAULT_FAN_VALUE may not perform as expected.
MIN_HASH_TIME_DELTA = 0
MAX_HASH_TIME_DELTA = 200

# If True, will sort peaks temporally for fingerprinting;
# not sorting will cut down number of fingerprints, but potentially
# affect performance.
PEAK_SORT = True

# Number of bits to throw away from the front of the SHA1 hash in the
# fingerprint calculation. The more you throw away, the less storage, but
# potentially higher collisions and misclassifications when identifying songs.
FINGERPRINT_REDUCTION = 20


def fingerprint(channel_samples, Fs=DEFAULT_FS,
                wsize=DEFAULT_WINDOW_SIZE,
                wratio=DEFAULT_OVERLAP_RATIO,
                fan_value=DEFAULT_FAN_VALUE,
                amp_min=DEFAULT_AMP_MIN,
                plots=False):

    # show samples plot
    if plots:
      plt.plot(channel_samples)
      plt.title('%d samples' % len(channel_samples))
      plt.xlabel('time (s)')
      plt.ylabel('amplitude (A)')
      plt.show()
      plt.gca().invert_yaxis()

    # FFT the channel, log transform output, find local maxima, then return
    # locally sensitive hashes.
    # FFT the signal and extract frequency components

    # plot the angle spectrum of segments within the signal in a colormap
    arr2D = mlab.specgram(
        channel_samples,
        NFFT=wsize,
        Fs=Fs,
        window=mlab.window_hanning,
        noverlap=int(wsize * wratio))[0]

    # show spectrogram plot
    if plots:
      plt.plot(arr2D)
      plt.title('FFT')
      plt.show()


    print(len(arr2D) * len(arr2D[0]))
    print(len(arr2D))
    print(len(arr2D[0]))
    print(len(channel_samples))
    #print(arr2D[0])
    # apply log transform since specgram() returns linear array
    arr2D = 10 * np.log10(arr2D) # calculates the base 10 logarithm for all elements of arr2D
    arr2D[arr2D == -np.inf] = 0  # replace infs with zeros

    print(arr2D[0])

    # find local maxima
    local_maxima = get_2D_peaks(arr2D, plot=plots, amp_min=amp_min)

    msg = '   local_maxima: %d of frequency & time pairs'
    #print colored(msg, attrs=['dark']) % len(local_maxima)

    # return hashes
    return generate_hashes(local_maxima, fan_value=fan_value)

def get_2D_peaks(arr2D, plot=False, amp_min=DEFAULT_AMP_MIN):
    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

    # find local maxima using our fliter shape
    local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D
    background = (arr2D == 0)
    eroded_background = binary_erosion(background, structure=neighborhood,
                                       border_value=1)

    # Boolean mask of arr2D with True at peaks
    detected_peaks = local_max ^ eroded_background

    # extract peaks
    amps = arr2D[detected_peaks]
    j, i = np.where(detected_peaks)

    # filter peaks
    amps = amps.flatten()
    peaks = zip(i, j, amps)
    peaks_filtered = [x for x in peaks if x[2] > amp_min]  # freq, time, amp
    #peaks_filtered = [x for x in peaks]
    print("***")
    print(peaks_filtered)
    print("***")


    # get indices for frequency and time
    frequency_idx = [x[1] for x in peaks_filtered]
    time_idx = [x[0] for x in peaks_filtered]

    # scatter of the peaks
    if plot:
      fig, ax = plt.subplots()
      ax.imshow(arr2D)
      ax.scatter(time_idx, frequency_idx)
      ax.set_xlabel('Time')
      ax.set_ylabel('Frequency')
      ax.set_title("Spectrogram")
      plt.gca().invert_yaxis()
      plt.show()

    print(frequency_idx, time_idx)
    return zip(frequency_idx, time_idx)

# Hash list structure: sha1_hash[0:20] time_offset
# example: [(e05b341a9b77a51fd26, 32), ... ]
def generate_hashes(peaks, fan_value=DEFAULT_FAN_VALUE):
    if PEAK_SORT:
      #peaks.sort(key=itemgetter(1))
      peaks = sorted(peaks, key = itemgetter(1))

    # bruteforce all peaks
    for i in range(len(peaks)):
      for j in range(1, fan_value):
        if (i + j) < len(peaks):

          # take current & next peak frequency value
          freq1 = peaks[i][IDX_FREQ_I]
          freq2 = peaks[i + j][IDX_FREQ_I]

          # take current & next -peak time offset
          t1 = peaks[i][IDX_TIME_J]
          t2 = peaks[i + j][IDX_TIME_J]

          # get diff of time offsets
          t_delta = t2 - t1

          # check if delta is between min & max
          if t_delta >= MIN_HASH_TIME_DELTA and t_delta <= MAX_HASH_TIME_DELTA:
            a = str(freq1).encode('utf-8')
            b = str(freq2).encode('utf-8')
            c = str(t_delta).encode('utf-8')
            #h = hashlib.sha1("%s|%s|%s" % (str(freq1), str(freq2), str(t_delta)))
            h = hashlib.sha1("%s|%s|%s".encode('utf-8') % (a, b, c))
            #h = hashlib.sha1(a + "|" + b + "|" + c)
            yield (h.hexdigest()[0:FINGERPRINT_REDUCTION], t1)



#drugi nacin - pokusaj, nije gotovo
'''
def fingerprint(channel_samples, Fs=DEFAULT_FS,
                wsize=DEFAULT_WINDOW_SIZE,
                wratio=DEFAULT_OVERLAP_RATIO,
                fan_value=DEFAULT_FAN_VALUE,
                amp_min=DEFAULT_AMP_MIN,
                plots=False):


    frate = 11025.0
    frate = 44100
   # w = np.fft.fft(channel_samples)
    #print(w)
   # freqs = np.fft.fftfreq(len(w))
   # print(len(freqs))
   # print(freqs)
    #print(freqs.min(), freqs.max())
    # (-0.5, 0.499975)


    totalSize = len(channel_samples)
    sampledChunkSize = int(totalSize / 10)
    print(sampledChunkSize)
    frequencies = []
    frequenciesreq = []
    for j in range(sampledChunkSize-1):
        v = channel_samples[j*10:10*j+10]
        #print(v)
        frequencies.append(np.fft.fft(v))
        frequenciesreq.append(np.fft.fftfreq(len(v)))
    print("55555555555555555555555")
    #print(frequencies[0])
    print(frequenciesreq[0])


        # Find the peak in the coefficients
       # idx = np.argmax(np.abs(w))
      #  freq = freqs[idx]
       # freq_in_hertz = abs(freq * frate)
        #print(freq_in_hertz)
        # 439.8975

    arr2D = 10 * np.log10(frequenciesreq)  # calculates the base 10 logarithm for all elements of arr2D
    arr2D[arr2D == -np.inf] = 0  # replace infs with zeros

    # find local maxima
    local_maxima = get_2D_peaks(arr2D, plot=plots, amp_min=amp_min)

    
    #range = [40, 80, 120, 180, 300]
    #highscores = []

    #for t in range(0, len(frequenciesreq)):
    #    highscores.append([])
    #    for freq in range(40, 300):
    #        mag = np.log10(frequenciesreq[t][freq].abs() + 1)

    #        index = getIndex(freq)
    #        if mag > highscores[t][index]:
    #            points[t][index] = freq


    #    h = hash(points[t][0], points[t][1], points[t][2], points[t][3])
    

    # return hashes
    return generate_hashes(local_maxima, fan_value=fan_value)

def getIndex(freq):
    rangee = [40, 80, 120, 180, 300]
    i = 0
    while rangee[i] < freq:
        i+=1
    return i


def get_2D_peaks(arr2D, plot=False, amp_min=DEFAULT_AMP_MIN):
    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

    # find local maxima using our fliter shape
    local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D
    background = (arr2D == 0)
    eroded_background = binary_erosion(background, structure=neighborhood,
                                       border_value=1)

    # Boolean mask of arr2D with True at peaks
    detected_peaks = local_max ^ eroded_background

    # extract peaks
    amps = arr2D[detected_peaks]
    j, i = np.where(detected_peaks)

    # filter peaks
    amps = amps.flatten()
    peaks = zip(i, j, amps)
    peaks_filtered = [x for x in peaks if x[2] > amp_min]  # freq, time, amp
    #peaks_filtered = [x for x in peaks]
    print("***")
    print(peaks_filtered)
    print("***")


    # get indices for frequency and time
    frequency_idx = [x[1] for x in peaks_filtered]
    time_idx = [x[0] for x in peaks_filtered]

    # scatter of the peaks
    if plot:
      fig, ax = plt.subplots()
      ax.imshow(arr2D)
      ax.scatter(time_idx, frequency_idx)
      ax.set_xlabel('Time')
      ax.set_ylabel('Frequency')
      ax.set_title("Spectrogram")
      plt.gca().invert_yaxis()
      plt.show()

    print(frequency_idx, time_idx)
    return zip(frequency_idx, time_idx)

# Hash list structure: sha1_hash[0:20] time_offset
# example: [(e05b341a9b77a51fd26, 32), ... ]
def generate_hashes(peaks, fan_value=DEFAULT_FAN_VALUE):
    if PEAK_SORT:
      #peaks.sort(key=itemgetter(1))
      peaks = sorted(peaks, key = itemgetter(1))

    # bruteforce all peaks
    for i in range(len(peaks)):
      for j in range(1, fan_value):
        if (i + j) < len(peaks):

          # take current & next peak frequency value
          freq1 = peaks[i][IDX_FREQ_I]
          freq2 = peaks[i + j][IDX_FREQ_I]

          # take current & next -peak time offset
          t1 = peaks[i][IDX_TIME_J]
          t2 = peaks[i + j][IDX_TIME_J]

          # get diff of time offsets
          t_delta = t2 - t1

          # check if delta is between min & max
          if t_delta >= MIN_HASH_TIME_DELTA and t_delta <= MAX_HASH_TIME_DELTA:
            a = str(freq1).encode('utf-8')
            b = str(freq2).encode('utf-8')
            c = str(t_delta).encode('utf-8')
            #h = hashlib.sha1("%s|%s|%s" % (str(freq1), str(freq2), str(t_delta)))
            h = hashlib.sha1("%s|%s|%s".encode('utf-8') % (a, b, c))
            #h = hashlib.sha1(a + "|" + b + "|" + c)
            yield (h.hexdigest()[0:FINGERPRINT_REDUCTION], t1)
'''










if __name__ == '__main__':
    print("start")

    wf = wave.open("end.wav", 'rb')

    print(wf)
    print(wf.getframerate())
    print(wf.getparams())
    #print(wf.readframes(184320))
    sound_arr = np.frombuffer(wf.readframes(184320), dtype=np.uint8)
    print(sound_arr)


    #data = np.fromstring(wf._data, np.int16)

    #channels = []
    #for chn in xrange(wf.getnchannels()):
    #    channels.append(data[chn::wf.getnchannels()])

    samplerate, data = wavfile.read("end.wav")
    print(samplerate, data)
    print(f"number of channels = {data.shape[1]}")
    length = data.shape[0] / samplerate
    print(f"length = {length}s")
    print("left channel: ", data[:, 0])
    print("right channel: ", data[:, 1])

    hashes = set()
    for i in range(wf.getnchannels()):
        channel_hashes = fingerprint(data[:, i], Fs=wf.getframerate())
        channel_hashes = set(channel_hashes)
        hashes |= channel_hashes
        #print(hashes)
    values = []
    for hash, offset in hashes:
        values.append(("pjesma", hash, offset))
    print(values)
    print(len(values))

    #conn = sqlite3.connect('example.db')
    #c = conn.cursor()

    #c.execute('''CREATE TABLE stocks
    #             (date text, trans text, symbol text, qty real, price real)''')

    #c.execute("INSERT INTO stocks VALUES ('2006-01-05','BUY','RHAT',100,35.14)")

    #conn.commit()

    #t = ('RHAT',)
    #c.execute('SELECT * FROM stocks WHERE symbol=?', t)
    #print(c.fetchone())

    #conn.close()


