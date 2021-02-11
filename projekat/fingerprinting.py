import threading
import wave
from time import time

import numpy as np
import pyaudio
import sqlite3
import scipy.io
from scipy.io import wavfile
import random
import os
import operator

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

    # print(len(arr2D) * len(arr2D[0]))
    # print(len(arr2D))
    # print(len(arr2D[0]))
    # print(len(channel_samples))
    # print(arr2D[0])
    # apply log transform since specgram() returns linear array
    # print(np.log10(arr2D))
    arr2D = 10 * np.log10(arr2D)  # calculates the base 10 logarithm for all elements of arr2D
    arr2D[arr2D == -np.inf] = 0  # replace infs with zeros

    # print(arr2D[0])
    # print(arr2D[1])

    '''
    print("=======================================")
    #totalSize = len(channel_samples)
    #sampledChunkSize = int(totalSize / 80)
    #print(sampledChunkSize)
    frequencies = []
    frequenciesreq = []
    for j in range(0, len(channel_samples), 80):
        if (j + 80) >= len(channel_samples):
            continue
        data_tmp = channel_samples[j:j + 80]
        data_tmp = data_tmp - np.mean(data_tmp)
        data_tmp = np.multiply(data_tmp, np.hanning(len(data_tmp)))
        fft_data_tmp = np.fft.fft(data_tmp, n=wsize)
        fft_data_tmp = abs(fft_data_tmp[:int(len(fft_data_tmp) / 2)]) ** 2
        frequencies.append(fft_data_tmp)

        #frequencies.append(np.abs(np.fft.fft(v)))
        frequenciesreq.append(np.abs(np.fft.fftfreq(len(fft_data_tmp))))
    # print(frequencies[0])
    print(frequencies[0])
    frequenciesreq = 10 * np.log10(frequenciesreq)
    frequenciesreq[frequenciesreq == -np.inf] = 0  # replace infs with zeros
    print(frequenciesreq[1])
    print(frequenciesreq[2])
    print("++++++++++++++++++++++++++++++++++++++++")
    '''

    # find local maxima
    local_maxima = get_2D_peaks(arr2D, plot=plots, amp_min=amp_min)

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
    # print(peaks_filtered)

    # get indices for frequency and time
    frequency_idx = [x[1] for x in peaks_filtered]
    time_idx = [x[0] for x in peaks_filtered]

    # print(frequency_idx, time_idx)
    return zip(frequency_idx, time_idx)


# Hash list structure: sha1_hash[0:20] time_offset
# example: [(e05b341a9b77a51fd26, 32), ... ]
def generate_hashes(peaks, fan_value=DEFAULT_FAN_VALUE):
    if PEAK_SORT:
        peaks = sorted(peaks, key=itemgetter(1))

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
                    h = hashlib.sha1("%s|%s|%s".encode('utf-8') % (
                        str(freq1).encode('utf-8'), str(freq2).encode('utf-8'), str(t_delta).encode('utf-8')))
                    yield (h.hexdigest()[0:FINGERPRINT_REDUCTION], t1)


# drugi nacin - pokusaj, nije gotovo
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


    print("=======================================")
    totalSize = len(channel_samples)
    sampledChunkSize = int(totalSize / 80)
    print(sampledChunkSize)
    frequencies = []
    frequenciesreq = []
    for j in range(sampledChunkSize):
        if j == sampledChunkSize:
            v = channel_samples[j*80:]
        else:
            v = channel_samples[j * 80:80 * j + 80]
        #if j == 4: print(v)
        frequencies.append(np.abs(np.fft.fft(v)))
        frequenciesreq.append(np.abs(np.fft.fftfreq(len(v))))
        #if j == 0: print(frequenciesreq)
    # print(frequencies[0])
    frequenciesreq = 10 * np.log10(frequenciesreq)
    frequenciesreq[frequenciesreq == -np.inf] = 0  # replace infs with zeros
    print(frequenciesreq[1])
    print(frequenciesreq[2])
    print("++++++++++++++++++++++++++++++++++++++++")


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


def take_fingerprints(file, filename):
    print("start")

    wf = wave.open(file, 'rb')

    print(wf)
    print(wf.getframerate())
    print(wf.getnframes())
    print(wf.getparams())
    # print(wf.readframes(184320))
    sound_arr = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.uint8)
    print(sound_arr)

    samplerate, data = wavfile.read(file)
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
        # print(hashes)
    values = []
    for hash, offset in hashes:
        values.append((filename.split('.')[0], hash, offset))
    print(values)
    print(len(values))

    conn = sqlite3.connect('songs.db')
    c = conn.cursor()
    for value in values:
        print(value[0], value[1], value[2])
        query = "INSERT INTO fingerprints (song, hash, offset) VALUES (?, ?, ?)"
        c.execute(query, value)

    conn.commit()
    conn.close()


def do_hash():
    file = "sample/sample.wav"
    wf = wave.open(file, 'rb')

    samplerate, data = wavfile.read(file)
    hashes = set()
    for i in range(wf.getnchannels()):
        channel_hashes = fingerprint(data[:, i], Fs=wf.getframerate())
        channel_hashes = set(channel_hashes)
        hashes |= channel_hashes
        # print(hashes)
    conn1 = sqlite3.connect('songs.db')
    c1 = conn1.cursor()
    answer = ""
    t1 = time()
    for hasha, offset in hashes:
        c1.execute('SELECT song FROM fingerprints WHERE hash = ? group by song order by count(hash) desc', (hasha,))
        result = c1.fetchall()
        print(result)
        if len(result) > 0:
            answer = result[0][0]
            break
    conn1.close()

    # answer = max(answer_dict, key=lambda key: answer_dict[key])
    print("DB time", time() - t1)
    print(answer)
    return answer


if __name__ == '__main__':
    # take_fingerprints("start.wav", "numa numa ej")

    # conn = sqlite3.connect('songs.db')
    # c = conn.cursor()
    # c.execute('''CREATE TABLE fingerprints (song, hash, offset)''')
    # conn.close()

    # for filename in os.listdir('data'):
    #    if filename.endswith(".wav"):
    #        print(os.path.join('data', filename))
    #        take_fingerprints(os.path.join('data', filename), filename)
    #        continue
    #    else:
    #        continue

    conn1 = sqlite3.connect('songs.db')
    c1 = conn1.cursor()
    # t = ('0603e28991b838e87a03')
    c1.execute('SELECT * FROM fingerprints WHERE song = ?', ('7 rings--Ariana Grande',))
    print(c1.fetchall())
    conn1.close()

    # print(int.from_bytes(b'\x02\x00\x00\x00\x00\x00\x00\x00', 'little'))
