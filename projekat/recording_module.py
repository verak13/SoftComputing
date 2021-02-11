import threading
import wave
from time import time

import pyaudio

import sqlite3
import numpy as np
import pyautogui
from playsound import playsound


class RecordingObject:

    def __init__(self, chunk=1024, channels=2, fs=44100, record_handler=None, generating=False):
        self.filename = None
        self.data = []
        self.chunk_size = chunk
        self.channels = channels
        self.fs = fs
        self.sample_format = pyaudio.paInt16  # 16 bits per sample

        self.middle_man = MiddleMan(False)
        self.stream = None
        self.pyaudio = None

        self.record_handler = record_handler
        self.generating = generating

        self.counter = 0

    def record_sample(self):
        # self.middle_man.set_condition(True)
        self.pyaudio = pyaudio.PyAudio()  # Create an interface to PortAudio

        for i in range(self.pyaudio.get_device_count()):
            try:
                info = self.pyaudio.get_device_info_by_index(i)
            except:
                continue
            if 'Stereo Mix' in info['name']:
                self.index = i
                self.channels = info["maxInputChannels"]
                break

        print('Recording')

        # self.stream = self.pyaudio.open(format=self.sample_format,
        #                                 channels=self.channels,
        #                                 rate=self.fs,
        #                                 frames_per_buffer=self.chunk,
        #                                 input=True,
        #                                 input_device_index=index)
        self.stream = self.pyaudio.open(
            format=self.sample_format,
            channels=self.channels,
            rate=self.fs,
            frames_per_buffer=self.chunk_size,
            input=True,
            input_device_index=self.index
        )

        x = threading.Thread(target=self.__recording_thread)
        x.setDaemon(True)
        x.start()

        # print('Finished recording')
        #
        # # Save the recorded data as a WAV file

    def stop_recording(self):
        self.middle_man.set_condition(False)

    def __recording_thread(self):
        ratio = self.fs // self.chunk_size
        # ratio *= 2
        # ratio //= 3
        # while self.middle_man.check_condition():
        #     pass
        self.middle_man.set_condition(True)
        t1 = time()

        if self.generating:
            while self.middle_man.check_condition():
                data = self.stream.read(self.chunk_size)
                self.data.append(data)
            self.save_sample("sample/sample" + str(self.counter) + ".wav")
            pyautogui.click(x=727, y=718)
            pyautogui.screenshot("./ss/ss" + str(self.counter) + ".png")
            self.counter += 1
        else:
            i = 0
            t1 = time()
            while self.middle_man.check_condition() and i < ratio:
                data = self.stream.read(self.chunk_size)
                self.data.append(data)
                i += 1

            self._save_sample("sample/sample.wav")
        self.data = []
        print("Recording time", time() - t1)
        print("Stopped recording")

        self.close_streams()

        self.middle_man.set_condition(False)

    def save_sample(self, filename):
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.pyaudio.get_sample_size(self.sample_format))
        wf.setframerate(self.fs)
        wf.writeframes(b''.join(self.data))
        wf.close()
        print("WRITTEN")

    def _save_sample(self, filename):
        self.save_sample(filename)
        if self.record_handler is not None:
            self.record_handler.handle()

    def close_streams(self):
        self.stream.stop_stream()
        self.stream.close()
        # Terminate the PortAudio interface
        self.pyaudio.terminate()
        print("Shut down pyaudio")


class MiddleMan:
    def __init__(self, value):
        self.message = value
        self.lock = threading.Lock()

    def check_condition(self):
        self.lock.acquire()
        value = self.message
        self.lock.release()
        return value

    def set_condition(self, value):
        self.lock.acquire()
        self.message = value
        self.lock.release()


if __name__ == '__main__':
    inp = ""

    rec = RecordingObject()
    rec.record_sample()
    while inp != "x":
        print("Enter 'x' to stop recording: ")
        inp = input()
    rec.stop_recording()
