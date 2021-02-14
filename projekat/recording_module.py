import abc
import threading
import wave
from time import time

import pyaudio

import sqlite3
import numpy as np
import pyautogui
from playsound import playsound

from projekat.read_text_module import OCR


class RecordingObject:

    def __init__(self, chunk=1024, channels=2, fs=44100, record_handler=None):
        self.thread = None
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

        self.counter = 0

        self.___recording_thread = None

    def record_sample(self):
        if self.___recording_thread != None:
            return
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

        self.___recording_thread = threading.Thread(target=self.__recording_thread)
        self.___recording_thread.setDaemon(True)
        self.___recording_thread.start()

        # print('Finished recording')
        #
        # # Save the recorded data as a WAV file

    def stop_recording(self):
        self.middle_man.set_condition(False)

    def __recording_thread(self):

        # while self.middle_man.check_condition():
        #     pass
        self.middle_man.set_condition(True)
        t1 = time()

        self.record_loop()

        self.data = []
        print("Recording time", time() - t1)
        print("Stopped recording")

        self.close_streams()

        self.middle_man.set_condition(False)
        self.___recording_thread = None

    @abc.abstractmethod
    def record_loop(self):
        pass

    def save_sample(self, filename):
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.pyaudio.get_sample_size(self.sample_format))
        wf.setframerate(self.fs)
        wf.writeframes(b''.join(self.data))
        wf.close()

    def save_sample_and_handle(self, filename):

        self.save_sample(filename)
        self.handle()

    def handle(self):
        if self.record_handler is not None:
            self.record_handler.handle()

    def close_streams(self):
        self.stream.stop_stream()
        self.stream.close()
        # Terminate the PortAudio interface
        self.pyaudio.terminate()
        print("Shut down pyaudio")


class DefaultRecorder(RecordingObject):

    def __init__(self, chunk=1024, channels=2, fs=44100, record_handler=None, stopper=None):
        super(DefaultRecorder, self).__init__(chunk, channels, fs, record_handler)
        self.stopper = stopper

    def record_loop(self):
        i = 0
        ratio = np.ceil(self.fs / self.chunk_size)
        # ratio *= 2
        # ratio //= 3
        # ratio = np.ceil(ratio/2)
        self.stopper.set_condition(True)
        while self.middle_man.check_condition():
            data = self.stream.read(self.chunk_size)
            self.data.append(data)
            i += 1
            if i >= ratio and self.stopper.check_condition():

                t1 = time()
                self.stopper.set_condition(False)
                self.save_sample("sample/sample.wav")
                self.thread = threading.Thread(target=self.handle)
                self.thread.start()
                print("HANDLE TIME", time() - t1)
                i = 0
                self.data = []
        self.stopper.set_condition(True)

        if self.thread is not None:
            self.thread.join()
        # Stisni prvi kad ne znas nista
        pyautogui.click(x=727, y=718)
        pyautogui.moveTo(100, 100)
        # ocr = OCR("TEXT_MODEL.h5")
        # ocr.check_answer()


class SentientRecorder(RecordingObject):
    def record_loop(self):
        while self.middle_man.check_condition():
            data = self.stream.read(self.chunk_size)
            self.data.append(data)
        self.save_sample_and_handle("sample/sample.wav")


class GenerateRecorder(RecordingObject):
    def record_loop(self):
        while self.middle_man.check_condition():
            data = self.stream.read(self.chunk_size)
            self.data.append(data)
        self.save_sample("sample/sample" + str(self.counter) + ".wav")
        pyautogui.click(x=727, y=718)
        pyautogui.screenshot("./ss/ss" + str(self.counter) + ".png")
        self.counter += 1


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
