import threading
import wave

import pyaudio


class RecordingModule:

    def __init__(self, chunk=1024, channels=2, fs=44100):
        self.data = []
        self.chunk = chunk
        self.channels = channels
        self.fs = fs
        self.sample_format = pyaudio.paInt16  # 16 bits per sample

        self.middle_man = MiddleMan(True)
        self.stream = None
        self.pyaudio = None

    def record_sample(self):
        self.pyaudio = pyaudio.PyAudio()  # Create an interface to PortAudio

        index = 0
        for i in range(30):
            info = self.pyaudio.get_device_info_by_index(i)
            print(info)
            if "stereo" in info["name"].lower():
                index = i
                print("Recording device index is {}".format(index))
                self.channels = info["maxInputChannels"]
                break
        print('Recording')

        self.stream = self.pyaudio.open(format=self.sample_format,
                                        channels=self.channels,
                                        rate=self.fs,
                                        frames_per_buffer=self.chunk,
                                        input=True,
                                        input_device_index=index)

        self.data = []  # Initialize array to store frames

        x = threading.Thread(target=self.__recording_thread)
        x.start()

        # print('Finished recording')
        #
        # # Save the recorded data as a WAV file

    def stop_recording(self):
        self.middle_man.set_condition(False)

    def __recording_thread(self):
        while self.middle_man.check_condition():
            data = self.stream.read(self.chunk)
            self.data.append(data)
            # print("Recorded chunk")
        print("Stopped recording")
        # Stop and close the stream
        self.stream.stop_stream()
        self.stream.close()
        # Terminate the PortAudio interface
        self.pyaudio.terminate()
        print("Shut down")

        filename = "test.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.pyaudio.get_sample_size(self.sample_format))
        wf.setframerate(self.fs)
        wf.writeframes(b''.join(self.data))
        wf.close()


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
        print("ATTEMPTED STOP")
        self.lock.release()


if __name__ == '__main__':
    inp = ""

    rec = RecordingModule()
    rec.record_sample()
    while inp != "x":
        print("Enter 'x' to stop recording: ")
        inp = input()
    rec.stop_recording()
