import math
import os
import threading
from time import sleep, time

import cv2
import d3dshot as d3dshot
import matplotlib.pyplot as plt
import numpy as np
import pyscreenshot as ImageGrab
from PIL import Image
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, LeakyReLU, Flatten, Dense, MaxPool2D
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy, mean_absolute_error, \
    mean_squared_error
from tensorflow.keras.metrics import Accuracy, BinaryAccuracy, MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow_core.python.keras.utils import Sequence
from playsound import playsound

BATCH = 32
TRAIN = 582
VALIDATE = 87
TEST = 51
EPOCHS = 10
LEARNING_RATE = 0.03


def configure_cnn():
    model = Sequential()

    # # 380x140
    # model.add(Conv2D(16, (3, 3), (2, 2), input_shape=(140, 380, 3)))
    # model.add(LeakyReLU())
    # # ~190x70
    # model.add(Conv2D(16, (3, 3), (2, 2)))
    # model.add(LeakyReLU())
    #
    # # ~90x30
    # model.add(Conv2D(32, (3, 3), (2, 2)))
    # model.add(LeakyReLU())
    #
    # # ~40x15
    # model.add(Conv2D(32, (3, 3), (2, 2)))
    # model.add(LeakyReLU())
    #
    # # ~20x7
    # model.add(Conv2D(32, (3, 3), (2, 2)))
    # model.add(LeakyReLU())

    # 380x140
    model.add(Conv2D(16, (3, 3), (2, 2), input_shape=(140, 380, 1), padding="same"))
    model.add(LeakyReLU())
    model.add(MaxPool2D((2, 2), strides=(1, 1), padding="same"))
    # 380x140

    model.add(Conv2D(16, (3, 3), (2, 2), padding="same"))
    model.add(LeakyReLU())
    model.add(MaxPool2D((2, 2), strides=(1, 1), padding="same"))
    # 380x140

    model.add(Conv2D(16, (3, 3), (2, 2)))
    model.add(LeakyReLU())
    model.add(MaxPool2D((2, 2), strides=(1, 1), padding="same"))

    # 380x140
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(LeakyReLU())

    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(LeakyReLU())

    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(LeakyReLU())

    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(LeakyReLU())
    model.add(MaxPool2D((2, 2), strides=(1, 1), padding="same"))

    # 380x140
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(LeakyReLU())

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(LeakyReLU())

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(LeakyReLU())

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(LeakyReLU())

    model.add(Flatten())
    model.add(Dense(16))
    model.add(LeakyReLU())
    model.add(Dense(8))
    model.add(LeakyReLU())
    model.add(Dense(2))

    model.add(LeakyReLU())
    model.add(Dense(1))

    model.build()

    model.summary()

    return model


class Generator(Sequence):
    # Class is a dataset wrapper for better training performance
    def __init__(self, x_set, y_set, batch_size=256):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])

    def __len__(self):
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[inds]
        batch_y = self.y[inds]
        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


def extract_yellow(img):

    # cv2.imshow("imag_hsv", img)
    # cv2.waitKey()

    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # cv2.imshow("imag_hsv", img_hsv)
    # cv2.waitKey()

    hsv_color1 = np.asarray([20, 100, 100])
    hsv_color2 = np.asarray([30, 255, 255])

    img = cv2.inRange(img_hsv, hsv_color1, hsv_color2)
    # cv2.imshow("mask", img)
    # cv2.waitKey()
    img = np.expand_dims(img, axis=2)
    img = np.float32(img)
    return img


def open_data(length, path="./data/", single_color=False):
    x = np.empty((length, 140, 380, 1))
    # x = np.empty((length, 140, 380, 3))
    # y = np.empty((length, 2))
    y = np.empty(length)
    classes = []
    with open(path + "classes.txt") as f:
        for line in f.readlines():
            try:
                index = int(line)
                classes.append(index)
            except Exception:
                pass

    for i in range(length):
        img = cv2.imread(path + "{}.png".format(i))
        cv2.imshow("LOAD", img)
        cv2.waitKey()
        conv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.waitKey()
        if single_color:
            img = extract_yellow(img)


        # cv2.imwrite("LMAO.png", mask)
        #
        # plt.imshow(mask)  # this colormap will display in black / white
        # plt.show()
        # sleep(1)

        # plt.imshow(rgb)
        # plt.show()
        # cv2.imshow("d", rgb)

        x[i] = img
        # y[i] = np.zeros(2)
        # y[i][classes[i]] = 1
        y[i] = classes[i]

    return x, y


def train_model(single_color=False):
    x, y = open_data(TRAIN, "./data/", single_color)
    x_val, y_val = open_data(VALIDATE, "./validate_data/", single_color)
    generator = Generator(x, y, BATCH)

    model = configure_cnn()

    model.compile(
        optimizer=Adam(
            # learning_rate=LEARNING_RATE
        ),
        loss=mean_absolute_error,
        metrics=[Accuracy(), BinaryAccuracy(), MeanSquaredError(), MeanAbsoluteError()],
    )

    history = model.fit_generator(generator,
                                  validation_data=(x_val, y_val),
                                  steps_per_epoch=TRAIN // BATCH,
                                  shuffle=True,
                                  epochs=EPOCHS)

    model.save("MODEL.h5")


def grab_interesting_area(d3d):
    pil = d3d.screenshot()
    img = np.asarray(pil)
    # img = ImageGrab.grab(bbox=(760, 470, 1140, 610))
    # img = ImageGrab.grab(bbox=(760, 520, 1140, 660))
    # img = ImageGrab.grab(bbox=(760, 570, 1140, 710))
    # img.show()
    return img[520:660, 760:1140]


def generate_data(delay=0.3, duration=60, start_delay=3, path="./data/", start_file=0):
    print("Starting recording in:")
    for i in range(start_delay):
        sleep(1)
        print(3 - i)

    sleep(1)
    index = start_file
    time = 0
    d3d = d3dshot.create()
    while time < duration:
        # img = pyautogui.screenshot()

        # part of the screen
        img = grab_interesting_area(d3d)
        img = Image.fromarray(img)
        img.save(path + "{}.png".format(index))

        # img = np.asarray(img)
        sleep(delay)
        time += delay
        index += 1


def sort(val):
    return int(val.split(".")[0])


def order_files_by_number(path):
    dir = os.listdir(path)
    dir2 = []
    for d in dir:
        try:
            int(d.split(".")[0])
            dir2.append(d)
        except Exception:
            pass
    dir2.sort(key=sort)
    print(dir2)

    for i, d in enumerate(dir2):
        img = Image.open(path + d)
        img.save(path + str(i) + ".png")


def load_model(path="./MODEL.h5"):
    model = keras.models.load_model(path)
    return model


def test_model(path="./MODEL.h5", single_color=False):
    model = load_model(path)
    x_test, y_test = open_data(TEST, "./test_data/", single_color)

    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=BATCH)
    print("test loss, test acc:", results)


def predict_model(path):
    model = load_model(path)

    x, y = open_data(TRAIN, "./data/")
    x_val, y_val = open_data(VALIDATE, "./validate_data/")

    res1 = model.predict_classes(x)
    print(res1)
    res1 = model.predict(x)
    print(res1)

    res1 = model.predict(x_val)
    print(res1)
    res2 = model.predict_classes(x_val)
    print(res2)


def hough(images):
    length = len(images)
    results = np.empty(length)
    i = 0
    for img in images:
        # if img.empty():
        #     print(i)
        print(np.count_nonzero(img))
        try:
            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 1)
            if circles is not None:
                results[i] = 1
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            results[i] = 0
        except Exception as e:
            print(e)
    return results


def record_thread(model, single_color=False):
    d3d = d3dshot.create()
    recording = False
    if single_color:
        samples = np.empty((1, 140, 380, 1))
    else:
        samples = np.empty((1, 140, 380, 3))

    while True:
        t1 = time()
        arr = grab_interesting_area(d3d)
        # cv2.imshow("CAPTURE", arr)
        # cv2.waitKey()
        if single_color:
            arr = extract_yellow(arr)
        samples[0] = arr
        res = model.predict_classes(samples)
        print(res[0][0])
        if res[0][0] == 0 and recording:
            recording = False
            st = threading.Thread(target=end)
            st.daemon = True  # Daemonize thread
            st.start()
            print("ENDED")
        elif res[0][0] == 1 and not recording:
            recording = True
            st = threading.Thread(target=start)
            st.daemon = True  # Daemonize thread
            st.start()
            print("STARTED")
        # sleep(0.04)
        print(time() - t1)


def start():
    playsound("./start.mp3")


def end():
    playsound("./end.mp3")


def real_time_detection(model_path, start_delay=3, single_color=False):
    print("Starting recording in:")
    for i in range(start_delay):
        sleep(1)
        print(3 - i)


    model = load_model(model_path)
    record_thread(model, single_color=single_color)


if __name__ == '__main__':
    # configure_cnn()
    # generate_data()
    # generate_data(path="./data/", start_file=TRAIN + 1, delay=0.1)

    # generate_data(path="./test_data/")
    # order_files_by_number("./data/")
    # train_model(single_color=True)

    # test_model("./MODEL.973.COLOR.h5")

    # predict_model(path="./MODEL.973.COLOR.h5")
    # x, y = open_data(15)
    # hough(x)

    # test_model("./MODEL.1.h5", single_color=True)

    # real_time_detection("./MODEL.987.COLOR.h5")
    # real_time_detection("./MODEL.1.h5", single_color=True)
    real_time_detection("./MODEL.1LARGE.h5", single_color=True)
