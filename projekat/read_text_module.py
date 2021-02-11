import numpy as np
import cv2  # OpenCV
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import pyautogui
import time
from difflib import SequenceMatcher

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD

CHAR_DIM = 32
CHAR_DIM_W = 32
CHAR_DIM_H = 48
EPOCHS = 300

ALPHABET = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p',
            'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z',
            'x', 'c', 'v', 'b', 'n', 'm', '(', ')', '&', '!',
            'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P',
            'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'Z',
            'X', 'C', 'V', 'B', 'N', 'M',
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
            '\'', 'Ø', '+', '-', ','
            ]

DIMS = [[690, 780, 490, 950], [690, 780, 1000, 1480], [800, 890, 490, 950], [800, 890, 1000, 1480]]


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin


def invert(image):
    return 255 - image


def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')


def dilate(image, kernel=np.ones((6, 6))):
    return cv2.dilate(image, kernel, iterations=1)


def erode(image, kernel=np.ones((6, 6))):
    return cv2.erode(image, kernel, iterations=1)


def resize_region(region):
    # return cv2.resize(region, (CHAR_DIM_W, CHAR_DIM_H), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(region, (CHAR_DIM, CHAR_DIM), interpolation=cv2.INTER_CUBIC)


def scale_to_range(image):
    return image / 255


def matrix_to_vector(image):
    return image.flatten()


def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
    return ready_for_ann


def convert_output(alphabet):
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)


def create_ann(output_size):
    ann = Sequential()
    # ann.add(Dense(128, input_dim=CHAR_DIM_W * CHAR_DIM_H, activation='sigmoid'))
    ann.add(Dense(128, input_dim=CHAR_DIM ** 2, activation='sigmoid'))
    ann.add(Dense(64, activation='sigmoid'))
    ann.add(Dense(output_size, activation='sigmoid'))
    return ann


def train_ann(ann, X_train, y_train, epochs):
    X_train = np.array(X_train, np.float32)  # dati ulaz
    y_train = np.array(y_train, np.float32)  # zeljeni izlazi na date ulaze

    print("\nTraining started...")
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='categorical_crossentropy',
                optimizer=sgd,
                metrics=["categorical_accuracy", "accuracy", "mean_squared_error"])
    ann.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=1, shuffle=False, )
    print("\nTraining completed...")
    return ann


def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]


def display_result(outputs, alphabet):
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result


def select_roi_with_distances(image_orig, image_bin):
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []
    print(len(contours))
    for i in range(0, len(contours)):
        center1, size1, angle1 = cv2.minAreaRect(contours[i])
        width1, height1 = size1
        if hierarchy[0][i][3] == 0:
            x, y, w, h = cv2.boundingRect(contours[i])
            region = image_bin[y:y + h + 1, x:x + w + 1]
            # cv2.imshow("woaow", region)
            # cv2.waitKey()

            region = dilate(region)

            # kernel = np.ones((3, 3))
            # kernel = np.asarray([
            #     [0, 1, 1, 0],
            #     [1, 1, 1, 1],
            #     [1, 1, 1, 1],
            #     [0, 1, 1, 0],
            # ],
            #     np.uint8
            # )
            # region = erode(region, kernel)

            # kernel = np.asarray([
            #     [1, 1, 1, 1, 1, 1],
            #     [1, 1, 1, 1, 1, 1],
            #     [0, 0, 0, 0, 0, 0],
            #     [0, 0, 0, 0, 0, 0],
            #     [1, 1, 1, 1, 1, 1],
            #     [1, 1, 1, 1, 1, 1]
            # ],
            #     np.uint8
            # )
            # kernel = np.asarray([
            #     [0, 0, 0, 0],
            #     [0, 1, 1, 0],
            #     [0, 1, 1, 0],
            #     [0, 0, 0, 0],
            # ],
            #     np.uint8
            # )
            # region = erode(region, kernel)
            # region = dilate(region, kernel)

            # region = erode(region)

            # cv2.imshow("woaow", region)
            # cv2.waitKey()
            # region = erode(region)
            region = resize_region(region)
            regions_array.append([region, (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # TODO
            # cv2.imshow("woaow", region)
            # cv2.waitKey()

    regions_array = sorted(regions_array, key=lambda x: x[1][0])

    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    # izdvojiti sortirane parametre opisujucih pravougaonika
    # izracunati rastojanja izmedju svih susednih regiona po X osi i dodati ih u niz rastojanja
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2])  # x_next - (x_current + w_current)
        region_distances.append(distance)

    return image_orig, sorted_regions, region_distances, contours


def display_result_with_spaces(outputs, alphabet, k_means):
    # odredjivanje indeksa grupe koja odgovara rastojanju izmedju reci
    w_space_group = max(enumerate(k_means.cluster_centers_), key=lambda x: x[1])[0]
    # print(max(enumerate(k_means.cluster_centers_), key=lambda x: x[1]))
    # print(k_means.cluster_centers_)
    # print(k_means.labels_)
    result = alphabet[winner(outputs[0])]
    # iterativno dodavanje prepoznatih elemenata
    # dodavanje space karaktera ako je rastojanje izmedju dva slova odgovara rastojanju izmedju reci
    for idx, output in enumerate(outputs[1:, :]):
        if k_means.labels_[idx] == w_space_group:
            result += ' '
        result += alphabet[winner(output)]
    return result


def image_grayy(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_binn(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    # ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def get_most_similar(correct, answers):
    max = similar(correct, answers[0])
    maxText = answers[0]
    index = 0
    for i in range(1, 4):
        current = similar(correct, answers[i])
        if current > max:
            max = current
            maxText = answers[i]
            index = i
    return max, maxText, index


def prepare_img_for_roi(img):
    img_bin = image_binn(image_grayy(img))
    # cv2.imshow("W", img_bin)
    # cv2.waitKey()
    # kernel = np.asarray([
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     ],
    #     np.uint8
    # )
    # kernel = np.asarray([
    #         [0, 0, 0, 1, 0, 0, 0],
    #         [0, 0, 0, 1, 0, 0, 0],
    #         [0, 0, 0, 1, 0, 0, 0],
    #         [1, 1, 1, 1, 1, 1, 1],
    #         [0, 0, 0, 1, 0, 0, 0],
    #         [0, 0, 0, 1, 0, 0, 0],
    #         [0, 0, 0, 1, 0, 0, 0],
    # ],
    #     np.uint8
    # )
    # kernel = np.asarray([
    #         [0, 0, 0, 1, 0, 0, 0],
    #         [0, 0, 1, 1, 1, 0, 0],
    #         [0, 1, 1, 1, 1, 1, 0],
    #         [1, 1, 1, 1, 1, 1, 1],
    #         [0, 1, 1, 1, 1, 1, 0],
    #         [0, 0, 1, 1, 1, 0, 0],
    #         [0, 0, 0, 1, 0, 0, 0],
    #     ],
    #     np.uint8
    # )
    kernel = np.asarray([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ],
        np.uint8
    )
    # img_bin = erode(img_bin, kernel=np.ones((7, 5)))
    img_bin = erode(img_bin, kernel)
    # img_bin = dilate(img_bin, kernel=np.ones((3, 3)))
    # cv2.imshow("W", img_bin)
    # cv2.waitKey()
    return img_bin


def train_and_savemodel():
    dirname = os.path.dirname(__file__)

    filenameproba = os.path.join(dirname, 'songpop_text/all.jpg')
    print(filenameproba)

    image_color = load_image(filenameproba)

    img_bin = prepare_img_for_roi(image_color)

    selected_regions, letters, region_distances, contours = select_roi_with_distances(image_color.copy(), img_bin)
    print(selected_regions)
    print(letters)
    print(region_distances)
    print(contours)

    print("Broj prepoznatih regiona: ", len(letters))
    display_image(selected_regions)
    imgg = selected_regions.copy()
    cv2.drawContours(imgg, contours, -1, (255, 0, 0), 1)

    print(len(ALPHABET))
    inputs = prepare_for_ann(letters)
    outputs = convert_output(ALPHABET)
    ann = create_ann(output_size=len(ALPHABET))
    ann = train_ann(ann, inputs, outputs, epochs=EPOCHS)

    ann.save("./TEXT_MODEL.h5")


def detect(answer, ss=True, img_path=None):
    ann = keras.models.load_model("./TEXT_MODEL.h5")

    detect_with_model(ann, answer, ss, img_path)


def detect_with_model(model, answer, ss=True, img_path=None):
    # time.sleep(1)
    dirname = os.path.dirname(__file__)
    X = []
    answers = []
    if ss:

        imm = pyautogui.screenshot()
        # imm.save(os.path.join(dirname, 'songpop_text\screen1.png'))

        # im = load_image('C:\\Users\Korisnik\PycharmProjects\proj\songpop_text\screenshot5.jpg')
        # im = load_image(os.path.join(dirname, 'songpop_text\screen1.png'))

        im = np.array(imm)
    else:
        im = load_image(os.path.join(dirname, img_path))

    for i in range(4):
        X.append(im[DIMS[i][0]:DIMS[i][1], DIMS[i][2]:DIMS[i][3]])
        # display_image(X[i])

        scale_percent = 182.95  # percent of original size
        width = int(X[i].shape[1] * scale_percent / 100)
        height = int(X[i].shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        resized = cv2.resize(X[i], dim, interpolation=cv2.INTER_CUBIC)
        # display_image(resized)

        eh = prepare_img_for_roi(resized)
        # cv2.imshow("www", eh)
        # cv2.waitKey()
        selected_regions1, letters1, distances1, contours1 = select_roi_with_distances(resized.copy(), eh)
        print("Broj prepoznatih regiona: ", len(letters1))
        if i == 2: display_image(selected_regions1)

        # neophodno je da u K-Means algoritam bude prosledjena matrica u kojoj vrste odredjuju elemente
        # distances1 = np.array(distances1).reshape(len(distances1), 1)
        #
        # print(distances1)
        #
        # k_means = KMeans(n_clusters=2)
        # k_means.fit(distances1)
        #
        inputs1 = prepare_for_ann(letters1)
        results1 = model.predict(np.array(inputs1, np.float32))
        # print(display_result_with_spaces(results1, alphabet, k_means))
        # answers.append(display_result_with_spaces(results1, alphabet, k_means))
        res = ""
        for j, letter in enumerate(results1):
            res += ALPHABET[winner(letter)]
            if j < len(letters1) - 1 and distances1[j] > 7:
                res += " "
        print(res)
        answers.append(res)

    solutions = answer.split("--")
    results = []
    for a in solutions:
        score, resultText, index = get_most_similar(a, answers)
        results.append((score, resultText, index))
    result = max(results, key=lambda x: x[0])
    index = result[2]
    resultText = result[1]
    score = result[0]
    print(score, resultText)

    # cv2.imshow("Winner", X[index])
    # cv2.waitKey()
    pyautogui.moveTo(DIMS[index][2] + 10, DIMS[index][0] + 10)
    pyautogui.click()


class OCR:
    def __init__(self, model=None):
        self.model = keras.models.load_model(model)

    def click_closest_answer(self, answer):
        detect_with_model(self.model, answer)


if __name__ == '__main__':
    # input = naziv snimka u bazi koji je algoritam prepoznao
    input = "Years & Years"

    # train_and_savemodel()
    detect(input, ss=False, img_path='songpop_text\\+.png')
