import numpy as np
import cv2 # OpenCV
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import pyautogui
import time
from difflib import SequenceMatcher
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD

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
    return 255-image

def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')

def dilate(image):
    kernel = np.ones((6, 6)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)

def erode(image):
    kernel = np.ones((6, 6)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)

def scale_to_range(image):
    return image/255

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
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(output_size, activation='sigmoid'))
    return ann


def train_ann(ann, X_train, y_train, epochs):
    X_train = np.array(X_train, np.float32)  # dati ulaz
    y_train = np.array(y_train, np.float32)  # zeljeni izlazi na date ulaze

    print("\nTraining started...")
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=0, shuffle=False)
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
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

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
    #print(max(enumerate(k_means.cluster_centers_), key=lambda x: x[1]))
    #print(k_means.cluster_centers_)
    #print(k_means.labels_)
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
    #ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
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



def main(input):
    # treniranje
    dirname = os.path.dirname(__file__)

    filenameproba = os.path.join(dirname, 'songpop_text/probamala.jpg')
    print(filenameproba)

    image_color = load_image(filenameproba)
    img = image_binn(image_grayy(image_color))
    img_bin = erode(img)
    selected_regions, letters, region_distances, contours = select_roi_with_distances(image_color.copy(), img_bin)
    print("Broj prepoznatih regiona: ", len(letters))
    display_image(selected_regions)
    imgg = selected_regions.copy()
    cv2.drawContours(imgg, contours, -1, (255, 0, 0), 1)

    alphabet = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p',
                'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm', '(', ')', '&', '!'
                ]
    print(len(alphabet))
    inputs = prepare_for_ann(letters)
    outputs = convert_output(alphabet)
    ann = create_ann(output_size=30)
    ann = train_ann(ann, inputs, outputs, epochs=2000)



    # prepoznavanje
    time.sleep(2)
    imm = pyautogui.screenshot()
    # dirname = os.path.dirname(__file__)
    imm.save(os.path.join(dirname, 'songpop_text\screen1.png'))

    dims = [[690, 780, 490, 950], [690, 780, 1020, 1480], [800, 890, 490, 950], [800, 890, 1020, 1480]]
    X = []
    answers = []
    # im = load_image('C:\\Users\Korisnik\PycharmProjects\proj\songpop_text\screenshot5.jpg')
    im = load_image(os.path.join(dirname, 'songpop_text\screen1.png'))
    for i in range(4):
        X.append(im[dims[i][0]:dims[i][1], dims[i][2]:dims[i][3]].copy())
        # display_image(X[i])

        scale_percent = 182.95  # percent of original size
        width = int(X[i].shape[1] * scale_percent / 100)
        height = int(X[i].shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        resized = cv2.resize(X[i], dim, interpolation=cv2.INTER_AREA)
        # display_image(resized)

        eh = image_bin(image_gray(resized))
        selected_regions1, letters1, distances1, contours1 = select_roi_with_distances(resized.copy(), erode(eh))
        print("Broj prepoznatih regiona: ", len(letters1))
        if i == 2: display_image(selected_regions1)

        # neophodno je da u K-Means algoritam bude prosledjena matrica u kojoj vrste odredjuju elemente
        distances1 = np.array(distances1).reshape(len(distances1), 1)

        k_means = KMeans(n_clusters=2)
        k_means.fit(distances1)

        inputs1 = prepare_for_ann(letters1)
        results1 = ann.predict(np.array(inputs1, np.float32))
        print(display_result_with_spaces(results1, alphabet, k_means))
        answers.append(display_result_with_spaces(results1, alphabet, k_means))

    result, resultText, index = get_most_similar(input, answers)
    print(result, resultText)
    pyautogui.moveTo(dims[index][2] + 10, dims[index][0] + 10)
    pyautogui.click()

if __name__ == '__main__':
    #input = naziv snimka u bazi koji je algoritam prepoznao
    input = "2U"
    main(input)















