import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import itemfreq
from scipy import cluster
from scipy.misc import fromimage
from collections import Counter
from sklearn.neural_network import MLPClassifier
import colorsys
import classifier
from os import walk


def get_hue(x):
    return x * 360.0 / 179.0


def get_rect(img_color):
    # As BigBIRD dataset gives too large images, crop them to capture relevant
    # part
    if img_color == 'yellow':
        return [495, 280, 320, 505]
    if img_color == 'green' or img_color == 'orange':
        return [580, 430, 130, 310]
    if img_color == 'blue':
        return [512, 245, 242, 504]
    return [570, 550, 145, 180]


def get_imgs():
    # Requires dataset folder to be present in directory above.
    # Inside that folder there should be N folders, named by color name. 
    data_folder = '../dataset/'

    img_paths = []
    for (dirpath, dirnames, filenames) in walk(data_folder):
        color = dirpath.replace(data_folder, '')
        if dirpath != data_folder:
            # get full path to img and expected color
            img_paths.extend(map(lambda x: [dirpath + '/' + x, get_rect(color), color], filenames))

    return img_paths


def predict_imgs():

    imgs = get_imgs()
    results = []
    for img in imgs:
        rect = img[1]
        filename = img[0]
        img_class = img[2]
        x, y, w, h = rect

        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[y:y + h, x:x + w]  # get only relevant portion of the image

        results = results + \
            [[img_class, colorsys.hsv_to_rgb(
                get_dominant_color(img) / 360., 1, 0.5)]]
    x = map(lambda xi: xi[1], results)
    y = map(lambda yi: yi[0], results)
    clf = classifier.load_classifier()
    predictions = map(lambda x: classifier.reverse_lookup(x), clf.predict(x))
    predictions = map(
        lambda i: [predictions[i], y[i], results[i]], range(len(y)))
    failed_predictions = filter(lambda x: x[0] != x[1], predictions)
    print failed_predictions
    print 'accuracy', 1 - float(len(failed_predictions)) / len(predictions)


def predict(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()

    result = [colorsys.hsv_to_rgb(get_dominant_color(img) / 360., 1, 0.5)]
    clf = classifier.load_classifier()
    prediction = classifier.reverse_lookup(clf.predict(result)[0])
    print prediction


def get_dominant_color(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        zero_indx = np.where(thresh == 0)
        img[zero_indx] = [0, 0, 0]

        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        Z = img.reshape((-1, 3))
        hues = map(lambda x: x[0], Z)
        hue_counter = Counter(hues)
        if(len(hue_counter) > 1):
            del hue_counter[0]
        return get_hue(max(hue_counter, key=hue_counter.get))

predict('../test.jpg')
