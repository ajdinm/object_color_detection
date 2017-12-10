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

def get_imgs():

    data_folder = '../dataset/yellow/'
    imgs = [('coke.jpg', [570, 330, 120, 400], 3),
            ('detergent.jpg', [545, 330, 220, 405], 3),
            ('palmolive_green.jpg', [580, 430, 130, 310], 3),
            ('red_cup.jpg', [570, 550, 145, 180], 3)]

    img_paths = []
    for (dirpath, dirnames, filenames) in walk(data_folder):
        img_paths.extend(filenames)

    #imgs = map(lambda x: [data_folder+x, [545, 330, 220, 405], 'yellow'])
    imgs = map(lambda x: [data_folder+x, [495, 280, 320, 505], 'yellow'], img_paths)
    # make rects bigger
    #for i in range(len(imgs)):
    #    imgs[i][1][0] = imgs[i][1][0] - 50
    #    imgs[i][1][1] = imgs[i][1][1] - 50
    #    imgs[i][1][2] = imgs[i][1][2] + 100
    #    imgs[i][1][3] = imgs[i][1][3] + 100

    return imgs

def predict():

    imgs = get_imgs()
    results = []
    for img in imgs:
        rect = img[1]
        filename = img[0]
        img_class = img[2]
        x, y, w, h = rect
        
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[y:y+h, x:x+w] # get only relevant portion of the image
        
        results = results + [[img_class, colorsys.hsv_to_rgb(predict_img_color(img) / 360., 1, 0.5)]]
    x = map(lambda xi: xi[1], results)
    y = map(lambda yi: yi[0], results)
    clf = classifier.load_classifier()
    predictions = map(lambda x: classifier.reverse_lookup(x), clf.predict(x))
    predictions = map(lambda i: [predictions[i], y[i]], range(len(y)))
    failed_predictions = filter(lambda x: x[0] != x[1], predictions)
    print failed_predictions
    print 'accuracy', 1 - float(len(failed_predictions)) / len(predictions)



        
        
def predict_img_color(img):
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        zero_indx = np.where(thresh==0)
        img[zero_indx] = [0, 0, 0]
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        Z = img.reshape((-1,3))
        hues = map(lambda x: x[0], Z)
        hue_counter = Counter(hues)
        if(len(hue_counter) > 1):
            del hue_counter[0]
        return get_hue(max(hue_counter, key=hue_counter.get))

predict()
