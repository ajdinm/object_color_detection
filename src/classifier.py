from os import walk
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold
from collections import Counter
from random import shuffle

import pickle

def read_data(): 
    data_folder = '../data/colors/'
    files = []
    for (dirpath, dirnames, filenames) in walk(data_folder):
        files.extend(filenames)
        break
    #files = map(lambda x: data_folder + x, files)
    data = []
    for f in files:
        with open(data_folder + f) as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar = '|')
            name = f[:f.find('.')]
            for row in reader:
                try: 
                    data.append([name, map(lambda x: int(x), row[1].split(';'))])
                except:
                    continue
    return data

def color_lookup(color_name):
    lookup['red'] = [1, 0, 0, 0]
    lookup['green'] = [0, 1, 0, 0]
    lookup['yellow'] = [0, 0, 1, 0]
    lookup['blue'] = [0, 0, 0, 1]

    return lookup[color_name]

def reverse_lookup(predicted):
    lookup = [['red', [1, 0, 0, 0]], ['green', [0, 1, 0, 0]], ['yellow', [0, 0, 1, 0]], ['blue', [0, 0, 0, 1]]]
    predicted = list(predicted)
    for x in lookup:
        if x[1] == predicted:
            return x[0]

    return None

def same_vector(x, y):
    for i in range(len(x)):
        if x[i] != y[i]:
            return False
    return True

def normalize_rgb(rgb):
    rgb = map(lambda x: x / 255., rgb)
    return rgb

def get_train_validation_data():
    
    data = read_data()
    shuffle(data)
    color_names = ['red', 'blue', 'yellow', 'green']

    # red, blue, yellow, green
    colors = map(lambda color: filter(lambda x: x[0] == color, data), color_names)
    
    # ensure same distribution of all classes
    min_len = min(map(lambda x: len(x), colors))

    x = []
    y = []
    for i in range(len(color_names)):
        x = x + colors[i][:min_len]
        y = y + [color_names[i]] * min_len

    x = map(lambda xi: normalize_rgb(xi[1]), x) # put everything in [0, 1] range
    y = map(lambda yi: color_lookup(yi), y)
     
    validation_x = []
    validation_y = []
    for i in range(len(color_names)):
        validation_x = validation_x + colors[i][min_len:]
        validation_y = validation_y + [color_names[i]] * (len(colors[i]) - min_len)

    validation_x = map(lambda xi: normalize_rgb(xi[1]), validation_x)
    validation_y = map(lambda yi: color_lookup(yi), validation_y)

    return x, y, validation_x, validation_y

def train_classifier():

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (5, 3), random_state=1, activation='relu')

    train_x, train_y, valid_x, valid_y = get_train_validation_data()

    clf.fit(train_x, train_y)
    #acc = map(lambda i, j: same_vector(i, j), clf.predict(valid_x), valid_y)
    #print acc.count(True) / float(len(acc))


def save_classifier(clf):
    filename = 'classifier.pkl'
    with open(filename, 'wb') as class_file:
        pickle.dump(clf, class_file)

def load_classifier():
    filename = 'classifier.pkl'
    with open(filename, 'rb') as class_file:
        return pickle.load(class_file)
    return None
