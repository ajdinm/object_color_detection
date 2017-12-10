from os import walk
import csv

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold

from collections import Counter
from random import shuffle

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
    lookup = dict()
    lookup['red'] = [1, 0, 0, 0]
    lookup['green'] = [0, 1, 0, 0]
    lookup['yellow'] = [0, 0, 1, 0]
    lookup['blue'] = [0, 0, 0, 1]

    return lookup[color_name]

def same_vector(x, y):
    for i in range(len(x)):
        if x[i] != y[i]:
            return False
    return True

def normalize_rgb(rgb):
    rgb = map(lambda x: x / 255., rgb)
    return rgb

data = read_data()
shuffle(data)
color_names = ['red', 'blue', 'yellow', 'green']

# red, blue, yellow, green
colors = map(lambda color: filter(lambda x: x[0] == color, data), color_names)

min_len = min(map(lambda x: len(x), colors))

x = []
y = []
for i in range(len(color_names)):
    x = x + colors[i][:min_len]
    y = y + [color_names[i]] * min_len

x = map(lambda xi: normalize_rgb(xi[1]), x)
y = map(lambda yi: color_lookup(yi), y)
 
validation_x = []
validation_y = []
for i in range(len(color_names)):
    validation_x = validation_x + colors[i][min_len:]
    validation_y = validation_y + [color_names[i]] * (len(colors[i]) - min_len)

validation_x = map(lambda xi: normalize_rgb(xi[1]), validation_x)
validation_y = map(lambda yi: color_lookup(yi), validation_y)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (5, 3), random_state=1, activation='relu')

clf.fit(x, y)
test = [[71, 235, 104], [235, 197, 71], [235, 71, 71]]
test = map(lambda x: normalize_rgb(x), test)
# green, yellow, red
acc = map(lambda i, j: same_vector(i, j), clf.predict(validation_x), validation_y)
print acc.count(True) / float(len(acc))
