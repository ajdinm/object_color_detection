from os import walk
import csv
from sklearn.neural_network import MLPClassifier

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

data = read_data()
x = map(lambda x: x[1], data) * 5
y = map(lambda x: color_lookup(x[0]), data) * 5

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (5, 3), random_state=1, activation='logistic')
clf.fit(x, y)
test = [[71, 235, 104], [235, 197, 71], [235, 71, 71]]
# green, yellow, red
acc = map(lambda i, j: same_vector(i, j), clf.predict(x), y)
print acc.count(True) / float(len(acc))
