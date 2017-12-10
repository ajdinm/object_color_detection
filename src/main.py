import classifier

clf = classifier.load_classifier()
test = [[1, 0, 0], [0, 1, 0]]
print clf.predict(test)
