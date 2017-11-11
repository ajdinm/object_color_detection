import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import itemfreq
from scipy import cluster
from scipy.misc import fromimage

data_folder = '../data/'
imgs = [('coke.jpg', [570, 330, 120, 400], 3),
        ('detergent.jpg', [545, 330, 220, 405], 3),
        ('palmolive_green.jpg', [580, 430, 130, 310], 3),
        ('red_cup.jpg', [570, 550, 145, 180], 3)]

# make rects bigger
for i in range(len(imgs)):
    imgs[i][1][0] = imgs[i][1][0] - 50
    imgs[i][1][1] = imgs[i][1][1] - 50
    imgs[i][1][2] = imgs[i][1][2] + 100
    imgs[i][1][3] = imgs[i][1][3] + 100

for img in imgs:
    rect = img[1]
    K = img[2]
    filename = data_folder + img[0]
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x, y, w, h = rect
    img = img[y:y+h, x:x+w]

    original = np.array(img)

    mask = np.zeros(img.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
     
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    zero_indx = np.where(thresh==0)
    img[zero_indx] = [0, 0, 0]

    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    freq = itemfreq(label)
    dominant_color = center[max(freq, key=lambda x: x[1])[0]]
    #print '\n\ndominant color:\t', dominant_color, '\ncenter\t', center
    fig, ax = plt.subplots(nrows=1, ncols=3)
    fig.suptitle(filename)

    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.imshow(original)

    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(img)
    
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.imshow(res2)
    
    plt.show()
