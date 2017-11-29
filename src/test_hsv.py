import cv2
import numpy as np

colors = [np.uint8([[[255, 0, 0]]]),
          np.uint8([[[0, 0, 255]]]),
          np.uint8([[[0, 255, 0]]])]
          
for col in colors:
    hsv_col = cv2.cvtColor(col, cv2.COLOR_BGR2HSV)
    print col, hsv_col
