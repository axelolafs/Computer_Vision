
# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv
import time
import numpy as np


# built-in module

def brightest(data):
    gray = cv.cvtColor(data, cv.COLOR_BGR2GRAY)
    (_, _, _, point) = cv.minMaxLoc(gray)
    return point

def reddest(data):
    img_hsv = cv.cvtColor(data, cv.COLOR_BGR2HSV)
    lowerBound = np.array([-5, 0, 100])
    upperBound = np.array([5, 255, 255])
    mask = cv.inRange(img_hsv, lowerBound, upperBound)
    (_, _, _, point) = cv.minMaxLoc(img_hsv[:, :, 1], mask)
    return point

def brightestRaw(data):
    gray = cv.cvtColor(data, cv.COLOR_BGR2GRAY)
    (rows, columns) = gray.shape
    brightestValue = 0
    for i in range(rows):
        for j in range(columns):
            if (gray[i,j] > brightestValue):
                brightestPoint = (j, i)
                brightestValue = gray[i, j]
    return brightestPoint

def main():

    vid = cv.VideoCapture(0)
    font = cv.FONT_HERSHEY_SIMPLEX
    fps = 0
    while True:
        start = time.time()
        _flag, img = vid.read()
        point = brightestRaw(img)
        nonraw = brightest(img)
        cv.circle(img, point, 9, (0, 0, 255), 2)
        cv.putText(img, 'FPS = '+ str(int(fps)), (10,450), font, 3, (0, 255, 0), 2, cv.LINE_AA)
        cv.imshow('cam', img)
        end = time.time()
        fps = 1 / (end - start)


        ch = cv.waitKey(5)
        if ch == 27:
            break
    print('Done')
    print(point)
    print(nonraw)

if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
