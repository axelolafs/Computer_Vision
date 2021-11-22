
# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv
import time
import numpy as np


# built-in module


def main():

    vid = cv.VideoCapture(0)
    font = cv.FONT_HERSHEY_SIMPLEX
    fps = 0
    while True:
        start = time.time()
        _flag, img = vid.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        (_, _, _ , brightestPoint) = cv.minMaxLoc(gray)
        cv.circle(img, brightestPoint, 10, (0, 0, 255), 2)
        cv.putText(img, 'FPS = '+ str(int(fps)), (10,450), font, 3, (0, 255, 0), 2, cv.LINE_AA)
        cv.imshow('cam', img)
        end = time.time()
        fps = 1 / (end - start)


        ch = cv.waitKey(5)
        if ch == 27:
            break
    
    print(a)
    print(b)
    print(c)
    print(d)
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
