import cv2 as cv
import numpy as np
import time
import random
from sklearn import linear_model

def lineFromPoints(pt1, pt2):
    m = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
    b = pt1[1] - m*pt1[0]
    return m, b

def boundaryLines(eq, delta):
    m = eq[0]
    b = eq[1]
    phi = np.arctan(m)
    theta = 90 - phi
    bDelta = delta / np.cos(theta)

    return b+bDelta, b-bDelta

def ransac(edges):
    if(cv.countNonZero(edges) == 0):
        return (0, 0)
    onlyEdges = cv.findNonZero(edges)
    # print(onlyEdges[:, 0, 1])
    # rows, columns = edges.shape
    # onlyEdges = np.array([[0, 0]])
    # for i in range(rows):
        # for j in range(columns):
            # if edges[i, j] != 0:
                # onlyEdges = np.append(onlyEdges, [[i, j]], axis=0)
            
    # sacc = linear_model.RANSACRegressor()
    # print(np.transpose(onlyEdges[:,0]))
    # sacc.fit(onlyEdges[:, 0, 1], onlyEdges[:, 0, 1])
    # y_line = sacc.predict()
    n = onlyEdges.shape[0]
    delta = 20
    maxN = 0
    s = 0
    bestLine = (0, 0)
    while s < 200:
        s+=1
        # sample 2 points
        index1 = random.randint(0, n-1)
        index2 = random.randint(0, n-1) 
        # while onlyEdges[index2, 0, 0] == onlyEdges[index1, 0, 0]:
            # index2 = random.randint(0, n-1)
        
        pt1 = onlyEdges[index1, 0, :]
        pt2 = onlyEdges[index2, 0, :]

        # find line eq
        (m, b) = lineFromPoints(pt1, pt2)
        (bUpper, bLower) = boundaryLines((m,b), delta)
        # score the line eq
        N = 0
        for i in range(n):
            if (onlyEdges[i, 0, 1] < m*onlyEdges[i, 0, 0] + bUpper) and (onlyEdges[i, 0, 1] > m*onlyEdges[i, 0, 0] + bLower):
                N += 1
        if N > maxN:
            bestLine = (m, b)
            maxN = N
    return bestLine

def main():
    vid = cv.VideoCapture(0)
    font = cv.FONT_HERSHEY_SIMPLEX
    fps = 0
    while True:
        start = time.time()
        _flag, img = vid.read()
        edges = cv.Canny(img, 400, 400)
        (m, b) = ransac(edges)
        pt1 = (-100, int(m*(-100)+b))
        pt2 = (800, int(m*(800)+b))
        cv.line(img, pt1, pt2, color=(0,0,255), thickness=5)
        cv.putText(img, 'FPS = '+ str(int(fps)), (450,50), font, 1, (255, 255, 255), 2, cv.LINE_AA)
        cv.imshow('video', img)
        end = time.time()
        fps = 1 / (end-start)

        ch = cv.waitKey(5)
        if ch == 27:
            break
    print('Done')

def test():
    a = random.randint(0, 100)
    b = random.randint(0, 100)
    print(a)
    a = random.randint(0, 100)
    print(a)

if __name__ == "__main__":
    main()
    cv.destroyAllWindows()