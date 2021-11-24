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

def ransac(edges, img):
    bestPts = np.array([0, 0]), np.array([0, 0])
    if(cv.countNonZero(edges) == 0):
        return 0, 0
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
    delta = 10
    maxN = 0
    s = 0
    while s < 20:
        s+=1
        # sample 2 points
        index1 = random.randint(0, n-1)
        index2 = random.randint(0, n-1) 
        # while onlyEdges[index2, 0, 0] == onlyEdges[index1, 0, 0]:
            # index2 = random.randint(0, n-1)
        
        pt1 = np.array(onlyEdges[index1, 0, :])
        pt2 = np.array(onlyEdges[index2, 0, :])
        # print(pt1)
        # print(pt2)
        # find line eq
        # (m, b) = lineFromPoints(pt1, pt2)
        # (bUpper, bLower) = boundaryLines((m,b), delta)
        # score the line eq
        N = 0
        for i in range(n):
            pt3 = np.array(onlyEdges[i, 0, :])
            if np.abs(np.cross(pt2-pt1,pt3-pt1)/np.linalg.norm(pt2-pt1)) < delta:
                N += 1
        if N > maxN:
            bestPts = (pt1, pt2)
            maxN = N

    X = np.empty(maxN)
    Y = np.empty(maxN)
    (pt1, pt2) = (bestPts[0], bestPts[1])
    counter = 0
    for i in range(n):
        pt3 = np.array(onlyEdges[i, 0, :])
        dist = np.abs(np.cross(pt2-pt1,pt3-pt1)/np.linalg.norm(pt2-pt1))
        if dist < delta:
            X[counter] = (onlyEdges[i, 0, 0])
            Y[counter] = (onlyEdges[i, 0, 1])
            cv.circle(img, (int(X[counter]), int(Y[counter])), 0, (0, 0, 255), -1)
            counter += 1
    poly = np.polyfit(X, Y, 1)
    return poly[0], poly[1], img

def main():
    vid = cv.VideoCapture(0)
    font = cv.FONT_HERSHEY_SIMPLEX
    fps = 0

    def nothing(*arg):
        pass

    cv.namedWindow('edge')
    cv.createTrackbar('thrs1', 'edge', 2000, 5000, nothing)
    cv.createTrackbar('thrs2', 'edge', 4000, 5000, nothing)
    while True:
        start = time.time()
        _flag, img = vid.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        thrs1 = cv.getTrackbarPos('thrs1', 'edge')
        thrs2 = cv.getTrackbarPos('thrs2', 'edge')
        edges = cv.Canny(gray, thrs1, thrs2, apertureSize=5)
        (m, b, img) = ransac(edges, img)
        C = 1000
        pt1 = (C, int(m*C+b))
        pt2 = ((-1)*C, int(m*(-1)*C+b))
        cv.line(img, pt1, pt2, color=(0,0,255), thickness=2)
        cv.putText(img, 'FPS = '+ str(int(fps)), (450,50), font, 1, (255, 255, 255), 2, cv.LINE_AA)
        cv.imshow('edge', img)
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