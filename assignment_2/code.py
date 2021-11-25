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

def ransac(onlyEdges):

    n = onlyEdges.shape[0]
    delta = 10
    maxN = 0
    s = 0
    if n == 0:
        return 0, 0, onlyEdges
    bestPts = (np.zeros(2), np.zeros(2))
    while s < 15:
        s+=1

        # sample 2 points
        index1 = random.randint(0, n-1)
        index2 = random.randint(0, n-1) 
        
        pt1 = np.array(onlyEdges[index1, :])
        pt2 = np.array(onlyEdges[index2, :])

        # find line eq
        # (m, b) = lineFromPoints(pt1, pt2)
        # (bUpper, bLower) = boundaryLines((m,b), delta)

        # score the line eq
        N = 0
        for i in range(n):
            pt3 = np.array(onlyEdges[i, :])
            if np.abs(np.cross(pt2-pt1,pt3-pt1)/np.linalg.norm(pt2-pt1)) < delta:
                N += 1
        if N > maxN:
            bestPts = (pt1, pt2)
            maxN = N

    # linear fit inliers
    X = np.empty(maxN)
    Y = np.empty(maxN)
    (pt1, pt2) = (bestPts[0], bestPts[1])
    counter = 0
    iRange = []
    for i in range(n):
        pt3 = np.array(onlyEdges[i, :])
        dist = np.abs(np.cross(pt2-pt1,pt3-pt1)/np.linalg.norm(pt2-pt1))
        if dist < delta:
            X[counter] = (onlyEdges[i, 0])
            Y[counter] = (onlyEdges[i, 1])
            iRange.append(i)
            counter += 1
    if X.size == 0 or Y.size == 0:
        return 0, 0, onlyEdges
    poly = np.polyfit(X, Y, 1)
    onlyEdges = np.delete(onlyEdges, iRange, 0)
    return poly[0], poly[1], onlyEdges

def vertices(linecoords):
    minDiff = np.inf
    vortex = np.empty((4, 3))
    for i in range(1,4):
        diff = np.abs(linecoords[0, 0] - linecoords[i, 0])
        if diff < minDiff:
            i1 = i
            minDiff = diff
    firstIteration = True
    for i in range(1,4):
        if i != i1 and firstIteration:
            vortex[0, :] = np.cross(linecoords[0, :], linecoords[i, :])
            vortex[1, :] = np.cross(linecoords[i1, :], linecoords[i, :])
            firstIteration = False
        elif i != i1:
            vortex[2, :] = np.cross(linecoords[0, :], linecoords[i, :])
            vortex[3, :] = np.cross(linecoords[i1, :], linecoords[i, :])
    for i in range(4):
        vortex[i, :] = vortex[i, :] / vortex[i, 2]
    vortex = vortex.astype(int)
    return np.delete(vortex, 2, 1)
            
    
    
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
        
        onlyEdges = cv.findNonZero(edges)
        U = np.empty((4,3))
        if(cv.countNonZero(edges) != 0):
            n = onlyEdges.shape[0]
            onlyEdges = onlyEdges[:n:25, 0]
            for i in range(4):
                if(cv.countNonZero(edges) == 0):
                    b = 0
                    m = 0
                    break
                else:
                    (m, b, onlyEdges) = ransac(onlyEdges)
                    U[i, :] = (m, -1, b)
                    C = 1000
                    pt1 = (C, int(m*C+b))
                    pt2 = ((-1)*C, int(m*(-1)*C+b))
                    cv.line(img, pt1, pt2, color=(0,0,255), thickness=2)
        vert = vertices(U)
        '''
        pts_dst = np.array([[23, 97], [50, 428], [587, 448], [599, 85]])
        h, status = cv.findHomography(vert, pts_dst)
        img_out = cv.warpPerspective(img, h, (img.shape[0], img.shape[1]))
        '''
        for i in range(4):
            cv.circle(img, vert[i, :], 7, (0, 255, 0), 3)


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