import cv2 as cv
import numpy as np
import time


def main():

    classesFile = 'coco.names'
    classNames = []

    with open(classesFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    modelConfiguration = 'yolov3.cfg'
    modelWeights = 'yolov3-320.weights'

    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    cam = cv.VideoCapture(0)

    whT = 320
    fps = 0

    def findObjects(outputs, img):
        hT, wT, cT = img.shape
        bbox = []
        classIds = []
        confs = []
        confThreshold = 0.5
        nmsThreshold = 0.3
        notHuman = False
        
        for output in outputs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    w, h = int(det[2]*wT), int(det[3]*hT)
                    x, y = int(det[0]*wT - w/2), int(det[1]*hT - h/2)
                    bbox.append([x,y,w,h])
                    classIds.append(classId)
                    confs.append(float(confidence))
                    if classId != 0:
                        notHuman = True
        
        indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

        for i in indices:
            i = i[0]
            box = bbox[i]
            x,y,w,h = box[0], box[1], box[2], box[3]
            cv.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%', (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return notHuman

    countFrames = 0
    countNonhumans = 0

    while True:
        # timing
        start = time.time()
        
        _success, img = cam.read()

        blob = cv.dnn.blobFromImage(img, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layerNames = net.getLayerNames()
        outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

        outputs = net.forward(outputNames)
        if(findObjects(outputs, img)):
            countNonhumans += 1
        countFrames += 1

        # display
        cv.putText(img, 'FPS = '+ str(int(fps)), (450,50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        cv.imshow('Frame', img)
        cv.waitKey(1)

        # timing
        end = time.time()
        fps = 1 / (end - start)

        print(countFrames)
        print(countNonhumans)


if __name__ == "__main__":
    main()

