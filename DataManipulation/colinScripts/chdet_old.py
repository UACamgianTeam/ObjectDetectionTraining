import cv2
import numpy as np
from boundingBox import boundingBox
from combine import segment
from motionDetection import rcnn

MEAN_NORM = 128
SD_NORM = 48
ALPHA = 0.05
BETA = 0.05

CHANGE_RATIO = 10
WINDOW_W = 608
WINDOW_H = 608

def normalize(matrix):
    dtype = matrix.dtype
    mean, sd = cv2.meanStdDev(matrix)
    mu = cv2.sumElems(mean)[0]
    sigma = cv2.sumElems(sd)[0]
    a = SD_NORM / sigma
    b = MEAN_NORM - SD_NORM * mu / sigma
    matrix = a * matrix + b
    return matrix.astype(dtype)
    
    
if __name__ == '__main__':
    
    cap = cv2.VideoCapture("/home/ubuntu/PycharmProjects/ObjectDetectionTraining/data/PEViD-UHD/exchanging_bags_day_indoor_1/Exchanging_bags_day_indoor_1_original.mp4")
    if not cap.isOpened():
        print("Error opening file")
        quit()
    
    _, currcolor = cap.read()
    curr = cv2.cvtColor(currcolor, cv2.COLOR_RGB2GRAY)
    
    curr = normalize(curr)
    valid, nextcolor = cap.read()
    
    meanMat = curr# * 1.0
    sdMat = np.zeros(curr.shape, dtype=np.float64)
    
    predictionsList = []
    i = 0
    
    while valid:
        next = cv2.cvtColor(nextcolor, cv2.COLOR_RGB2GRAY)
        
        next = normalize(next)
        i += 1
        diff = np.empty(curr.shape, dtype=curr.dtype)
        
        ### Difference ###
        diff = cv2.absdiff(nextcolor, currcolor)
        diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        
        ### Ratioing ###
#        r1 = cv2.divide(next, curr, scale=32)
#        r2 = cv2.divide(curr, next, scale=32)
#        diff = cv2.absdiff(r2, r1)

        ### Mean / SD ###
#        diff = cv2.absdiff(meanMat, next)#.astype(np.float64))
#        d1 = diff * (1 - ALPHA)
#        d2 = sdMat * ALPHA
#        sdMat = cv2.add(d1, d2).astype(np.uint8)
#        d3 = next * (1 - ALPHA)
#        d4 = meanMat * ALPHA
#        meanMat = cv2.add(d3, d4).astype(np.uint8)
#        r1 = cv2.divide(sdMat, diff, scale=16)
#        r2 = cv2.divide(diff, sdMat, scale=16)
#        diff = cv2.absdiff(r1, r2)

        mean, sd = cv2.meanStdDev(diff)
        nz = cv2.countNonZero(diff)
#        print(str(mean) + ' ' + str(sd) + ' ' + str(nz))
        
        min, max, _, _ = cv2.minMaxLoc(diff)

        preds = None
        
        if i % CHANGE_RATIO:
#            diff = np.clip(diff, 0, 255)
            preds = boundingBox(currcolor, diff)
#            dim = (diff.shape[1] // 4, diff.shape[0] // 4)
#            small = cv2.resize(currcolor, dim, interpolation=cv2.INTER_AREA)
#            cv2.imshow("Movement Mask", small)
#            cv2.waitKey(1)
        else:  # If anything breaks, it's probably because of this
            for x, y in segment(diff.shape, (WINDOW_W, WINDOW_H)):
                preds, _ = rcnn(nextcolor[x : x+WINDOW_W, y : y+WINDOW_H])
                
        predictionsList.append(preds)
            
        
        currcolor = nextcolor
        curr = next
        valid, nextcolor = cap.read()
        
#    return predictionsList




