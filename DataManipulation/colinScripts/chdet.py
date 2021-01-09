import cv2
import numpy as np
import os
import time
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

    video_dir = "/home/ubuntu/PycharmProjects/ObjectDetectionTraining/data/PEViD-UHD/exchanging_bags_day_indoor_1/"
   
    for filename in os.listdir(video_dir):

        # TODO: DELETE THIS (just makes sure we only read in one single video frame) #####
        if filename != "Exchanging_bags_day_indoor_1_original.mp4":
            continue
        ##################################################################################

        try:
            cap = cv2.VideoCapture(os.path.join(video_dir, filename))
        except:
            continue
        if not cap.isOpened():
            print("Error opening file " + filename)
            continue

        startTime = time.time()

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
           
           
            if i % CHANGE_RATIO:
    #            diff = np.clip(diff, 0, 255)
                preds = boundingBox(currcolor, diff)
                predictionsList += preds
    #            dim = (diff.shape[1] // 4, diff.shape[0] // 4)
    #            small = cv2.resize(currcolor, dim, interpolation=cv2.INTER_AREA)
    #            cv2.imshow("Movement Mask", small)
    #            cv2.waitKey(1)
            else:  # If anything breaks, it's probably because of this
                print(f"segment(diff.shape, (WINDOW_W, WINDOW_H)): {segment(diff.shape, (WINDOW_W, WINDOW_H))}")
                for x, y in segment(diff.shape, (WINDOW_W, WINDOW_H)):
                    preds, _ = rcnn(nextcolor[x : x+WINDOW_W, y : y+WINDOW_H])
                if i % 100 == 0:
                    print(filename, i)

                    predictionsList.append((i, x, y, preds))
           
           
            currcolor = nextcolor
            curr = next
            valid, nextcolor = cap.read()

        print('TIME', filename, time.time() - startTime)

        #return predictionsList
        try:
            os.makedirs('predictions')
        except OSError:
            pass
        with open('predictions/' + filename.split('.')[0] + '.py', 'w') as f:
            f.write(repr(predictionsList))


        with open("runtime", 'a') as f:
            f.write(filename + ': ' + str(time.time()-startTime) + '\n')

################# boundingBox.py #####################
# Adapted from https://gist.github.com/bigsnarfdude/d811e31ee17495f82f10db12651ad82d

import cv2
import numpy as np

from motionDetection import rcnn
from combine import greedy as combine

MIN_SIZE = 20
BUFFER = 2
SENSITIVITY = 5

def boundingBox(colorimg, bwimg, i):

    orig = colorimg.copy()
   
    bwimg = np.clip(bwimg * SENSITIVITY, 0, 255)
    bwimg = bwimg.astype(np.uint8)
   
    #threshold image
    ret, threshed_img = cv2.threshold(bwimg,
                            127, 255, cv2.THRESH_BINARY)
   
    # find countours
    contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   
    rois = [] # x0, y0, x1, y1
   
    for c in contours:
   
        x, y, w, h = cv2.boundingRect(c)
   
        if w + h > MIN_SIZE:
            x0 = int(x - w *  BUFFER)
            x1 = int(x + w * (BUFFER + 1))
            y0 = int(y - h *  BUFFER)
            y1 = int(y + h * (BUFFER + 1))
            cv2.rectangle(colorimg, (x0, y0), (x1, y1), (0, 0, 255), 2)
            rois.append((x0, y0, x1, y1))
       
       
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
       
        box = np.int0(box)

    combinedRois = combine(rois, orig.shape)
    #print(combinedRois)
   
   
    preds = []
   
    for r in combinedRois:
        if (r[3] - r[1]) * (r[2] - r[0]) > 160:
            cv2.rectangle(colorimg, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 4)
            try:
                pred, _ = rcnn(orig[r[1]:r[3], r[0]:r[2]])
                #pred, colorimg[r[1]:r[3], r[0]:r[2]] = rcnn(orig[r[1]:r[3], r[0]:r[2]])
                preds.append((i, r[0], r[1], pred))
            except Exception as e:
                print(e)
       
        else:
           
            pass
       
#    dim = (colorimg.shape[1] // 4, colorimg.shape[0] // 4)
#    small = cv2.resize(colorimg, dim, interpolation=cv2.INTER_AREA)
#    cv2.imshow("Moving Objects", small)
#    cv2.waitKey(1)
   
    while False:
#    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == 65:
            cv2.destroyAllWindows()
            exit(0)
   
#    cv2.destroyAllWindows()
   
    return preds
