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

    filteredRois = []
    for r in combinedRois:
        if (r[3] - r[1]) * (r[2] - r[0]) > 160:
            filteredRois.append(r)

    return filteredRois