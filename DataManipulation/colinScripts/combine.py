import numpy as np
from math import sqrt

def overlap(xl0, yl0, xr0, yr0, xl1, yl1, xr1, yr1):
    if yr1 < yl0 or yr0 < yl1 or xr1 < xl0 or xr0 < xl1:
        return False
    a0 = (xr0 - xl0) * (yr0 - yl0)
    a1 = (xr1 - xl1) * (yr1 - yl1)
    x = [xl0, xr0, xl1, xr1]
    y = [yl0, yr0, yl1, yr1]
    x.sort()
    y.sort()
    insct = (x[2] - x[1]) * (y[2] - y[1])
    bound = (x[3] - x[0]) * (y[3] - y[0])
    aff = (a0 + a1 - insct) / bound
    return aff > sqrt(bound) / 700


def center(box: tuple):
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)


def distance(p: tuple, q: tuple):
    return sqrt((p[0]-q[0]) ** 2 + (p[1]-q[1]) ** 2)


def combine(rois: list, _):
    newlist = []
    continueFlag = True
    while continueFlag is True:
        continueFlag = False
        for i in range(len(rois)):
            for j in range(i, len(rois)):
                if i != j:
                    if rois[i] is not None and rois[j] is not None:
                        v = overlap(rois[i][0], rois[i][1],
                                    rois[i][2], rois[i][3],
                                    rois[j][0], rois[j][1],
                                    rois[j][2], rois[j][3])
                        if v:
                            rois[i] = (min(rois[i][0], rois[j][0]),
                                       min(rois[i][1], rois[j][1]),
                                       max(rois[i][2], rois[j][2]),
                                       max(rois[i][3], rois[j][3]))
                            rois[j] = None
                            continueFlag = True
            if rois[i] is not None:
                newlist.append(rois[i])
        rois = newlist.copy()
        newlist.clear()
    return rois


def windows(rois: list, shape: tuple):
    MAX_W = 400
    MAX_H = 300
    xboxes = (shape[1] - 1) // MAX_W + 1
    yboxes = (shape[0] - 1) // MAX_H + 1
    boxes = np.zeros((yboxes, xboxes), dtype=np.bool_)
    for r in rois:
        boxes[ int(r[1] / MAX_H) ][ int(r[0] / MAX_W) ] = True
        boxes[ int(r[3] / MAX_H) ][ int(r[2] / MAX_W) ] = True
        boxes[ int(r[1] / MAX_H) ][ int(r[2] / MAX_W) ] = True
        boxes[ int(r[3] / MAX_H) ][ int(r[0] / MAX_W) ] = True
    newlist = []
    for i in range(yboxes):
        for j in range(xboxes):
            if boxes[i][j]:
                newlist.append((j * MAX_W, i * MAX_H, (j+1) * MAX_W, (i+1) * MAX_H))
    return newlist


def greedy(rois: list, _):
    MAX_W = 400
    MAX_H = 300
    newlist = []
    continueFlag = True
    while continueFlag:
        continueFlag = False
        min_dist = sqrt(MAX_W ** 2 + MAX_H ** 2) / 2
        t = -1
        for i in range(len(rois)):
            for j in range(i, len(rois)):
                if i != j:
                    if rois[i] is not None and rois[j] is not None:
                        curr_dist = distance(center(rois[i]), center(rois[j]))
                        if curr_dist < min_dist:
                            t = j
                            min_dist = curr_dist
                            continueFlag = True
            if t >= 0:
                if rois[i] is not None and rois[t] is not None:
                    rois[i] = (min(rois[i][0], rois[t][0]),
                               min(rois[i][1], rois[t][1]),
                               max(rois[i][2], rois[t][2]),
                               max(rois[i][3], rois[t][3]))
                    rois[t] = None
            if rois[i] is not None:
                newlist.append(rois[i])
        rois = newlist.copy()
        newlist.clear()
    return rois


def segment(shape: tuple, window: tuple=(608, 608)):
    xlen = int(shape[1] - window[1])
    ylen = int(shape[0] - window[0])
    if xlen <= 0 or ylen <= 0:
        1/0 # idk how to make exceptions
    xwins = xlen // (xlen // window[1] + 1)
    ywins = ylen // (ylen // window[0] + 1)
    return (range(0, xlen + 1, ywins), range(0, ylen + 1, ywins))