import cv2
import numpy as np
import os
import time
import json
from boundingBox import boundingBox

inputsFolder = "/home/ubuntu/PycharmProjects/ObjectDetectionTraining/data/PEViD-UHD/exchanging_bags_day_indoor_1/"  # Be sure to add a '/' after the folders
annotationsFolder = "/home/ubuntu/PycharmProjects/ObjectDetectionTraining/data/PEViD-UHD/exchanging_bags_day_indoor_1/annotations/"
imgFolder = "/home/ubuntu/PycharmProjects/ObjectDetectionTraining/data/PEViD-UHD/exchanging_bags_day_indoor_1/annotations/crops/"

MEAN_NORM = 128
SD_NORM = 48


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

    for filename in os.listdir(inputsFolder):
        try:
            cap = cv2.VideoCapture(inputsFolder + filename)
            fn = filename.split('.')[0]
            with open(annotationsFolder + fn + ".json", 'r') as fp:
                truth = json.load(fp)
        except Exception:
            print('File ' + filename + ' is not a video')
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

        img_id = 0
        ann_id = 0
        frame = 0
        annotations = []

        while valid:
            next = cv2.cvtColor(nextcolor, cv2.COLOR_RGB2GRAY)

            next = normalize(next)
            diff = np.empty(curr.shape, dtype=curr.dtype)

            ### Difference ###
            diff = cv2.absdiff(nextcolor, currcolor)
            diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

            rois = boundingBox(currcolor, diff, frame)
            for r in rois:
                cv2.imwrite(f"{imgFolder}{fn}_{frame}_{img_id}", currcolor[r[1]:r[3], r[0]:r[2]])
                images.append({
                    'id' : img_id,
                    'height': r[3] - r[1],
                    'width' : r[2] - r[0]
                })
                img_id += 1
                for ann in truth['annotations']:
                    if int(ann['image_id']) == frame:
                        b = (
                            ann['bbox'][0],
                            ann['bbox'][1],
                            ann['bbox'][2] + ann['bbox'][0],
                            ann['bbox'][3] + ann['bbox'][1]
                        )
                        if ((b[0] <= r[2] and r[0] <= b[2])) and ((b[1] <= r[3] and r[1] <= b[3])):
                            n = (
                                max(b[0], r[0]),
                                max(b[1], r[1]),
                                min(b[2], r[2]),
                                min(b[3], r[3])
                            )
                            annotations.append({
                                    'id' : ann_id,
                                    'image_id' : img_id,
                                    'bbox' : (
                                        n[0] - r[0],
                                        n[1] - r[1],
                                        n[2] - n[0],
                                        n[3] - n[1]
                                    )
                            })
                            ann_id += 1

            frame += 1
            currcolor = nextcolor
            curr = next
            valid, nextcolor = cap.read()

            with open(imgFolder + fn + '.json', 'w') as fp:
                json.dump({'images': images, 'annotations': annotations}, fp)