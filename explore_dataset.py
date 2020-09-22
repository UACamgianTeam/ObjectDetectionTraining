from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


dataDir = 'coco_2017'
dataType = 'val'
ann_file = f'{dataDir}/annotations/instances_{dataType}2017.json'

# Define the classes (out of the 81) which you want to see. Others will not be shown.
filterClasses = ['laptop', 'tv', 'cell phone']


def test_tfrecord():
    # Get the raw dataset of TFRecords
    filenames = glob.glob('data/*.tfrecord')
    coco_1class_train_name = [i for i in filenames if 'coco_2017_train_1class.tfrecord' in i]
    raw_dataset = tf.data.TFRecordDataset(coco_1class_train_name)
    print(raw_dataset)

    # Parse these serialized tensors using tf.train.Example.ParseFromString
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)



def get_class_name(class_id, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == class_id:
            return cats[i]['name']
    return "None"


def createBoundingBoxesForRandomImage():
    # Load random image and mask.
    image_id = random.choice(dataset.image_ids)
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    # Compute Bounding box
    bbox = utils.extract_bboxes(mask)

    # Display image and additional stats
    print("image_id ", image_id, dataset.image_reference(image_id))
    log("image", image)
    log("mask", mask)
    log("class_ids", class_ids)
    log("bbox", bbox)
    # Display image and instances
    visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)


def getNumAllPossibleCombinations(classes):
    """ Gets all possible combinations of class types in COCO dataset"""
    ########## ALl POSSIBLE COMBINATIONS ########
    classes = ['laptop', 'tv', 'cell phone']

    images = []
    if classes != None:
        # iterate for each individual class in the list
        for className in classes:
            # get all images containing given class
            catIds = coco.getCatIds(catNms=className)
            imgIds = coco.getImgIds(catIds=catIds)
            images += coco.loadImgs(imgIds)
    else:
        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)

    # Now, filter out the repeated images
    unique_images = []
    for i in range(len(images)):
        if images[i] not in unique_images:
            unique_images.append(images[i])

    dataset_size = len(unique_images)

    print("Number of images containing the filter classes:", dataset_size)

# Initialize the COCO API for instance annotations
coco = COCO(ann_file)

# Fetch class IDs only corresponding to the filterClasses
catIds = coco.getCatIds(catNms=filterClasses)
# Get all images containing the above Category IDs
imgIds = coco.getImgIds(catIds=catIds)
print("Number of images containing ALL the filter classes: ", len(imgIds))

# load a random image
img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
imageToShow = io.imread(f"{dataDir}/{dataType}2017/{img['file_name']}") / 255.0

# Load and display instance annotations
plt.imshow(imageToShow)
plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
# plt.show()

getNumAllPossibleCombinations(classes=filterClasses)
