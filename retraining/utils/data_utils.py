import matplotlib
import matplotlib.pyplot as plt

import os
import random
import io
import imageio
import glob
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Javascript
from IPython.display import Image as IPyImage
from lxml import etree

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


def load_bboxes_into_numpy_array(path):
    """Load bounding boxes from an annotation file into a numpy array.

    Puts bounding boxes into numpy array to feed into tensorflow graph.
    Note that we put

    Args:
    path: a file path.

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """
    # TODO: Fill this function out!
    return None


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
    path: a file path.

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None):
    """Wrapper function to visualize detections.

  Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_name: a name for the image file.
  """
    image_np_with_annotations = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_annotations,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.8)
    if image_name:
        plt.imsave(image_name, image_np_with_annotations)
    else:
        plt.imshow(image_np_with_annotations)


def load_images_and_bboxes(train_dir, train_annotation_dir):
    """
    Loads images and their bounding boxes from train_dir and train_annotation_dir, respectively
    Args:
        train_dir: The root directory of the training images
        train_annotation_dir: The root directory of the training image annotations

    Returns:
        train_images_np, gt_boxes_np: A list of training images (np arrays) and a corresponding list of ground
            truth bounding boxes (np arrays)
    """

    train_images_np = []
    gt_boxes_np = []

    # Iterate recursively over annotation files
    for root_name, dir_names, file_names in os.walk(train_annotation_dir, topdown=False):
        for i, file_name in enumerate(file_names):
            if os.path.join(root_name, file_name).endswith('.xml'):
                with open(os.path.join(root_name, file_name), 'r+') as annotation_file:
                    # Parse the annotation file
                    tree = etree.parse(annotation_file)

                    # Get the root of the annotation file
                    root = tree.getroot()

                    # Get the width and height of the image
                    image_size = root.find('size')
                    image_width = image_size.find('width').text
                    image_height = image_size.find('height').text

                    # Iterate over each 'object' (bounding box + class name) in the file
                    for obj in root.findall('object'):

                        # Get the class name of the bbox
                        obj_name = obj.find('name').text

                        # Get the bounding box itself
                        bbox = obj.find('bndbox')

                        # Get the bounding box coordinates from the bbox
                        xmin = float(bbox.find('xmin'))
                        ymin = float(bbox.find('ymin'))
                        xmax = float(bbox.find('xmax'))
                        ymax = float(bbox.find('ymax'))


def prepare_training_data(train_dir, train_annotation_dir):

    # 1. Get list of images (np arrays) alongside list of annotations (np arrays)
    train_images_np, gt_boxes = load_images_and_bboxes(train_dir, train_annotation_dir)


if __name__ == "__main__":
    prepare_training_data('../../data/coco_2017/val2017', '../../data/coco_2017/annotations/instance_val_annotation')
