import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from typing import List

from object_detection.utils import visualization_utils as viz_utils


def visualize_image_set(images_np: List, gt_boxes: List, category_index: dict, scores=[]) -> None:
    """ Displays the first 16 images and their corresponding annotations for the given
  image set data
  """
    print(scores)
    print(not scores)
    dummy_scores = scores
    plt.figure(figsize=(30, 15))
    for idx in range(16):
        plt.subplot(4, 4, idx + 1)
        # print('ground truths...')
        # print(gt_boxes[idx])
        if not scores:  # set scores to equal 100%
            dummy_scores = [1.0] * len(gt_boxes[idx])
        else:
            dummy_scores = scores[idx]
        #   print('scores...')
        # print(dummy_scores)
        print('scores > 50%:')
        print([score for score in dummy_scores if score > 0.5])
        plot_detections(
            images_np[idx],
            gt_boxes[idx],
            np.ones(shape=[gt_boxes[idx].shape[0]], dtype=np.int32),
            dummy_scores,
            category_index)
    plt.ion()
    plt.show()
    plt.pause(0.001)
    input('Press [enter] to continue.')


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
        min_score_thresh=0.5)  # paper uses 0.3 i believe
    if image_name:
        plt.imsave(image_name, image_np_with_annotations)
    else:
        plt.imshow(image_np_with_annotations)
