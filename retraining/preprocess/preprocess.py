import numpy as np
from typing import Tuple, List

from .window import get_windows, get_window_ids, map_box_to_window
from utils.load_data import get_images


# ********** external functions *****************

def preprocess_train_images(images_dict: dict, file_name_dict: dict, train_image_dir: str,
                            train_annotations: dict, category_index: dict,
                            win_set: Tuple[int, int, int, int], verbose: bool) -> Tuple[List, List]:
    """ Preprocesses the training images into a format that the model can be retrained on
    1. constructs a list of all of the windows in the training set and map them to their corresponding image (images_dict)
    2. maps each annotation to its corresponding window(s)
    3. constructs the ground truth boxes for each of the training windows and gets dict of windows with no associated annotations
    4. removes windows and their corresponding "boxes" in which there are no annotations
    * train_images_np: a list of the windows that will be used to retrain the model
    * gt_boxes: The ground truth boxes for each of the windows in train_images_np
    """
    # construct a list of all of the windows in the training set and map them to their corresponding image (images_dict)
    (train_images_np, images_dict) = construct_images_np(images_dict, file_name_dict, train_image_dir, verbose, win_set)
    # map each annotation to its corresponding window(s)
    images_dict = map_annotations_to_windows(images_dict, train_annotations, category_index, verbose)
    # construct the ground truth boxes for each of the training windows and get dict of windows with no associated annotations
    (gt_boxes, no_annotation_ids) = construct_gt_boxes(images_dict, len(train_images_np), verbose)
    # remove windows and their corresponding "boxes" in which there are no annotations
    # remove in-place using list comprehensions to avoid copying large arrays
    train_images_np[:] = [train_image_np for index, train_image_np in enumerate(train_images_np) if
                          not index in no_annotation_ids]
    gt_boxes[:] = [gt_box for index, gt_box in enumerate(gt_boxes) if not index in no_annotation_ids]

    return (train_images_np, gt_boxes)


def preprocess_test_images(test_images_dict: dict, file_name_dict: dict, test_image_dir: str,
                           test_annotations: dict, category_index: dict,
                           win_set: Tuple[int, int, int, int], verbose: bool) -> Tuple[List, List, dict]:
    """
    """
    # construct a list of all of the windows in the training set and map them to their corresponding image (images_dict)
    (test_images_np, test_images_dict) = construct_images_np(test_images_dict, file_name_dict, test_image_dir, verbose,
                                                             win_set)

    # map each annotation to its corresponding window(s)
    test_images_dict = map_annotations_to_windows(test_images_dict, test_annotations, category_index, verbose)

    # construct the ground truth boxes for each of the training windows and get dict of windows with no associated annotations
    (gt_boxes, no_annotation_ids) = construct_gt_boxes(test_images_dict, len(test_images_np), verbose)

    print(test_images_dict)

    return (test_images_np, gt_boxes, test_images_dict)


# ************** Internal helper functions *****************

def construct_images_np(images_dict: dict, file_name_dict: dict, train_image_dir: str,
                        verbose: bool, win_set: Tuple[int, int, int, int]) -> Tuple[List, dict]:
    """ Takes each image in the given dictionary of images, breaks them up into windows, maps the window
    information to the dictionary for the image (images_dict), and adds each window to a list of numpy
    images (images_np)
    """
    images_np = []  # the sliding windows of the images
    # construct images_np - should be a list of all the windows of all the images
    for index, image_np in enumerate(get_images(train_image_dir, file_name_dict)):
        # divide image into windows of size win_height * win_width
        # and add to dictionary corresponding to that image
        windows = {}
        for win_index, (window, xmin, ymin, xmax, ymax) in enumerate(get_windows(image_np, *win_set, False)):
            if verbose: print('NEW WINDOW with dimensions ' + str(window.shape))
            window_dict = {}
            # window_dict['window'] = window
            window_dict['xmin'] = xmin
            window_dict['ymin'] = ymin
            window_dict['xmax'] = xmax
            window_dict['ymax'] = ymax
            window_dict['dimensions'] = (xmax - xmin, ymax - ymin)
            window_dict['boxes'] = []
            windows[len(images_np)] = window_dict  # 0,1,2,3...# of windows - 1
            images_np.append(window)

        images_dict[index + 1]['num windows'] = len(windows)
        images_dict[index + 1]['windows'] = windows

    return (images_np, images_dict)


def construct_category_index(train_annotations: dict, desired_categories: dict) -> dict:
    """ Takes the category index from the training annotations and constructs it in the
    correct format to be used in the retraining process
    """
    # construct category index in correct format
    category_index = {}
    for category in train_annotations['categories']:
        if category['name'] in desired_categories:
            category_index[category['id']] = {
                'id': category['id'],
                'name': category['name']
            }
    return category_index


def construct_gt_boxes(images_dict: dict, num_windows: int, verbose: bool) -> Tuple[List, dict]:
    """ Constructs a list of the ground truth boxes to be used in the retraining step. This list
    corresponds to the training image set so that for each index in train_images_np, there is a
    corresponding index in gt_boxes that contains a list of all of the gt_boxes in that image
    (in our case, each of the windows).
    We also keep track of all of the windows in which there are no annotations associated with
    them so that we can remove these from the retraining set later.
    """
    # construct gt_boxes array of numpy lists of boxes for each window in each image
    gt_boxes = [None] * num_windows
    no_annotation_ids = {}
    for image_id in images_dict:
        # print('analyzing image ' + str(image_id))
        for win_id, window in images_dict[image_id]['windows'].items():
            # print(window['boxes'])
            if win_id < num_windows:
                gt_boxes[win_id] = np.array(window['boxes'], dtype=np.float32)
            # print(window['boxes'])
            if not window['boxes']:
                no_annotation_ids[win_id] = ''

    num_annotated_windows = num_windows - len(no_annotation_ids)
    percent_annotated_windows = round((num_annotated_windows / num_windows) * 100)
    if verbose:
        print(str(percent_annotated_windows) + '% of windows have annotations associated with them.')
        print('This is ' + str(num_annotated_windows) + ' windows out of ' + str(num_windows))

    return (gt_boxes, no_annotation_ids)


def map_annotations_to_windows(images_dict: dict, train_annotations: dict, category_index: dict,
                               verbose: bool) -> dict:
    """ Maps the image annotations to their corresponding windows. Note that the same annotation
    can be mapped to multiple windows as the windows may overlap.
    We map an annotation to a window if at least 70% of the annotation is preserved
    """
    num_successes = 0
    num_failures = 0

    # Get annotated boxes for each image window
    annotations_dict = train_annotations['annotations']
    for annotation in annotations_dict:
        if annotation['category_id'] in category_index:
            # **** DOTA annotation info ****
            # Two forms of annotation exist in this annotations dictionary
            #   (1) Arbitrary BB: {(xi,yi) for i = 1,2,3,4}
            #   (2) HBB: [xmin, ymin, width, height]
            # We use (2) here because it more closely fits the original BB format
            hbb = annotation['bbox']
            # calculate HBB in new format
            xmin = hbb[0]
            ymin = hbb[1]
            xmax = xmin + hbb[2]
            ymax = ymin + hbb[3]
            box = [xmin, ymin, xmax, ymax]
            if verbose: print('annotation: ' + str(box))
            # get corresponding window ids associated with the annotated box
            image_id = annotation['image_id']
            window_ids = get_window_ids(box, images_dict[image_id])
            if not window_ids:
                num_failures += 1
                if verbose: print(
                    'Couldn\'t find window for annotation ' + str(box) + ' in image ' + images_dict[image_id]['name'])
            else:
                if verbose: print('annotation ' + str(annotation['category_id']) + ' corresponds to ' + str(
                    window_ids) + ' windows in image: ' + images_dict[image_id]['name'])
                num_successes += 1
                for window_id in window_ids:
                    if verbose: print('mapping box to window ' + str(window_id))
                    new_box = map_box_to_window(box, images_dict[image_id]['dimensions'],
                                                images_dict[image_id]['windows'][window_id])
                    if verbose: print('new box is ' + str(new_box))
                    images_dict[image_id]['windows'][window_id]['boxes'].append(new_box)

    if verbose:
        print('Num successes: ' + str(num_successes))
        print('Num failures: ' + str(num_failures))
        print('Percent of annotations preserved: ' + str(
            round((num_successes / (num_successes + num_failures)) * 100, 2)) + '%')

    return images_dict