import numpy as np
from shapely.geometry import Polygon

from typing import List, Tuple, Generator


def get_windows(arr: np.ndarray, win_height: int, win_width: int,
                stride_vert: int, stride_horiz: int, should_pad: bool) -> Generator[
    Tuple[np.ndarray, int, int, int, int], None, None]:
    """ Divides an image into (not necessarily disjoint) windows
    :param np.ndarray arr: The image to divide
    :param int win_height: The height in pixels of each window
    :param int win_width: The width in pixels of each window
    :param int stride_vert: How many pixels to move downward after finishing a row of windows
    :param int stride_horiz: How many pixels to move to the right after generating a window
    :returns: The windows (one-by-one, so that you can loop over them in a for loop without them all being in memory)
    """
    (height, width, _) = arr.shape
    for y in range(0, height, stride_vert):
        ymin = y
        new_height = y + win_height
        pad_y = 0
        if y + win_height - 1 >= height:  # need to pad the y axis
            new_height = height
            pad_y = y + win_height - height
        for x in range(0, width, stride_horiz):
            xmin = x
            new_width = x + win_width
            pad_x = 0
            if x + win_width - 1 >= width:  # need to pad the x axis
                new_width = width
                pad_x = x + win_width - width
            win = arr[y: new_height, x: new_width]
            if should_pad:
                # pad the array in the top and right sides of the image to get desired shape
                win = np.pad(win, ((pad_y, 0), (0, pad_x), (0, 0)), mode='constant', constant_values=3)
            yield (win, xmin, ymin, new_width, new_height)


def get_window_ids(box: List, image_dict: dict) -> List:
    """ Maps an annotation box to its corresponding window_ids
    Previously associated with the image_id
    NOTE that there could be multiple window ids associated with one annotation
    box because of the stride on the window slider
    We keep a "sliced" annotation if at least 70% of the annotation is preserved in
    the window.
    """
    xmin, ymin, xmax, ymax = box
    annotation_box = [[xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]]  # used for iou calculation
    # window has info on xmin, ymin, width, and height
    window_ids = []
    for window_id, window in image_dict['windows'].items():
        win_xmin = window['xmin']
        win_xmax = window['xmax']
        win_ymin = window['ymin']
        win_ymax = window['ymax']
        window_box = [[win_xmin, win_ymax], [win_xmax, win_ymax], [win_xmax, win_ymin], [win_xmin, win_ymin]]
        # Annotation belongs to window_id if at least 70% of it is in the window
        if percent_overlap(annotation_box, window_box) >= 0.7:
            window_ids.append(window_id)

    return window_ids


def map_box_to_window(box: List, image_dimensions, window_dict: dict) -> List:
    """ Maps an annotation to its corresponding window. The annotations are originally
    per-image instead of per-window so the coordinates need to be converted.
    We keep a "sliced" annotation if at least 70% of the annotation is preserved in
    the window.
    """
    (img_width, img_height) = image_dimensions
    (win_width, win_height) = window_dict['dimensions']
    win_xmin = window_dict['xmin']
    win_xmax = window_dict['xmax']
    win_ymin = window_dict['ymin']
    win_ymax = window_dict['ymax']
    win_box = [win_xmin, win_ymin, win_xmax, win_ymax]

    # convert annotation coordinates (relative to image) to window coordinates
    # and normalize
    xmin = (box[0] - win_xmin) / win_width
    ymin = (box[1] - win_ymin) / win_height
    xmax = (box[2] - win_xmin) / win_width
    ymax = (box[3] - win_ymin) / win_height
    # if any of the coordinates are outside of the window bounds, clamp to window
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if xmax > win_width:
        xmax = win_width
    if ymax > win_height:
        ymax = win_height

    return [ymin, xmin, ymax, xmax]


def percent_overlap(annotation: List, image: List) -> float:
    """ Determines the percentage an annotation box overlaps with an image box
    each in the shape: [[xmin,ymax], [xmax,ymax], [xmax,ymin], [xmin,ymin]]
    """
    ann_poly = Polygon(annotation)
    image_poly = Polygon(image)
    return ann_poly.intersection(image_poly).area / ann_poly.area