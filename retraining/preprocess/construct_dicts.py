import json

from typing import Tuple


def construct_dicts(annotations: dict) -> Tuple[dict, dict]:
    """ Constructs 2 dictionaries using the information in the annotations
    for the image set:
      1. images_dict: contains info for each image's...
        * file name
        * dimensions (width, height)
        * window information (will be generated later in the preprocessing steps)
      2. file_name_dict: maps each image's file name to its corresponding id in
      the images_dict
    """
    images_dict = construct_images_dict(annotations)
    file_name_dict = map_file_name_to_id(annotations)
    return (images_dict, file_name_dict)


def construct_images_dict(annotations: dict) -> dict:
    """ Constructs a dictionary for the images that maps image ids to
    corresponding information about that image

    Args:
      A dictionary of annotations in the coco format
    Returns:
      A dictionary with...
        (1) image dimensions information: (width, height)
        (2) list of ground-truth HBB... empty initially: []
    """
    images_dict = {}
    image_info_list = annotations['images']
    for image in image_info_list:
        id = image['id']
        file_name = image['file_name']
        width = image['width']
        height = image['height']
        images_dict[id] = {}
        images_dict[id]['name'] = file_name
        images_dict[id]['dimensions'] = (width, height)
        images_dict[id]['windows'] = {}  # will store sliding window info  for image here
    return images_dict


def map_file_name_to_id(annotations: dict) -> dict:
    """ Constructs a dictionary for the images that maps their file names
    to their corresponding ids

    Args:
      A dictionary of annotations in the coco format
    Returns:
      A dictionary that maps file_name -> image id
    """
    file_name_dict = {}
    image_info_list = annotations['images']
    for image in image_info_list:
        id = image['id']
        file_name = image['file_name']
        # file_name_dict[file_name] = id   # map file name -> id
        file_name_dict[id] = file_name  # map id -> file name
    return file_name_dict