import json
from nms import nms
from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
from .cocoeval import COCOeval

from typing import Tuple, List

# evaluate using pycocotools
annType = 'bbox'


def evaluate_per_image(data_path: str, test_images_dict: dict) -> None:
    evaluation_annotation_path = data_path + '/annotations/evaluation/'

    labels_path = evaluation_annotation_path + 'validation_image_subset.json'  # ground-truth
    results_path = evaluation_annotation_path + 'image_results.json'

    write_image_results(results_path, test_images_dict)

    cocoGt = COCO(labels_path)
    cocoDt = cocoGt.loadRes(results_path)

    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.catIds = [8]  # set category ids we want to evaluate on
    # cocoEval.params.useCats = [1]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def evaluate_per_window(data_path: str, test_windows_dict):
    evaluation_annotation_path = data_path + '/annotations/evaluation/'

    labels_path = evaluation_annotation_path + 'validation_window_subset.json'
    results_path = evaluation_annotation_path + 'window_results.json'

    write_window_results(results_path, test_windows_dict)

    cocoGt = COCO(labels_path)
    cocoDt = cocoGt.loadRes(results_path)

    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = cocoGt.getImgIds()
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def write_image_results(results_path: str, test_images_dict: dict) -> None:
    # write results to a coco-format results file
    results = []
    for image_id, test_image_dict in test_images_dict.items():
        for index, bbox in enumerate(test_image_dict['predicted_boxes']):
            # for index, bbox in enumerate(test_image_dict['boxes']): # ground truth (should get 1.0 scores)
            # (xmin, ymin, xmax, ymax) = bbox
            # coco_bbox = (xmin, ymin, xmax - xmin, ymax - ymin) # coco bbox is (xmin, ymin, width, height)
            (xmin, ymin, width, height) = bbox
            new_bbox = (ymin, xmin, height, width)
            detection = {
                'image_id': image_id,
                'category_id': 8,  # TODO - store category id in test_images_dict for each annotation
                'bbox': new_bbox,  # coco_bbox,
                'score': test_image_dict['predicted_scores'][index].item()  # 100.0
            }
            results.append(detection)

    with open(results_path, 'w') as outfile:
        json.dump(results, outfile)


def write_window_results(data_path: str, test_windows_dict: dict) -> None:
    # # write results to a coco-format results file
    results = []
    for image_id, test_window_dict in test_windows_dict.items():
        for index, annotation in enumerate(test_window_dict['annotations']):
            detection = {
                'image_id': image_id,
                'category_id': 8,
                'bbox': annotation,
                'score': test_window_dict['scores'][index].item()
            }
            results.append(detection)

    results_path = data_path + '/annotations/window_results.json'
    labels_path = data_path + '/annotations/validation.json'

    with open(results_path, 'w') as outfile:
        json.dump(results, outfile)


def restore_image_detections(test_images_dict: dict) -> (dict, List, List):
    """ Combine window results to restore detection results on original image

    Also return lists of boxes and detections for visualization purposes
    """
    predicted_boxes = []
    predicted_scores = []
    for image_id, test_image_dict in test_images_dict.items():
        # predicted_boxes[image_id] = []
        # predicted_scores[image_id] = []
        for window_id, window_dict in test_image_dict['windows'].items():
            print(window_id)
            print(window_dict)
            window_dimensions = window_dict['dimension_box']
            annotated_boxes = window_dict['predicted_boxes']
            scores = window_dict['predicted_scores']
            for index in range(len(annotated_boxes)):
                score = scores[index]
                if score > 0.5:
                    image_box = map_box_to_image(tuple(annotated_boxes[index]), window_dimensions)
                    test_images_dict[image_id]['predicted_boxes'].append(image_box)
                    test_images_dict[image_id]['predicted_scores'].append(score)
                    # predicted_boxes[image_id].append(image_box)
                    # predicted_scores[image_id].append(score)

    return (test_images_dict, predicted_boxes, predicted_scores)


def map_box_to_image(box: Tuple, win_dimensions: Tuple) -> List:
    """ Maps a box in a window to its original image
    Goes from [xmin%, ymin%, xmax%, ymax%] in window to [xmin,ymin,width,height] in image
    """
    (xmin_percentage, ymin_percentage, xmax_percentage, ymax_percentage) = box
    (win_xmin, win_ymin, win_xmax, win_ymax) = win_dimensions
    # convert percentage to absoluate position
    x_length = win_xmax - win_xmin
    y_length = win_ymax - win_ymin
    xmin = xmin_percentage * x_length
    xmax = xmax_percentage * x_length
    ymin = ymin_percentage * y_length
    ymax = ymax_percentage * y_length
    # convert from window coords -> image coords
    xmin += win_xmin
    xmax += win_xmin
    ymin += win_ymin
    ymax += win_ymin
    # [ top-left, top-right, bottom-right, bottom-left]
    # return [[xmin,ymax], [xmax,ymax], [xmax,ymin], [xmin,ymin]]
    # [x,y,width,height] -> coco format
    return [xmin, ymin, xmax - xmin, ymax - ymin]


def write_validation_subsets(data_path: str, desired_categories: dict, test_annotations: dict):
    """ Writes a subset of the validation set to file using the given dictionary
    of desired categories. It does this...
    (1) per image (validation_image_subset.json)
    (2) per window (validation_window_subset.json)
    """
    image_file_path = data_path + '/annotations/evaluation/validation_image_subset.json'
    window_file_path = data_path + '/annotations/evaluation/validation_window_subset.json'

    # per image...
    new_image_validation = {}
    new_image_validation['info'] = test_annotations['info']
    new_image_validation['images'] = test_annotations['images']
    new_image_validation['categories'] = []
    for category in test_annotations['categories']:
        if category['name'] in desired_categories:
            new_image_validation['categories'].append(category)

    new_image_validation['annotations'] = []
    for test_annotation in test_annotations['annotations']:
        if test_annotation['category_id'] in desired_categories:
            new_image_validation['annotations'].append(test_annotation)

    with open(image_file_path, 'w') as outfile:
        json.dump(new_image_validation, outfile)


def run_nms(test_images_dict: dict) -> dict:
    """ Runs non-maximum suppression on all of the predicted boxes/scores for each of the images and
    all of the images' windows
    Stores the results in test_images dict
    This should prune any boxes that overlap considerably and it should favor the higher-scoring boxes
    """
    # run non-maximum suppression per image
    for image_id, image_info in test_images_dict.items():
        image_indices = non_max_suppression(image_info['predicted_boxes'], image_info['predicted_scores'])
        test_images_dict[image_id]['predicted_boxes'] = \
            [box for index, box in enumerate(image_info['predicted_boxes']) if index in image_indices]
        test_images_dict[image_id]['predicted_scores'] = \
            [box for index, box in enumerate(image_info['predicted_scores']) if index in image_indices]
        # run non-maxmium suppression per-window
        # for window_id, window_info in image_info['windows'].items():
        #     window_indices = non_max_suppression(window_info['predicted_boxes'], window_info['predicted_boxes'])
        #     test_images_dict[image_id]['windows'][window_id]['predicted_boxes'] = \
        #         [box for index, box in enumerate(window_info['predicted_boxes'])  if index in window_indices]
        #     test_images_dict[image_id]['windows'][window_id]['predicted_scores'] = \
        #         [box for index, box in enumerate(window_info['predicted_scores'])  if index in window_indices]

    return test_images_dict


def non_max_suppression(boxes: List, scores: List) -> List:
    """ Takes a list of boxes in the format (xmin,ymin,xmax,ymax) and a list of scores
    associated with those boxes and...
        (1) Changes format of boxes -> (xmin,ymin,width,height)
        (2) Runs non-maximum suppression on the boxes/scores
        (3) Returns a list of indices that should be kept
    """
    # boxes stored as (xmin,ymin,xmax,ymax) so need to map...
    for index, box in enumerate(boxes):
        (xmin, ymin, xmax, ymax) = box
        boxes[index] = (xmin, ymin, xmax - xmin, ymax - ymin)
    # takes boxes in form (x,y,w,h)
    return nms.boxes(boxes, scores)
