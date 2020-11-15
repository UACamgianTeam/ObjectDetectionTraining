  
import unittest
import numpy as np

from utils.load_data import get_annotations
from preprocess.construct_dicts import construct_dicts, construct_desired_ids, \
    construct_id_mapping
from preprocess.preprocess import construct_category_index, preprocess_test_images, get_unsliced_images_np, \
    calculate_label_id_offsets, map_category_ids_to_index, map_indices_to_category_ids
from utils.plot import visualize_image_set

from evaluate.postprocess import restore_image_detections, run_nms
from evaluate.eval import write_window_validation_file, evaluate

# The minimum threshold of scores to keep
min_threshold = 0.5

desired_categories = { # simple test for one category
  'tennis-court',
  'soccer-ball-field',
  'baseball-diamond'
}

verbose = False
debug = False

class TestImageProcessing(unittest.TestCase):
    def test_image_processing(self):
        """ Tests the image processing steps (pre and post) of the retraining process. It does
        this on 2 images and their corresponding annotations over one of their categories.

        (1) Load test annotations and image information
        (2) Break images into windows
        (3) Fake running predictions by setting predicted boxes and scores to the ground truths
            with 100% scores
        (4) 

        """
        test_annotations_dir = 'tests/annotations/validation.json'
        test_image_dir = 'tests/images/'
        test_annotations = get_annotations(test_annotations_dir)
        (test_images_dict, test_file_name_dict)   = construct_dicts(test_annotations)
        category_index = construct_category_index(test_annotations, desired_categories)
        desired_ids = construct_desired_ids(desired_categories, test_annotations['categories'])
        id_to_category = construct_id_mapping(desired_categories, test_annotations['categories'])
         # set windowing information (size of window and stride); these values taken from DOTA paper
        win_height = 1024
        win_width  = 1024
        win_stride_vert  = 512
        win_stride_horiz = 512
        win_set = (win_height, win_width, win_stride_vert, win_stride_horiz) # windowing information
        (test_images_np, test_gt_boxes, test_gt_classes, test_images_dict) = preprocess_test_images(
            test_images_dict, test_file_name_dict, test_image_dir, test_annotations,
            category_index, win_set, verbose)
        if verbose: print(test_images_dict)
        if debug:
            visualize_image_set(test_images_np, test_gt_boxes, test_gt_classes, category_index, 'Test set', min_threshold)
            print(test_images_dict)
        # set the category id mapping information for the retraining process
        label_id_offsets = calculate_label_id_offsets(category_index)
        # set ground truth as predicted in test set (simulates run_inference in detect.py)
        predicted_window_boxes   = [None] * len(test_images_np)
        predicted_window_classes = [None] * len(test_images_np)
        predicted_window_scores  = [None] * len(test_images_np)
        for image_id, image_info in test_images_dict.items():
            for window_id, window_info in image_info['windows'].items():
                predicted_boxes   = window_info['boxes']
                predicted_classes = window_info['classes']
                # indices = map_category_ids_to_index(label_id_offsets, predicted_classes)
                # print(indices)
                # print(map_indices_to_category_ids(label_id_offsets, indices))
                predicted_scores  = [1.0] * len(predicted_boxes) # 100% accuracy
                predicted_window_boxes[window_id]    = predicted_boxes
                predicted_window_classes[window_id]  = predicted_classes
                predicted_window_scores[window_id]   = predicted_scores
                test_images_dict[image_id]['windows'][window_id]['predicted_boxes']  = predicted_boxes
                test_images_dict[image_id]['windows'][window_id]['predicted_scores'] = predicted_scores
                test_images_dict[image_id]['windows'][window_id]['predicted_classes'] = \
                    np.array(predicted_classes, dtype=np.uint32)
        # test that the mapping from category_ids -> 0-index array -> category_ids would work
        # print(label_id_offsets)
        # print(predicted_window_classes)
        # indices = map_category_ids_to_index(label_id_offsets, predicted_window_classes)
        # print(indices)
        # category_ids = map_indices_to_category_ids(label_id_offsets, indices)
        # print(category_ids)
        # calculate new box information for testing purposes
        # boxes in format [ymin, xmin, ymax, xmax] -> [0.4208984375, 0.6514346439957492, 0.5888671875, 0.7385759829968119]
        original_box = test_images_dict[1]['windows'][0]['predicted_boxes'][0]
        # dimensions in format (width, height) -> (941, 1062)
        img_dimensions = test_images_dict[1]['dimensions']
        # dimensions in format (width, height) -> (941, 1024)
        window_dimensions = test_images_dict[1]['windows'][0]['dimensions']
        # dimension_box in format (ymin, xmin, ymax, xmax) -> (0,0,1024,941)
        window_dimension_box = test_images_dict[1]['windows'][0]['dimension_box']
        if debug:
            visualize_image_set(test_images_np, predicted_window_boxes, predicted_window_classes, 
                category_index, 'Test set', min_threshold, predicted_window_scores)
            print(test_images_dict)
            print('first bounding box: ' + str(original_box))
            print('image dimensions: ' + str(img_dimensions))
            print('window dimensions: ' + str(window_dimensions))
            print('window dimension box: ' + str(window_dimension_box))
        # resulting box should be [ymin * win_height + win_ymin, xmin * win_width + win_xmin, ymax * win_height + win_ymin, xmax * win_width + win_xmin]
        (ymin_percentage, xmin_percentage, ymax_percentage, xmax_percentage) = tuple(original_box)
        (win_ymin, win_xmin, win_ymax, win_xmax) = window_dimension_box
        (win_width, win_height) = window_dimensions
        (img_width, img_height) = img_dimensions
        ymin = ymin_percentage * win_height + win_ymin
        xmin = xmin_percentage * win_width + win_xmin
        ymax = ymax_percentage * win_height + win_ymin
        xmax = xmax_percentage * win_width + win_xmin
        absolute_box = [ymin, xmin, ymax, xmax]
        relative_box = [ymin / img_height, xmin / img_width, ymax / img_height, xmax / img_width]
        if debug:
            print('Resulting absolute box should be: ' + str(absolute_box))
            print('Resulting relative box should be: ' + str(relative_box))
        # restore annotations back to original image
        (test_images_dict, predicted_image_boxes, predicted_image_classes, predicted_image_scores)  = \
            restore_image_detections(test_images_dict, min_threshold)
        if debug:
            print('Calculated box is: ' + str(predicted_image_boxes[0][0]))
        assert(predicted_image_boxes[0][0] == relative_box)
        # visualize restored detections
        unsliced_test_images_np = get_unsliced_images_np(test_image_dir, test_file_name_dict)
        if debug:
            visualize_image_set(unsliced_test_images_np, predicted_image_boxes, predicted_image_classes,
                category_index, 'Predicted for Images', min_threshold, predicted_image_scores)
        # test nms is working
        test_images_dict[1]['predicted_boxes'].append([0.421, 0.652, 0.59, 0.73]) # add box slightly off of an existing box
        test_images_dict[1]['predicted_scores'].append(0.5) # give it a worse score
        length_before_nms = len(test_images_dict[1]['predicted_boxes'])
        if debug:
            print(len(test_images_dict[1]['predicted_boxes']))
        test_images_dict = run_nms(test_images_dict)
        if debug:
            print('*****************')
            print(len(test_images_dict[1]['predicted_boxes']))
        assert(len(test_images_dict[1]['predicted_boxes']) < length_before_nms)
        # test writing window validation json to file
        data_path = 'tests'
        write_window_validation_file(data_path, test_annotations, test_images_dict)
        # test evaluating results per window and per image (results should be 1.0 for both)
        evaluate(data_path, test_images_dict, desired_ids, id_to_category, min_threshold)

if __name__ == '__main__':
    unittest.main()