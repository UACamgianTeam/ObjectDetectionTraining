import os
import sys, getopt
import logging
import json

from preprocess.construct_dicts import construct_dicts
from preprocess.preprocess import preprocess_train_images, preprocess_test_images, \
    construct_category_index, get_unsliced_images_np
from retrain.retrain import retrain
from utils.load_data import get_annotations
from utils.plot import visualize_image_set, visualize_unsliced_predictions
from test.detect import run_inference
from test.eval import restore_image_detections, evaluate_per_image, evaluate_per_window, \
    write_validation_subsets, run_nms

verbose = False
debug = False

desired_categories = {
    'person',
    8
}


def main(argv):
    """ This is the main entrypoint of the retraining process.

    Proper usage: python main [-i ./path/to/data] [-v] [-d] [-h]
      * -i: The path to the training and validation data (see README for proper format)
      * -v: "verbose"; prints out more things...
      * -d: "debug"; prints out things related to debugging the program

    It performs the following steps:
      1. Preprocesses the training set
        * Reformats the training annotations
        * Constructs dictionaries containing info about the images in the training set
        * Maps images and their annotations to corresponding windows (sliding window approach)
          - annotations are kept if at least 70% of the ground truth box is preserved in the window
        * NOTE - could try image augmentation here
      2. Retrains an existing object detection model (ssd_resnet) on the new class categories
      3. Preprocesses the test set (windows, maps annotations to windows, etc.)
      4. Runs inference on the test images using the retrained model
      5. Evaluates the results of the inference step
        * Combines the window results into original image results
        * TODO - Runs non-maximum suppression on both the windows and the original image
        * TODO - Evaluates the results per window and per image separately
    TODO - refactor the image and window dictionaries into classes (Window.py and Image.py)
         - can better postpone loading the images/windows into memory to preserve RAM
    """
    # set up error logging
    logging.basicConfig(filename='error.log', level=logging.WARNING)
    log = logging.getLogger()
    # default path to data
    data_path = './data'
    # handle arguments and options
    try:
        opts, args = getopt.getopt(argv, 'hdvi:', ['ifiles='])
    except getopt.GetoptError:
        print('Incorrect usage.')
        print_help_message()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print_help_message()
            sys.exit()
        elif opt in ('-d', '--debug'):
            global debug
            debug = True
        elif opt in ('-i', '--ifiles'):
            if not debug:
                data_path = arg
        elif opt in ('-v', '--verbose'):
            global verbose
            verbose = True

    try:
        # set image path info
        train_image_dir = data_path + '/train/images/'
        test_image_dir = data_path + '/validation/images/'
        train_annotations_dir = data_path + '/annotations/train.json'
        test_annotations_dir = data_path + '/annotations/validation.json'
        # open annotation information
        train_annotations = get_annotations(train_annotations_dir)
        test_annotations = get_annotations(test_annotations_dir)
        # construct dictionaries containing info about images
        (train_images_dict, train_file_name_dict) = construct_dicts(train_annotations)
        (test_images_dict, test_file_name_dict) = construct_dicts(test_annotations)
        # create category index in the correct format for retraining and detection
        category_index = construct_category_index(train_annotations, desired_categories)
        # set windowing information (size of window and stride); these values taken from DOTA paper
        win_height = 1024
        win_width = 1024
        win_stride_vert = 512
        win_stride_horiz = 512
        win_set = (win_height, win_width, win_stride_vert, win_stride_horiz)  # windowing information
        # preprocess images (map images and their annotations -> windows)
        (train_images_np, gt_boxes) = preprocess_train_images(
            train_images_dict, train_file_name_dict, train_image_dir,
            train_annotations, category_index, win_set, verbose)
        # visualize the training set if you would like (only first 16 images)
        if debug: visualize_image_set(train_images_np, gt_boxes, category_index, 'Training set')
        # retrain an existing object detection model on the new training set
        detection_model = retrain(train_images_np, gt_boxes, category_index, verbose)
        # open and preprocess test set
        (test_images_np, test_gt_boxes, test_images_dict) = preprocess_test_images(
            test_images_dict, test_file_name_dict, test_image_dir, test_annotations,
            category_index, win_set, verbose)
        if debug: visualize_image_set(test_images_np, test_gt_boxes, category_index, 'Test set')
        # run retrained detection model on test set and store results per window in each image
        (test_images_dict, predicted_boxes, predicted_scores) = run_inference(
            test_images_np, test_images_dict, detection_model)

        visualize_image_set(test_images_np, predicted_boxes, category_index, 'Predicted', predicted_scores)
        # restore window detections back to their original images
        print(test_images_dict)
        (test_images_dict, predicted_image_boxes, predicted_image_scores) = restore_image_detections(test_images_dict)
        print(test_images_dict)
        # visualize restored detections
        unsliced_test_images_np = get_unsliced_images_np(test_image_dir, test_file_name_dict)
        visualize_unsliced_predictions(test_images_dict, category_index, unsliced_test_images_np)

        # TODO - run non-maximum suppression to combine similar detections

        test_images_dict = run_nms(test_images_dict)

        # Rewrite validation annotations with only the subset of desired categories for images and windows
        write_validation_subsets(data_path, desired_categories, test_annotations)

        # evaluate results per window and per image
        # evaluate_per_window(data_path, test_windows_dict)
        evaluate_per_image(data_path, test_images_dict)
        evaluate_per_window(data_path, test_images_dict)

        input('Press [enter] to end program.')

        return 0
    except Exception as err:
        log.exception('ERROR: ')
        sys.stderr.write('Error: ' + str(err))
        return 1


def print_help_message():
    """ Prints a message that details the correct usage of the project """
    print('Correct usage: python main.py -i <inputDataPath>\n')
    print('Options:')
    print('\t-v: verbose')
    print('\t-h: help')
    print('\t-d: debug')


if __name__ == '__main__':
    main(sys.argv[1:])