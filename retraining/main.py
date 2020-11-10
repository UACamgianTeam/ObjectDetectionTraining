import os
import sys, getopt
import logging
import json

from preprocess.construct_dicts import construct_dicts, construct_desired_ids, \
  construct_id_mapping
from preprocess.preprocess import preprocess_train_images, preprocess_test_images, \
  construct_category_index, get_unsliced_images_np, calculate_label_id_offsets
from retrain.retrain import retrain
from utils.load_data import get_annotations
from utils.plot import visualize_image_set
from evaluate.detect import run_inference
from evaluate.postprocess import restore_image_detections, run_nms
from evaluate.eval import write_window_validation_file, evaluate

verbose = False
debug = False

# The minimum threshold of scores to keep
min_threshold = 0.3

desired_categories = {
  'tennis-court',
  'soccer-ball-field',
  'ground-track-field',
  'baseball-diamond',
#  'plane',
#  'bridge',
#  'small-vehicle',
#  'large-vehicle',
#  'ship',
#  'basketball-court',
#  'storage-tank',
#  'roundabout',
#  'harbor',
#  'swimming-pool',
#  'helicopter'
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
      * Runs non-maximum suppression on both the windows and the original image
      * Evaluates the results per window and per image separately

  TODO - refactor the image and window dictionaries into classes (Window.py and Image.py)
       - can better postpone loading the images/windows into memory to preserve RAM
  """
  # set up error logging
  logging.basicConfig(filename = 'error.log', level = logging.WARNING)
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
        data_path = arg
    elif opt in ('-v', '--verbose'):
        global verbose
        verbose = True

  try:
    # set image path info
    train_image_dir = data_path + '/train/images/'
    test_image_dir  = data_path + '/validation/images/'
    train_annotations_dir = data_path + '/annotations/train.json'
    test_annotations_dir  = data_path + '/annotations/validation.json'
    # open annotation information
    train_annotations = get_annotations(train_annotations_dir)
    test_annotations  = get_annotations(test_annotations_dir)
    # construct dictionaries containing info about images and categories
    (train_images_dict, train_file_name_dict) = construct_dicts(train_annotations)
    (test_images_dict, test_file_name_dict)   = construct_dicts(test_annotations)
    desired_ids = construct_desired_ids(desired_categories, train_annotations['categories'])
    id_to_category = construct_id_mapping(desired_categories, train_annotations['categories'])
    # create category index in the correct format for retraining and detection
    category_index = construct_category_index(train_annotations, desired_categories)
    # set windowing information (size of window and stride); these values taken from DOTA paper
    win_height = 1024
    win_width  = 1024
    win_stride_vert  = 512
    win_stride_horiz = 512
    win_set = (win_height, win_width, win_stride_vert, win_stride_horiz) # windowing information
    # preprocess images (map images and their annotations -> windows)
    (train_images_np, gt_boxes, gt_classes) = preprocess_train_images(
        train_images_dict, train_file_name_dict, train_image_dir, 
        train_annotations, category_index, win_set, verbose)
    # visualize the training set if you would like (only first 16 images)
    if debug: 
      visualize_image_set(train_images_np, gt_boxes, gt_classes, category_index, 'Training set', min_threshold)
    # open and preprocess test set
    (test_images_np, test_gt_boxes, test_gt_classes, test_images_dict, 
            valid_images_np, valid_gt_boxes, valid_gt_classes) = preprocess_test_images(
        test_images_dict, test_file_name_dict, test_image_dir, test_annotations,
        category_index, win_set, verbose)
    if debug:
      visualize_image_set(test_images_np, test_gt_boxes, test_gt_classes, category_index,'Test set', min_threshold)
    # Set the label_id_offsets (we need to convert classes to 0-indexed arrays for the retraining step)
    label_id_offsets = calculate_label_id_offsets(category_index)
    # retrain an existing object detection model on the new training set

    detection_model = retrain(train_images_np, gt_boxes, gt_classes, 
      valid_images_np, valid_gt_boxes, valid_gt_classes, 
      category_index, label_id_offsets, verbose)
    # run retrained detection model on test set and store results per window in each image
    (test_images_dict, predicted_boxes, predicted_classes, predicted_scores) = run_inference(
        test_images_np, test_images_dict, label_id_offsets, detection_model)

    visualize_image_set(test_images_np, predicted_boxes, predicted_classes, category_index, 
      'Predicted', min_threshold, predicted_scores)
    # restore window detections back to their original images
    (test_images_dict, predicted_image_boxes, predicted_image_classes, predicted_image_scores)  = \
      restore_image_detections(test_images_dict, min_threshold)
    # visualize restored detections
    unsliced_test_images_np = get_unsliced_images_np(test_image_dir, test_file_name_dict)
    visualize_image_set(unsliced_test_images_np, predicted_image_boxes, predicted_image_classes, 
      category_index, 'Predicted for Images', min_threshold, predicted_image_scores)
    
    # Run non-maximum suppression per-window and per-image
    # test_images_dict = run_nms(test_images_dict)

    # Write validation annotations for each window
    write_window_validation_file(data_path, test_annotations, test_images_dict)

    # evaluate results per window and per image
    evaluate(data_path, test_images_dict, desired_ids, id_to_category, min_threshold)

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
