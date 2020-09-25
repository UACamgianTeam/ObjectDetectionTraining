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

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import colab_utils
from object_detection.builders import model_builder


def create_model_and_restore_weights(num_classes, pipeline_config_path, checkpoint_path):
    """
    Creates a model from a given pipeline configuration file and restores all but the classification layer at the head
    (which will be automatically randomly initialized)

    Args:
        num_classes: An integer representing the total number of classes that we wish to train on.
            This should be equivalent to the number of class labels in the label map file TODO: where is the label map?
        pipeline_config_path: The path to the pipeline configuration path defining the model that we want to
            restore from
        checkpoint_path: The path to the model checkpoints that we wish to restore from

    Returns:
        detection_model: A Detection model based on the passed-in pipeline configuration path

    """

    tf.keras.backend.clear_session()

    print('Building model and restoring weights for fine-tuning...', flush=True)

    # Load pipeline config and build a detection model.
    #
    # Since we are working off of a COCO architecture which predicts 90
    # class slots by default, we override the `num_classes` field here to be our
    # passed-in num_classes variable
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    model_config = configs['model']
    model_config.ssd.num_classes = num_classes
    model_config.ssd.freeze_batchnorm = True
    detection_model = model_builder.build(
        model_config=model_config, is_training=True)

    # Set up object-based checkpoint restore --- RetinaNet has two prediction
    # `heads` --- one for classification, the other for box regression.  We will
    # restore the box regression head but initialize the classification head
    # from scratch (we show the omission below by commenting out the line that
    # we would add if we wanted to restore both heads)
    fake_box_predictor = tf.compat.v2.train.Checkpoint(
        _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
        # _prediction_heads=detection_model._box_predictor._prediction_heads,
        #    (i.e., the classification head that we *will not* restore)
        _box_prediction_head=detection_model._box_predictor._box_prediction_head,
    )
    fake_model = tf.compat.v2.train.Checkpoint(
        _feature_extractor=detection_model._feature_extractor,
        _box_predictor=fake_box_predictor)
    ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
    ckpt.restore(checkpoint_path).expect_partial()

    # Run model through a dummy image so that variables are created
    image, shapes = detection_model.preprocess(tf.zeros([1, 640, 640, 3]))
    prediction_dict = detection_model.predict(image, shapes)
    _ = detection_model.postprocess(prediction_dict, shapes)
    print('Weights restored!')

    return detection_model


# Set up forward + backward pass for a single train step.
def get_model_train_step_function(model, optimizer, vars_to_fine_tune):
    """Get a tf.function for training step."""

    # Use tf.function for a bit of speed.
    # Comment out the tf.function decorator if you want the inside of the
    # function to run eagerly.
    @tf.function
    def train_step_fn(image_tensors,
                      groundtruth_boxes_list,
                      groundtruth_classes_list):
        """A single training iteration.

        Args:
          image_tensors: A list of [1, height, width, 3] Tensor of type tf.float32.
            Note that the height and width can vary across images, as they are
            reshaped within this function to be 640x640.
          groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
            tf.float32 representing groundtruth boxes for each image in the batch.
          groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
            with type tf.float32 representing groundtruth boxes for each image in
            the batch.

        Returns:
          A scalar tensor representing the total loss for the input batch.
        """
        shapes = tf.constant(batch_size * [[640, 640, 3]], dtype=tf.int32)
        model.provide_groundtruth(
            groundtruth_boxes_list=groundtruth_boxes_list,
            groundtruth_classes_list=groundtruth_classes_list)
        with tf.GradientTape() as tape:
            preprocessed_images = tf.concat(
                [detection_model.preprocess(image_tensor)[0]
                 for image_tensor in image_tensors], axis=0)
            prediction_dict = model.predict(preprocessed_images, shapes)
            losses_dict = model.loss(prediction_dict, shapes)
            total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
            gradients = tape.gradient(total_loss, vars_to_fine_tune)
            optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
        return total_loss

    return train_step_fn


def get_vars_to_fine_tune(trainable_variables, prefixes_to_train):
    """
    Selects the variables in a model to fine-tune

    Args:
        trainable_variables: A list of all the trainable variables from a Detection model
        prefixes_to_train: The prefixes of the variable names that we wish to
            train.

    Returns:
        vars_to_fine_tune: A pruned list of trainable variables that we want actually
            want to train
    """
    vars_to_fine_tune = []

    for var in trainable_variables:
        if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
            to_fine_tune.append(var)

    return vars_to_fine_tune


def fine_tune_model(data_stub, train_step_function):
    """
    Fine tunes an Object Detection model using the train_step_function

    Args:
        data_stub: TODO: Somehow pull data from XML files into this function
        train_step_function: Function for a single training step of the Detector model

    Returns:
        None
    """

    tf.keras.backend.set_learning_phase(True)

    print('Start fine-tuning!', flush=True)
    for idx in range(num_batches):
        # Grab keys for a random subset of examples
        all_keys = list(range(len(train_images_np)))
        random.shuffle(all_keys)
        example_keys = all_keys[:batch_size]

        # Note that we do not do data augmentation in this demo.  If you want a
        # a fun exercise, we recommend experimenting with random horizontal flipping
        # and random cropping :)
        gt_boxes_list = [gt_box_tensors[key] for key in example_keys]
        gt_classes_list = [gt_classes_one_hot_tensors[key] for key in example_keys]
        image_tensors = [train_image_tensors[key] for key in example_keys]

        # Training step (forward pass + backwards pass)
        total_loss = train_step_fn(image_tensors, gt_boxes_list, gt_classes_list)

        if idx % 10 == 0:
            print('batch ' + str(idx) + ' of ' + str(num_batches)
                  + ', loss=' + str(total_loss.numpy()), flush=True)

    print('Done fine-tuning!')


if __name__ == "__main__":

    num_classes = 1
    pipeline_config_path = '../models/original_models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/pipeline.config'
    checkpoint_path = '../models/retrained_models/my_ssd_mobilenet_v2_fpnlite/checkpoint/ckpt-0'

    # Create the object detection model from the pipeline configuration file
    object_detection_model = create_model_and_restore_weights(num_classes, pipeline_config_path, checkpoint_path)

    # Hyperparameters
    batch_size = 4
    learning_rate = 0.01
    num_batches = 100

    # Select variables in top layers to fine-tune
    to_fine_tune = get_vars_to_fine_tune(trainable_variables=object_detection_model.trainable_variables,
                                         prefixes_to_train=[
                                             'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
                                             'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead'])

    # Define the optimizer for training
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

    # Define the train step function
    train_step_fn = get_model_train_step_function(object_detection_model, optimizer, to_fine_tune)

    # TODO: Get data from the XML annotations
    data_stub = np.zeros(shape=(1, 640, 640, 3), dtype=np.int8)

    # Train the model!
    fine_tune_model(data_stub=data_stub, train_step_function=train_step_fn)
