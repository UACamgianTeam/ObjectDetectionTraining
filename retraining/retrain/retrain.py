import tensorflow as tf
import numpy as np
import random
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

from typing import List

pipeline_config = 'models/research/object_detection/configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config'
checkpoint_path = 'models/research/object_detection/test_data/checkpoint/ckpt-0'

# retraining hyperparameters
batch_size = 4
learning_rate = 0.01
num_batches = 10  # 30


def retrain(train_images_np: List, gt_boxes: List, category_index: dict, verbose: bool) -> any:
    """ Fine-tunes the stored model (defined above in pipeline_config and checkpoint_path) on the
    new training set and new categories.

    The hyperparameters for the retraining step are listed above.
    """
    num_classes = len(category_index)
    # convert training set information to tensors
    (train_image_tensors, gt_box_tensors, gt_classes_one_hot_tensors) = convert_to_tensors(
        train_images_np, gt_boxes, num_classes)
    # build model and restore weights before fine-tuning
    detection_model = restore_weights(num_classes, verbose)
    # fine-tune model (retrain using our training set)
    detection_model = fine_tune(detection_model, gt_box_tensors, gt_classes_one_hot_tensors,
                                train_image_tensors, train_images_np, verbose)

    return detection_model


def convert_to_tensors(train_images_np: List, gt_boxes: List, num_classes: int):
    """ Converts class labels to one-hot; converts everything to tensors.
    The `label_id_offset` here shifts all classes by a certain number of indices;
    we do this here so that the model receives one-hot labels where non-background
    classes start counting at the zeroth index.  This is ordinarily just handled
    automatically in our training binaries, but we need to reproduce it here.
    """
    label_id_offset = 1
    train_image_tensors = []
    gt_classes_one_hot_tensors = []
    gt_box_tensors = []
    for (train_image_np, gt_box_np) in zip(train_images_np, gt_boxes):
        train_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(
            train_image_np, dtype=tf.float32), axis=0))
        # print(gt_box_np)
        gt_box_tensors.append(tf.convert_to_tensor(gt_box_np, dtype=tf.float32))
        zero_indexed_groundtruth_classes = tf.convert_to_tensor(
            np.ones(shape=[gt_box_np.shape[0]], dtype=np.int32) - label_id_offset)
        gt_classes_one_hot_tensors.append(tf.one_hot(
            zero_indexed_groundtruth_classes, num_classes))

    return (train_image_tensors, gt_box_tensors, gt_classes_one_hot_tensors)


def restore_weights(num_classes: int, verbose: bool) -> any:
    """ Restores the weights of the stored model by running the model through a
    dummy image
    TODO - determine what the type is for the model...
    """
    # Load pipeline config and build a detection model.
    #
    # Since we are working off of a COCO architecture which predicts 90
    # class slots by default, we override the `num_classes` field here to be just
    # one (for our new rubber ducky class).
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
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
    image, shapes = detection_model.preprocess(tf.zeros([1, 1024, 1024, 3]))
    prediction_dict = detection_model.predict(image, shapes)
    _ = detection_model.postprocess(prediction_dict, shapes)
    if verbose: print('Weights restored!')

    return detection_model


def fine_tune(detection_model: any, gt_box_tensors: List, gt_classes_one_hot_tensors: List,
              train_image_tensors: List, train_images_np: List, verbose: bool) -> any:
    """ Fine-tunes (retrains) the stored model on the new training set. The hyperparameters
    for this process are defined at the top of this file.
    Uses gradient tape
    """
    tf.keras.backend.set_learning_phase(True)
    # Select variables in top layers to fine-tune.
    trainable_variables = detection_model.trainable_variables
    to_fine_tune = []
    prefixes_to_train = [
        'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
        'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead']
    for var in trainable_variables:
        if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
            to_fine_tune.append(var)

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

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    train_step_fn = get_model_train_step_function(
        detection_model, optimizer, to_fine_tune)

    if verbose: print('Start fine-tuning!', flush=True)
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
    # print(gt_boxes_list)
    total_loss = train_step_fn(image_tensors, gt_boxes_list, gt_classes_list)

    if idx % 10 == 0:
        print('batch ' + str(idx) + ' of ' + str(num_batches)
              + ', loss=' + str(total_loss.numpy()), flush=True)

    print('batch ' + str(num_batches) + ' of ' + str(num_batches)
          + ', loss=' + str(total_loss.numpy()), flush=True)

    print('Done fine-tuning!')
    input('Press [enter] to continue.')

    return detection_model