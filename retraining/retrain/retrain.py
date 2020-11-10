import tensorflow as tf
import numpy as np
import random
from collections import namedtuple
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import wandb
from preprocess.preprocess import map_category_ids_to_index
from typing import List, Tuple

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

pipeline_config = 'models/research/object_detection/configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config'
checkpoint_path = 'models/research/object_detection/test_data/ssd/checkpoint/ckpt-0'

# dataset format
ObjDataset = namedtuple("ObjDataset", ["images", "boxes", "classes"])

# retraining hyperparameters
wandb.init(config={"batch_size": 8, "learning_rate": 0.01, "num_epochs": 100, "epoch_start": 50})


def retrain(train_images_np: List,
            train_gt_boxes: List,
            train_gt_classes: List,
            test_images_np: List,
            test_gt_boxes: List,
            test_gt_classes: List,
            category_index: dict,
            label_id_offsets: dict,
            verbose: bool) -> any:
    """ Fine-tunes the stored model (defined above in pipeline_config and checkpoint_path) on the
    new training set and new categories.
    
    The hyperparameters for the retraining step are listed above.
    """
    num_classes = len(category_index)
    # offset the classes so we feed a 0-indexed class array to the retraining step
    train_gt_classes = map_category_ids_to_index(label_id_offsets, train_gt_classes)
    # convert training set information to tensors
    (train_image_tensors, train_gt_box_tensors, train_gt_classes_one_hot_tensors) = convert_to_tensors(
        train_images_np, train_gt_boxes, train_gt_classes, label_id_offsets, num_classes)
    (test_image_tensors, test_gt_box_tensors, test_gt_classes_one_hot_tensors) = convert_to_tensors(
        test_images_np, test_gt_boxes, test_gt_classes, label_id_offsets, num_classes)
    # convert to tensorflow dataset format
    train_dataset = ObjDataset(train_image_tensors, train_gt_box_tensors, train_gt_classes_one_hot_tensors)
    test_dataset = ObjDataset(test_image_tensors, test_gt_box_tensors, test_gt_classes_one_hot_tensors)
    # build model and restore weights before fine-tuning
    detection_model = restore_weights(num_classes, verbose)
    # fine-tune model (retrain using our training set)
    detection_model = fine_tune(detection_model, train_dataset, test_dataset, train_images_np, verbose)

    return detection_model


def convert_to_tensors(train_images_np: List, gt_boxes: List, gt_classes: List,
                       label_id_offsets: dict, num_classes: int) -> Tuple:
    """ Converts class labels to one-hot; converts everything to tensors.

    The `label_id_offset` here shifts all classes by a certain number of indices;
    we do this here so that the model receives one-hot labels where non-background
    classes start counting at the zeroth index.  This is ordinarily just handled
    automatically in our training binaries, but we need to reproduce it here.
    """
    train_image_tensors = []
    gt_classes_one_hot_tensors = []
    gt_box_tensors = []
    for (train_image_np, gt_box_np, gt_class_np) in zip(train_images_np, gt_boxes, gt_classes):
        train_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(
            train_image_np, dtype=tf.float32), axis=0))
        gt_box_tensors.append(tf.convert_to_tensor(gt_box_np, dtype=tf.float32))
        zero_indexed_groundtruth_classes = tf.convert_to_tensor(gt_class_np)
        gt_classes_one_hot_tensors.append(tf.one_hot(
            zero_indexed_groundtruth_classes, num_classes))

    return (train_image_tensors, gt_box_tensors, gt_classes_one_hot_tensors)


def restore_weights(num_classes: int, verbose: bool) -> any:
    """ Restores the weights of the stored model by running the model through a
    dummy image

    NOTE - apparently the model is type SSDMetaArch...
    """
    # Load pipeline config and build a detection model.
    #
    # Since we are working off of a COCO architecture which predicts 90
    # class slots by default, we override the `num_classes` field here to be the
    # number of classes we want to predict
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


def fine_tune(detection_model: any, train_dataset: tf.data.Dataset, valid_dataset: tf.data.Dataset,
              train_images_np: List, verbose: bool) -> any:
    """ Fine-tunes (retrains) the stored model on the new training set. The hyperparameters
    for this process are defined at the top of this file.

    Uses gradient tape
    """
    writer = tf.summary.create_file_writer("./logs")
    valid_loss_fn = get_valid_loss_fn(valid_dataset, detection_model, batch_size=wandb.config.batch_size)

    (image_tensors, gt_box_tensors, gt_classes_one_hot_tensors) = train_dataset
    image_tensors = tf.squeeze(tf.stack([detection_model.preprocess(t)[0] for t in image_tensors], axis=0))
    raw_dataset = tf.data.Dataset.from_generator(lambda: range(len(image_tensors)), output_types=tf.int32)

    train_loss = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)

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

    optimizer = tf.keras.optimizers.SGD(learning_rate=wandb.config.learning_rate, momentum=0.9)
    train_step_fn = get_model_train_step_function(
        detection_model, optimizer, to_fine_tune)

    if verbose: print('Start fine-tuning!', flush=True)
    for i_epoch in range(wandb.config.epoch_start, wandb.config.epoch_start + wandb.config.num_epochs):
        dataset = shuffle_and_batch(raw_dataset, batch_size=wandb.config.batch_size, random_seed=i_epoch)
        for (i_batch, indices) in enumerate(dataset):
            batch_images = tf.gather(image_tensors, indices)
            # Grab keys for a random subset of examples
            # all_keys = list(range(len(train_images_np)))
            # random.shuffle(all_keys)
            # example_keys = all_keys[:wandb.config.batch_size]

            # Note that we do not do data augmentation in this demo.  If you want a
            # a fun exercise, we recommend experimenting with random horizontal flipping
            # and random cropping :)
            gt_boxes_list = [gt_box_tensors[key] for key in indices]
            gt_classes_list = [gt_classes_one_hot_tensors[key] for key in indices]
            # image_tensors = [image_tensors[key] for key in indices]

            # Training step (forward pass + backwards pass)
            total_loss = train_step_fn(batch_images, gt_boxes_list, gt_classes_list)
            train_loss(total_loss)  # update train_loss

            if i_batch % 10 == 0:
                print('batch ' + str(i_batch) + ': loss=' + str(train_loss.result()), flush=True)

        # summarize results per epoch
        with writer.as_default():
            tf.summary.scalar('training loss', train_loss.result(), step=i_epoch - wandb.config.epoch_start + 1)
            tf.summary.scalar('validation loss', valid_loss_fn(), step=i_epoch - wandb.config.epoch_start + 1)
        train_loss.reset_states()

    print('epoch ' + str(wandb.config.num_epochs) + ': loss=' + str(train_loss.result()), flush=True)

    print('Done fine-tuning!')
    input('Press [enter] to continue.')

    return detection_model


# Set up forward + backward pass for a single train step.
def get_model_train_step_function(model, optimizer, vars_to_fine_tune):
    """Get a tf.function for training step."""

    # Use tf.function for a bit of speed.
    # Comment out the tf.function decorator if you want the inside of the
    # function to run eagerly.
    @tf.function
    def train_step_fn(batch_images,
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
        shapes = tf.constant(wandb.config.batch_size * [[640, 640, 3]], dtype=tf.int32)
        model.provide_groundtruth(
            groundtruth_boxes_list=groundtruth_boxes_list,
            groundtruth_classes_list=groundtruth_classes_list)
        with tf.GradientTape() as tape:
            # preprocessed_images = tf.concat(
            #     [detection_model.preprocess(image_tensor)[0]
            #     for image_tensor in image_tensors], axis=0)
            prediction_dict = model.predict(batch_images, shapes)
            losses_dict = model.loss(prediction_dict, shapes)
            # Log the loss metrics
            localization_loss = losses_dict['Loss/localization_loss']
            classification_loss = losses_dict['Loss/classification_loss']
            total_loss = localization_loss + classification_loss
            wandb.log({'localization_loss': localization_loss.numpy(),
                       'classification_loss': classification_loss.numpy(),
                       'loss': total_loss})
            gradients = tape.gradient(total_loss, vars_to_fine_tune)
            optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))

        return total_loss

    return train_step_fn


def get_valid_loss_fn(valid_dataset: ObjDataset, model, batch_size=8):
    (image_tensors, box_tensors, class_tensors) = valid_dataset
    image_tensors = tf.squeeze(tf.stack([model.preprocess(t)[0] for t in image_tensors], axis=0))
    raw_index_dataset = tf.data.Dataset.from_generator(lambda: range(len(image_tensors)), output_types=tf.int32)
    valid_loss = tf.keras.metrics.Mean("valid_loss", dtype=tf.float32)

    def f():
        index_dataset = raw_index_dataset.batch(wandb.config.batch_size)
        shapes = tf.constant(wandb.config.batch_size * [[640, 640, 3]], dtype=tf.int32)
        for indices in index_dataset:
            model.provide_groundtruth(
                [box_tensors[i] for i in indices],
                [class_tensors[i] for i in indices]
            )
            prediction_dict = model.predict(tf.gather(image_tensors, indices), None)
            losses_dict = model.loss(prediction_dict, shapes)
            total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
            valid_loss(total_loss)
        result = valid_loss.result()
        valid_loss.reset_states()
        return result

    return f


def shuffle_and_batch(dataset: tf.data.Dataset, batch_size=8, random_seed=0) -> tf.data.Dataset:
    dataset = dataset.shuffle(buffer_size=wandb.config.batch_size * 4, seed=random_seed)
    dataset = dataset.batch(wandb.config.batch_size)
    return dataset
