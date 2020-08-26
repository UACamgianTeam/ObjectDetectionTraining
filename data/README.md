# Data

This folder will house all of the input data for the models, as well as the [label map](mscoco_label_map.pbtxt) for the classes.

See [tensorflow_object_detection_create_coco_tfrecord](https://github.com/elmines/JetsonBenchmarking/tree/master/model_retraining/CreateTFRecords/tensorflow_object_detection_create_coco_tfrecord) for generating input data

Data in this folder will be stored as TFRecords, so that they can be fed into the Tensorflow API.

# Storage

When running tests and training models, the TFRecords produced by following along with the [input data => TFRecord conversion](JetsonBenchmarking/model_retraining/CreateTFRecords/tensorflow_object_detection_create_coco_tfrecord/) script will be too large to host in an online repository.

To run these yourself, place the TFRecords in this repository for your own use. You can't push these to GitHub, they're way too big. However, you can place them in this repository on your machine and reference the path when training/testing your models.
