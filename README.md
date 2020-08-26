# Model Training
Scripts for (re)training Tensorflow Zoo models on a selective subset of COCO classes

## Dependencies
Run `./install_deps.sh` to install necessary dependencies.
Set the environment variable `NOT_NVIDIA=true` if you
are installing on a non-Jetson device.

The script creates a Python virtual environment, which you
can activate by running `source env/bin/activate`.

## Generating input data
The Tensorflow API uses a special type of file for its input images: The [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) format.

A fully documented implementation of the conversion of COCO datasets into this proper TFRecord file format are found [here](tensorflow_object_detection_create_coco_tfrecord)

## New datasets
While the COCO dataset is vast, and contains many examples of each of the classes from the limited subset we care about (people, cars, trucks, etc.), the dataset doesn't contain many strange images from unusual angles, such as the overhead perspective of the objects captured by drones or surveillance footage. Here are some promising datasets that should help to mitigate this issue:
* [Vis-Drone](https://github.com/VisDrone/VisDrone-Dataset)
* [UAVDT](https://sites.google.com/site/daviddo0323/projects/uavdt)

If one of these (or other similar) datasets proves to be unsuitable for the task, the tool [labelImg](https://github.com/tzutalin/labelImg) is an effective little application to create bounding boxes for image datasets. It's time consuming, but an accurate and effective tool for the job.
