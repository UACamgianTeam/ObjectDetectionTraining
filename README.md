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

A fully documented implementation of the conversion of COCO datasets into this proper TFRecord file format are found [here](CreateTFRecords).

## New datasets
While the COCO dataset is vast, and contains many examples of each of the classes from the limited subset we care about (people, cars, trucks, etc.), the dataset doesn't contain many strange images from unusual angles, such as the overhead perspective of the objects captured by drones or surveillance footage. Here are some promising datasets that should help to mitigate this issue:
* [Vis-Drone](https://github.com/VisDrone/VisDrone-Dataset)
* [UAVDT](https://sites.google.com/site/daviddo0323/projects/uavdt)

If one of these (or other similar) datasets proves to be unsuitable for the task, the tool [labelImg](https://github.com/tzutalin/labelImg) is an effective little application to create bounding boxes for image datasets. It's time consuming, but an accurate and effective tool for the job.

## Retraining Models

First, run ```./install_deps.sh``` to install necessary dependencies in a virtual environment. Afterwards, activate this virtual environment by running ```source env/bin/activate```.

Then, download Tensorflow 2 Detection Model Zoo models from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

Then, untar the models and put them in [models/original_models]()
```
# Uncompress the model folder
tar -xzfv <tf2_zoo_model_name>.tar.gz

# Move the model folder to models/original_models
mv <tf2_zoo_model_name> <your_path_to>/ObjectDetectionTraining/models/original_models
```

Then, change the ```pipeline.config``` file within your new TF2 Zoo model to configure settings such as # of classes, path to train/val data, and any other
hyperparameters.

Finally, run ```bash retrain_model.sh``` to retrain models

## [UAVDT](https://sites.google.com/site/daviddo0323/projects/uavdt)



1. Download [UAVDT-Benchmark-M](https://drive.google.com/file/d/1m8KA6oPIRK_Iwt9TYFquC87vBc_8wRVc/view)
