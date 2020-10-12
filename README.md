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
In order to retrain models, you must first clone the [tensorflow/models](https://github.com/tensorflow/models) repository. 
```
git clone https://github.com/tensorflow/models.git
```
**NOTE:** There is a known issue with some of the scripts in the tensorflow/models repository that may cause errors upon training or evaluating models when using a CUDA-enabled GPU. These errors stem from the fact that, by default, TensorFlow may allocate all of the GPU memory to the model to be trained, causing some serious issues. Therefore, add the following lines to files in [tensorflow/models/object_detection](https://github.com/tensorflow/models/object_detection) or [tensorflow/models/object_detection/legacy](https://github.com/tensorflow/models/object_detection/legacy) if you see some errors along the lines of ```CUDNN_STATUS_INTERNAL_ERROR```:
```
# Fix a bug in TensorFlow by setting allow_growth to True to dynamically grow the memory used
session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=session_config)
```

Then, run ```./install_deps.sh``` to install necessary dependencies in a virtual environment. Afterwards, activate this virtual environment by running ```source env/bin/activate```.

Finally, run ```bash models/retrain_model.sh``` to retrain models, assuming that you've downloaded those models from the Tensorflow Model Zoo and put them in the folder '.../ObjectDetection/models/original_models'
