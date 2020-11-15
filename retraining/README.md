# DOTA-Retraining

This project is based off [this notebook](https://colab.research.google.com/drive/13esF5tJVdQ7x1t2HyWW3mlrR_69kAtcL?authuser=1#scrollTo=HWBqFVMcweF-) called "eager_few_shot_od_training_tf2_colab.ipynb." It takes an existing object detection model and retrains it on a new set of classes using a new training set. 

## Setup

This project uses **python 3.6**.

1. `pipenv install` or `python -m pipenv install`

2. Clone the tensorflow/models project

    `git clone --depth 1 https://github.com/tensorflow/models`

3. Download the checkpoint for the model you are retraining and put it into models/research/object_detection/test_data/
    * `wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz`
    * `tar -xf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz`
    * `mv ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint models/research/object_detection/test_data/`

4. Change variables pipeline_config and checkpoint_path to match that of the above model you have downloaded (located in retrain.py)

5. Place training and test data in the coco format in some directory within the project. Format:

    ```
    data/
        annotations/
            train.json
            validation.json
        train/
            images/
                image1.png
                image2.png
                ...
                image3.png
        validation/
            images/
                image1.png
                image2.png
                ...
                image3.png
    ```

6. Change the desired categories in main.py to perform object detection on

## Run Details

After [setup](#Setup), the project can be run normally within the pipenv. To run the project, perform the following steps:

1. `pipenv shell` or `python -m pipenv shell`
2. `python main.py`

The program can also be run with several different options:

* **-i** or **--ifiles**: This option defines where to find the input training and test set files. It takes one argument which is the directory to the input files.
    * Example usage: `python main -i ./tennis_courts`
* **-h** or **--help**: Prints out a help message. Takes no arguments.
* **-v** or **-verbose**: Prints out more statements related to the execution of the program than what would be outputted normally. Takes no arguments.
* **-d** or **-debug**: Similar to verbose but prints more debugging-related messages. Takes no arguments.

