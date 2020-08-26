# Models

All of the models and their contents will be held in this repo as follows:

*./insert_model_name/*

Just remember, these files are too big for GitHub, so you should download them onto your machine, but you won't be able to push the models into the actual repo.

# Instructions

1. Download a model file(s) from the [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). 

2. Unzip, and then place the model(s) into this directory. 

3. For each model, you'll need to go into the model directory and find the *pipeline.config* file. This file is how you will tell the TensorFlow API how to set up your model. 

4. Inside of [my_configuration_files](my_configuration_files), you'll find a file with the name *<insert_model_name>pipeline.config*. You should overwrite the contents of the file named *pipeline.config* inside of your new model directory with this pre-made file inside of my_configuration_files. 

5. After overwriting the *pipeline.config* files, there are a few lines that will need to be changed
   - **label_map_path**: This should be changed to *path_on_your_machine/JetsonBenchmarking/model_retraining/data/mscoco_label_map.pbtxt*
   - **input_path**: (Under train_input_reader) *path_on_your_machine/JetsonBenchmarking/model_retraining/data/coco_2017_train.tfrecord*
   - **input_path**: (Under val_input_reader) *path_on_your_machine/JetsonBenchmarking/model_retraining/data/coco_2017_val.tfrecord*
   
 6. Lastly, to train your model, run the following command from the **TensorFlow GitHub repository** (Clone this repo [here](https://github.com/tensorflow/models))
 
```shell
 # From the tensorflow/models/research/ directory
   python object_detection/train.py \
 --logtostderr \
 --train_dir=${PATH_TO_TRAIN_DIR} \
 --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG}
 ```
  
