# Alternate TFRecord Creator

This folder will house an alternate method for the creation of input data for training object detection models.

# How is this different from [tensorflow_object_detection_create_coco_tf_record](../tensorflow_object_detection_create_coco_tf_record)?

In [tensorflow_object_detection_create_coco_tf_record](../tensorflow_object_detection_create_coco_tf_record), we download the COCO dataset in the standard COCO format.
This format implies that all of the annotations for the train dataset are stored in one big JSON file, and likewise for the validation set. In that repo, we simply
used that format and some tensorflow APIs to only select a subset of the class labels at runtime during our 
[create_tf_record](../tensorflow_object_detection_create_coco_tf_record/create_tf_record) script.

In this repository, we break this up into multiple steps.

First, we convert the COCO-style JSON annotations into Pascal-VOC-style XML annotations. This separates the annotations for each class
into separate folders, and allows us to easily select our class subset simply by deleting or omitting the class folders that we don't
care about. 

# Instructions

1. Edit the 

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
  
