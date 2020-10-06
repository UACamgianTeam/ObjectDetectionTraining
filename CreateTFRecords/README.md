# CreateTFRecords

This folder will house the method for the creation of input data for training object detection models.

In this repository, we break the conversion from COCO annotations into TensorFlow TFRecord files into multiple steps.

First, we convert the COCO-style JSON annotations into Pascal-VOC-style XML annotations. This separates the annotations for each class
into separate folders, and allows us to easily select our class subset simply by deleting or omitting the class folders that we don't
care about. 

# Instructions

## 1. Download COCO dataset
Download the COCO dataset from [here](https://cocodataset.org/#download), and put it under the
```.../ObjectDetectionTraining/data``` repository with the following directory structure:
```
Project Folder
└───CreateTFRecords
│   ...
└───data
    └───coco_2017   
        └───images
        │   └───train
        │   │    │   000000000009.jpg
        │   │    │   000000000025.jpg
        │   │    │   ...
        │   └───val   
        │        │   000000000139.jpg
        │        │   000000000285.jpg
        │        │   ...
        └───annotations
            │   instances_train.json
            │   instances_val.json
```

## 2. Convert COCO (JSON) annotations to Pascal VOC (XML) annotations
If the directory structure of the COCO dataset inside of this repository doesn't exactly match that of the above 
diagram, modify the annotation filepath and output directory arguments in 
[coco_to_voc_annotations.sh](), and then run the following command:
```
# Located in ObjectDetectionTraining/CreateTFRecords/alternate_tfrecord_creator/
bash coco_to_voc_annotations.sh
``` 

## 3. Parse newly created XML annotations
Now that we've converted our JSON annotations into XML annotations, we need to select the classes that we wish to 
retrain our model on.

To do so, change the "class_subset" variable in [parse_xml_annotations.py]().

Then, run [parse_xml_annotations.py]() to parse the XML annotations with the following command:
```
# Located in ObjectDetectionTraining/CreateTFRecords/alternate_tfrecord_creator/
python parse_xml_annotations.py
```

## 3. Convert parsed XML annotations to TFRecord
Finally, we can create new TFRecord files containing our training and validation data with the [generate_tfrecord.sh]()
bash script

Before doing so, uncomment either the ```## Validation data ``` or ```## Training data ``` sections of the
[generate_tfrecord.sh]() file for converting either the validation data or training data, and make sure to
change the arguments ```-l``` and ```-o``` to the label map path and output file name, respectively.

Then, run [generate_tfrecord.sh]() to convert the XML annotations into TFrecords
```
# Located in ObjectDetectionTraining/CreateTFRecords/alternate_tfrecord_creator/
bash generate_tfrecord.sh
```