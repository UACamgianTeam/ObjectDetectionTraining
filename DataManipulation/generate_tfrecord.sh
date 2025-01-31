#!/usr/bin/env bash
# Runs generate_tfrecord.py and passes command line arguments to signify directories and parameters

## Validation data
#python generate_tfrecord.py -i ../../data/coco_2017/val2017 \
#-x ../../data/coco_2017/annotations/instance_val_annotation \
#-l ../../data/reduced_label_map_9labels.pbtxt \
#-o ../../data/coco_2017/annotations/coco_2017_val_9labels.record
#
## Training data
#python generate_tfrecord.py -i ../../data/coco_2017/train2017 \
#-x ../../data/coco_2017/annotations/instance_train_annotation \
#-l ../../data/reduced_label_map_9labels.pbtxt \
#-o ../../data/coco_2017/annotations/coco_2017_train_9labels.record

# PEViD-UHD data (currently all of the dataset, need to split into train/val splits
python generate_tfrecord.py -i ../data/PEViD-UHD \
-x ../data/PEViD-UHD \
-l ../data/label_maps/pevid_label_map.pbtxt \
-o ../data/tfrecords/pevid_uhd_val.record
