#!/usr/bin/env bash

python3 coco_to_voc_annotations.py --anno_file ../../data/coco_2017/annotations/instances_train2017.json \
                         --type instance \
                         --output_dir ../../data/coco_2017/annotations/instance_train_annotation
python3 coco_to_voc_annotations.py --anno_file ../../data/coco_2017/annotations/instances_val2017.json \
                         --type instance \
                         --output_dir ../../data/coco_2017/annotations/instance_val_annotation
