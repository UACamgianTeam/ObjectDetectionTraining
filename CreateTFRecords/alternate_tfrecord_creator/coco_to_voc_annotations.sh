#!/usr/bin/env bash

python3 anno_coco2voc.py --anno_file /home/ubuntu/Downloads/coco_dataset/annotations/instances_train2017.json \
                         --type instance \
                         --output_dir /home/ubuntu/Downloads/coco_dataset/annotations/instance_train_annotation
python3 anno_coco2voc.py --anno_file /home/ubuntu/Downloads/coco_dataset/annotations/instances_val2017.json \
                         --type instance \
                         --output_dir /home/ubuntu/Downloads/coco_dataset/annotations/instance_val_annotation

#python anno_coco2voc.py --anno_file /media/arkenstone/startdt_data/COCO/dataset/annotations/person_keypoints_train2014.json \
#                        --type keypoint \
#                        --output_dir /media/arkenstone/startdt_data/COCO/dataset/keypoints_train_annotation
#python anno_coco2voc.py --anno_file /media/arkenstone/startdt_data/COCO/dataset/annotations/person_keypoints_val2014.json \
#                        --type keypoint \
#                        --output_dir /media/arkenstone/startdt_data/COCO/dataset/keypoints_val_annotation