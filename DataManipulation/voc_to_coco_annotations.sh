#!/usr/bin/env bash

# Create COCO annotations
python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/exchanging_bags_day_indoor_1/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/exchanging_bags_day_indoor_1/annotations/output.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/exchanging_bags_day_indoor_2/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/exchanging_bags_day_indoor_2/annotations/output.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/exchanging_bags_day_indoor_3/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/exchanging_bags_day_indoor_3/annotations/output.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/exchanging_bags_day_outdoor_4/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/exchanging_bags_day_outdoor_4/annotations/output.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/exchanging_bags_day_outdoor_5/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/exchanging_bags_day_outdoor_5/annotations/output.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/exchanging_bags_day_outdoor_6/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/exchanging_bags_day_outdoor_6/annotations/output.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/fighting_day_indoor_1/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/fighting_day_indoor_1/annotations/output.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/fighting_day_indoor_2/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/fighting_day_indoor_2/annotations/output.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/fighting_day_outdoor_4/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/fighting_day_outdoor_4/annotations/output.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/stealing_day_indoor_1/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/stealing_day_indoor_1/annotations/output.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/stealing_day_indoor_2/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/stealing_day_indoor_2/annotations/output.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/stealing_day_indoor_4/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/stealing_day_indoor_4/annotations/output.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/stealing_day_outdoor_5/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/stealing_day_outdoor_5/annotations/output.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/stealing_day_outdoor_6/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/stealing_day_outdoor_6/annotations/output.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/stealing_day_outdoor_7/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/stealing_day_outdoor_7/annotations/output.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/stealing_day_outdoor_8/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/stealing_day_outdoor_8/annotations/output.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/stealing_day_outdoor_9/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/stealing_day_outdoor_9/annotations/output.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/walking_day_indoor_4/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/walking_day_indoor_4/annotations/output.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/walking_day_outdoor_1/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/walking_day_outdoor_1/annotations/output.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/walking_day_outdoor_2/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/walking_day_outdoor_2/annotations/output.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/walking_day_outdoor_3/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/walking_day_outdoor_3/annotations/output.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/walking_day_outdoor_5/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/walking_day_outdoor_5/annotations/output.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/walking_day_outdoor_6/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/walking_day_outdoor_6/annotations/output.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/walking_night_indoor_7/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/walking_night_indoor_7/annotations/output.json

