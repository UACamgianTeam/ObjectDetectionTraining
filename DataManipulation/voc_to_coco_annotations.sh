#!/usr/bin/env bash

# Create COCO annotations for each of the videos in PEViD-UHD

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/exchanging_bags_day_indoor_1/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/exchanging_bags_day_indoor_1/annotations/exchanging_bags_day_indoor_1.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/exchanging_bags_day_indoor_2/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/exchanging_bags_day_indoor_2/annotations/exchanging_bags_day_indoor_2.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/exchanging_bags_day_indoor_3/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/exchanging_bags_day_indoor_3/annotations/exchanging_bags_day_indoor_3.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/exchanging_bags_day_outdoor_4/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/exchanging_bags_day_outdoor_4/annotations/exchanging_bags_day_outdoor_4.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/exchanging_bags_day_outdoor_5/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/exchanging_bags_day_outdoor_5/annotations/exchanging_bags_day_outdoor_5.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/exchanging_bags_day_outdoor_6/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/exchanging_bags_day_outdoor_6/annotations/exchanging_bags_day_outdoor_6.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/fighting_day_indoor_1/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/fighting_day_indoor_1/annotations/fighting_day_indoor_1.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/fighting_day_indoor_2/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/fighting_day_indoor_2/annotations/fighting_day_indoor_2.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/fighting_day_outdoor_4/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/fighting_day_outdoor_4/annotations/fighting_day_outdoor_4.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/stealing_day_indoor_1/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/stealing_day_indoor_1/annotations/stealing_day_indoor_1.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/stealing_day_indoor_2/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/stealing_day_indoor_2/annotations/stealing_day_indoor_2.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/stealing_day_indoor_4/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/stealing_day_indoor_4/annotations/stealing_day_indoor_4.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/stealing_day_outdoor_5/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/stealing_day_outdoor_5/annotations/stealing_day_outdoor_5.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/stealing_day_outdoor_6/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/stealing_day_outdoor_6/annotations/stealing_day_outdoor_6.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/stealing_day_outdoor_7/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/stealing_day_outdoor_7/annotations/stealing_day_outdoor_7.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/stealing_day_outdoor_8/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/stealing_day_outdoor_8/annotations/stealing_day_outdoor_8.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/stealing_day_outdoor_9/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/stealing_day_outdoor_9/annotations/stealing_day_outdoor_9.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/walking_day_indoor_4/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/walking_day_indoor_4/annotations/walking_day_indoor_4.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/walking_day_outdoor_1/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/walking_day_outdoor_1/annotations/walking_day_outdoor_1.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/walking_day_outdoor_2/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/walking_day_outdoor_2/annotations/walking_day_outdoor_2.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/walking_day_outdoor_3/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/walking_day_outdoor_3/annotations/walking_day_outdoor_3.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/walking_day_outdoor_5/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/walking_day_outdoor_5/annotations/walking_day_outdoor_5.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/walking_day_outdoor_6/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/walking_day_outdoor_6/annotations/walking_day_outdoor_6.json

python3 voc_to_coco_annotations.py \
--ann_paths_list ../data/PEViD-UHD/walking_night_indoor_7/annotations/annpaths_list.txt \
--labels ../data/PEViD-UHD/labels.txt \
--output_file ../data/PEViD-UHD/walking_night_indoor_7/annotations/walking_night_indoor_7.json

