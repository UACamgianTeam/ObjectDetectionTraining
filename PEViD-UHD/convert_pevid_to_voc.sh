#!/usr/bin/env bash
# Converts PEViD-UHD annotations to Pascal-VOC XML annotations
# Should be run from .../ObjectDetectionTraining/

# Set the path to the input x.xgtf annotation file
input_gt_file="data/PEViD-UHD/stealing_day_indoor_1/Stealing_day_indoor_1_4K.xgtf"

# Set the output folder of the resulting annotation.xml
output_folder="data/PEViD-UHD/stealing_day_indoor_1/annotations"

# Convert the PEViD-UHD annotations to Pascal-VOC XML annotations
python PEViD-UHD/convert_pevid_to_voc.py --input_gt_file="${input_gt_file}" --output_folder="${output_folder}"
