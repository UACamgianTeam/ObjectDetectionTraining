#!/usr/bin/env bash
# Converts PEViD-UHD annotations to Pascal-VOC XML annotations
# Should be run from .../ObjectDetectionTraining/

# Set the path to the input x.xgtf annotation file
input_gt_file="data/PEViD-UHD/walking_day_outdoor_3/Walking_day_outdoor_3_4K_trimmed.xgtf"

# Set the output folder of the resulting annotation.xml
output_folder="data/PEViD-UHD/walking_day_outdoor_3/"

# Convert the PEViD-UHD annotations to Pascal-VOC XML annotations
python PEViD-UHD/convert_pevid_to_voc.py --input_gt_file="${input_gt_file}" --output_folder="${output_folder}"
