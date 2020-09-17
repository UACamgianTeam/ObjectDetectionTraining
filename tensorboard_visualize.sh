#!/usr/bin/env bash
# Used to run tensorboard visualizations of models
# Should be run from .../ObjectDetectionTraining

# Get the model type that the user wishes to evaluate
printf "Which model do you want to evaluate?\n"
printf "1. ssd_mobilenet_v2_fpnlite (original)\n"
printf "2. ssd_mobilenet_v2_fpnlite (retrained)\n"
printf "\n"
printf "Select a number: "
read -r MODEL_NUM

# Set the path to the pipeline.config file and the output model directory based upon the model type
if [ "$MODEL_NUM" == 1 ] # ssd_mobilenet_v2_fpnlite (original)
then
  LOGDIR="models/original_models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/eval"
elif [ "$MODEL_NUM" == 2 ] # ssd_mobilnet_v2_fpnlite (original)
then
  LOGDIR="models/retrained_models/my_ssd_mobilenet_v2_fpnlite/eval"
else
  { echo "Selection ${MODEL_NUM} not found. Try again, make sure the number you enter matches the model you wish to visualize!" ; exit 1; }
fi

## Visualize model
printf "Running tensorboard...\n"
tensorboard --logdir="${LOGDIR}"
