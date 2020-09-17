#!/usr/bin/bash
# Used to visualize models using tensorboard

# Find and store the current path, which should be .../ObjectionDetectionTraining/models
OBJECTDETECTIONTRAINING_REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Get the model type that the user wishes to visualize
printf "Which model do you want to visualize?\n"
printf "1. ssd_mobilenet_v2\n"
printf "2. ssd_mobilenet_v1\n"
printf "\n"
printf "Select a number: "
read -r MODEL_NUM

# Set the path to the log directory of the proper model
if [ "$MODEL_NUM" == 1 ] # ssd_mobilenet_v2_coco_2018_03_29
then
  EVAL_LOG_DIR="${OBJECTDETECTIONTRAINING_REPO}/retrained_models/ssd_mobilenet_v2_coco_2018_03_29/eval_0"
  ORIGINAL_EVAL_SUMMARIES_DIR="${OBJECTDETECTIONTRAINING_REPO}/original_models/ssd_mobilenet_v2_coco_2018_03_29/eval_summaries"
  RETRAINED_EVAL_SUMMARIES_DIR="${OBJECTDETECTIONTRAINING_REPO}/retrained_models/ssd_mobilenet_v2_coco_2018_03_29/eval_summaries"
  EXPORT_DIR="${OBJECTDETECTIONTRAINING_REPO}/retrained_models/ssd_mobilenet_v2_coco_2018_03_29/export"
  MAIN_MODEL_DIR="${OBJECTDETECTIONTRAINING_REPO}/retrained_models/ssd_mobilenet_v2_coco_2018_03_29"
elif [ "$MODEL_NUM" == 2 ] # ssd_mobilenet_v1_coco_2018_01_28
then
  EVAL_LOG_DIR="${OBJECTDETECTIONTRAINING_REPO}/retrained_models/ssd_mobilenet_v1_coco_2018_01_28/eval_0"
  ORIGINAL_EVAL_SUMMARIES_DIR="${OBJECTDETECTIONTRAINING_REPO}/original_models/ssd_mobilenet_v1_coco_2018_01_28/eval_summaries"
  ORIGINAL_MAIN_MODEL_DIR="${OBJECTDETECTIONTRAINING_REPO}/original_models/ssd_mobilenet_v1_coco_2018_01_28/eval_summaries"
  RETRAINED_EVAL_SUMMARIES_DIR="${OBJECTDETECTIONTRAINING_REPO}/retrained_models/ssd_mobilenet_v1_coco_2018_01_28/eval_summaries"
  RETRAINED_MAIN_MODEL_DIR="${OBJECTDETECTIONTRAINING_REPO}/retrained_models/ssd_mobilenet_v1_coco_2018_01_28"
else
  { echo "Selection ${MODEL_NUM} not found. Try again, make sure the number you enter matches the model you wish to train!" ; exit 1; }
fi

## Run tensorboard on the EVAL_LOG_DIR
#tensorboard --logdir "${EVAL_LOG_DIR}"

## Run tensorboard on the MAIN_MODEL_DIR
#tensorboard --logdir "${MAIN_MODEL_DIR}"

## Run tensorboard on the ORIGINAL_EVAL_SUMMARIES_DIR
#tensorboard --logdir "${ORIGINAL_EVAL_SUMMARIES_DIR}"

# Run tensorboard on the ORIGINAL_MAIN_MODEL_DIR
tensorboard --logdir "${ORIGINAL_MAIN_MODEL_DIR}"

## Run tensorboard on the RETRAINED_EVAL_SUMMARIES_DIR
#tensorboard --logdir "${RETRAINED_EVAL_SUMMARIES_DIR}"
