#!/usr/bin/bash
# Used to visualize models using tensorboard

# Find and store the current path, which should be .../ObjectionDetectionTraining/models
OBJECTDETECTIONTRAINING_REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Move into the tensorflow/models/research repo if it exists, otherwise exit with an error message
if [ "$TENSORFLOW_MODEL_REPO" == "\n" ] || [ -z "$TENSORFLOW_MODEL_REPO" ]
then
  echo "Could not find tensorflow/models/research repository. Clone that repo and try again!" && exit 1
else
  cd "$TENSORFLOW_MODEL_REPO" && echo "tensorflow/models repo found at $TENSORFLOW_MODEL_REPO"
fi

# Get the model type that the user wishes to retrain
printf "Which model do you want to retrain?\n"
printf "1. ssd_mobilenet_v2_coco_2019_03_29\n"
printf "\n"
printf "Select a number: "
read -r MODEL_NUM

# Set the path to the log directory of the proper model
if [ "$MODEL_NUM" == 1 ] # ssd_mobilenet_v2_coco_2018_03_29
then
  EVAL_LOG_DIR="${OBJECTDETECTIONTRAINING_REPO}/original_models/ssd_mobilenet_v2_coco_2018_03_29/eval_0"
  EXPORT_DIR="${OBJECTDETECTIONTRAINING_REPO}/original_models/ssd_mobilenet_v2_coco_2018_03_29/export"
  MAIN_MODEL_DIR="${OBJECTDETECTIONTRAINING_REPO}/original_models/ssd_mobilenet_v2_coco_2018_03_29"
else
  { echo "Selection ${MODEL_NUM} not found. Try again, make sure the number you enter matches the model you wish to train!" ; exit 1; }
fi

# Run tensorboard on the EVAL_LOG_DIR
tensorboard --logdir "${EVAL_LOG_DIR}"
