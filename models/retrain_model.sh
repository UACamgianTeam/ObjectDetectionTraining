#!/usr/bin/bash
# Used to train TensorFlow 1 models

# Find and store the current path, which should be .../ObjectionDetectionTraining/models
OBJECTDETECTIONTRAINING_REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "$OBJECTDETECTIONTRAINING_REPO" && exit 0

# Find and store the tensorflow/models/research repo path
TENSORFLOW_MODEL_REPO=$(find ~ -type d -wholename "*/tensorflow/models/research" )

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

# Set the path to the pipeline.config file and the output model directory based upon the model type
if [ "$MODEL_NUM" == 1 ] # ssd_mobilenet_v2_coco_2018_03_29
then
  PIPELINE_CONFIG_PATH="${OBJECTDETECTIONTRAINING_REPO}/original_models/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config"
  TRAIN_DIR="${OBJECTDETECTIONTRAINING_REPO}/retrained_models/ssd_mobilenet_v2_coco_2018_03_29"
else
  { echo "Selection ${MODEL_NUM} not found. Try again, make sure the number you enter matches the model you wish to train!" ; exit 1; }
fi

# (For debugging) print out our variables
printf "\nPIPELINE_CONFIG_PATH: %s\n" "$PIPELINE_CONFIG_PATH"
printf "TRAIN_DIR: %s\n\n" "$TRAIN_DIR"

# Update PYTHONPATH
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Run the training script
echo "Running retraining script..."
python object_detection/legacy/train.py \
    --pipeline_config_path="${PIPELINE_CONFIG_PATH}" \
    --train_dir="${TRAIN_DIR}"
