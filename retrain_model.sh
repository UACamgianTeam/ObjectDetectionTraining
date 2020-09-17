#!/usr/bin/env bash
# Used to retrain models
# Should be run from .../ObjectDetectionTraining

# Get the model type to train
printf "Which model do you want to train?\n"
printf "1. ssd_mobilenet_v2_fpnlite\n"
printf "\n"
printf "Select a number: "
read -r MODEL_NUM

# Set the path to the pipeline.config file and the output model directory based on the model type
if [ "$MODEL_NUM" == 1 ] # ssd_mobilenet_v2_fpnlite
then
  PIPELINE_CONFIG_PATH="models/retrained_models/my_ssd_mobilenet_v2_fpnlite/pipeline.config"
  MODEL_DIR="models/retrained_models/my_ssd_mobilenet_v2_fpnlite"
else
  { echo "Selection ${MODEL_NUM} not found. Try again, make sure the number you enter matches the model you wish to train!" ; exit 1; }
fi

# (For debugging) print out our variables
printf "\nPIPELINE_CONFIG_PATH: %s\n" "$PIPELINE_CONFIG_PATH"
printf "ORIGINAL_MODEL_DIR: %s\n\n" "$ORIGINAL_MODEL_DIR"
printf "RETRAINED_MODEL_DIR: %s\n\n" "$RETRAINED_MODEL_DIR"

## Retrain model
printf "Retraining model...\n"
python model_main_tf2.py \
    --model_dir="${MODEL_DIR}" \
    --pipeline_config_path="${PIPELINE_CONFIG_PATH}" \
printf "Done retraining model!\n"
