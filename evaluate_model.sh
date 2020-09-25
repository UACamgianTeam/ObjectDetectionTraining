#!/usr/bin/env bash
# Used to evaluate models
# Should be run from .../ObjectDetectionTraining

# Get the model type that the user wishes to evaluate
printf "Which model do you want to evaluate?\n"
printf "1. ssd_mobilenet_v2_fpnlite\n"
printf "\n"
printf "Select a number: "
read -r MODEL_NUM

# Set the path to the pipeline.config file and the output model directory based upon the model type
if [ "$MODEL_NUM" == 1 ] # ssd_mobilenet_v2_fpnlite
then
  PIPELINE_CONFIG_PATH="models/retrained_models/my_ssd_mobilenet_v2_fpnlite/pipeline.config"
  ORIGINAL_PIPELINE_CONFIG_PATH="models/original_models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/pipeline.config"
  ORIGINAL_MODEL_DIR="models/original_models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8"
  ORIGINAL_MODEL_CHECKPOINT_DIR="models/original_models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/checkpoint"
  RETRAINED_MODEL_DIR="models/retrained_models/my_ssd_mobilenet_v2_fpnlite"
else
  { echo "Selection ${MODEL_NUM} not found. Try again, make sure the number you enter matches the model you wish to evaluate!" ; exit 1; }
fi

# (For debugging) print out our variables
printf "\nPIPELINE_CONFIG_PATH: %s\n" "$PIPELINE_CONFIG_PATH"
printf "ORIGINAL_MODEL_DIR: %s\n\n" "$ORIGINAL_MODEL_DIR"
printf "RETRAINED_MODEL_DIR: %s\n\n" "$RETRAINED_MODEL_DIR"

## Evaluate retrained model
#printf "Evaluating retrained model...\n"
#python model_main_tf2.py \
#    --model_dir="${RETRAINED_MODEL_DIR}" \
#    --pipeline_config_path="${PIPELINE_CONFIG_PATH}" \
#    --checkpoint_dir="${RETRAINED_MODEL_DIR}"
#printf "Done evaluating retrained model!\n"

# Evaluate original model
printf "Evaluating original model...\n"
python model_main_tf2.py \
    --model_dir="${ORIGINAL_MODEL_DIR}" \
    --pipeline_config_path="${ORIGINAL_PIPELINE_CONFIG_PATH}" \
    --checkpoint_dir="${ORIGINAL_MODEL_CHECKPOINT_DIR}"
printf "Done evaluating original model!\n"

