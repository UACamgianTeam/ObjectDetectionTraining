#!/usr/bin/bash
# Used to train TensorFlow 1 models

# Save the current repo name
OBJECTDETECTIONTRAINING_REPO=$(pwd)

# Find the tensorflow/models/research repo
TENSORFLOW_MODEL_REPO=$(find ~ -type d -wholename "*/tensorflow/models/research" )

# Move into the tensorflow/models/research repo if it exists, otherwise exit with an error message
{ cd "$TENSORFLOW_MODEL_REPO" ; echo "tensorflow/models repo found at $TENSORFLOW_MODEL_REPO"; } || \
{ echo "Could not find tensorflow/models/research repository. Clone that repo and try again!" ; exit 1; }

# Get the model type that the user wishes to retrain
echo "Which model do you want to retrain?"
echo "1. ssd_mobilenet_v2_coco_2019_03_29"
echo "Select a number: "
read -r MODEL_NUM

# Set the path to the pipeline.config file and the output model directory based upon the model type
if [ "$MODEL_NUM" == 1 ] # ssd_mobilenet_v2_coco_2018_03_29
then
  PIPELINE_CONFIG_PATH={"${OBJECTDETECTIONTRAINING_REPO}/original_models/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config"}
  MODEL_DIR={"${OBJECTDETECTIONTRAINING_REPO}/retrained_models/ssd_mobilenet_v2_coco_2018_03_29"}
else
  { echo "Selection ${MODEL_NUM} not found. Try again, make sure the number you enter matches the model you wish to train!" ; exit 1; }
fi

# Set the number of training steps
NUM_TRAIN_STEPS=50000
# Sample 1 out of every N evaluation examples for eval
SAMPLE_1_OF_N_EVAL_EXAMPLES=1

# (For debugging) print out our variables
echo "PIPELINE_CONFIG_PATH: ${PIPELINE_CONFIG_PATH}"
echo "MODEL_DIR: ${MODEL_DIR}"
echo "NUM_TRAIN_STEPS: ${NUM_TRAIN_STEPS}"
echo "SAMPLE_1_OF_N_EVAL_EXAMPLES: ${SAMPLE_1_OF_N_EVAL_EXAMPLES}"

exit 0


## Run the training script
#echo "Running retraining script..."
#python object_detection/legacy/train.py \
#    --pipeline_config_path="${PIPELINE_CONFIG_PATH}" \
#    --model_dir="${MODEL_DIR}" \
#    --num_train_steps=${NUM_TRAIN_STEPS} \
#    --sample_1_of_n_eval_examples=${SAMPLE_1_OF_N_EVAL_EXAMPLES} \
#    --alsologtostderr