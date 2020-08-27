#!/usr/bin/bash
# Used to train TensorFlow 1 models

# Find the tensorflow/models/research repo
TENSORFLOW_MODEL_REPO=$(find ~ -type d -wholename "*/tensorflow/models/research" )

# Move into the tensorflow/models/research repo if it exists, otherwise exit with an error message
{ cd "$TENSORFLOW_MODEL_REPO" ; echo "tensorflow/models repo found at $TENSORFLOW_MODEL_REPO"; } || \
{ echo "Could not find tensorflow/models/research repository. Clone that repo and try again!" ; exit 1; }

# Set the path to the pipeline.config file
PIPELINE_CONFIG_PATH={path to pipeline config file}
# Set the path to the model directory to save
MODEL_DIR={path to model directory}
# Set the number of training steps
NUM_TRAIN_STEPS=50000
# Sample 1 out of every N evaluation examples for eval
SAMPLE_1_OF_N_EVAL_EXAMPLES=1

# Run the training script
python object_detection/legacy/train.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=${SAMPLE_1_OF_N_EVAL_EXAMPLES} \
    --alsologtostderr