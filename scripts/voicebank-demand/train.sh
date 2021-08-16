#!/bin/bash
DATASET_DIR=${1:-"./datasets/voicebank-demand"}  # The first argument is dataset directory.
WORKSPACE=${2:-"./workspaces/bytesep"}  # The second argument is workspace directory.

echo "WORKSPACE=${WORKSPACE}"

# Users can modify the following config file.
TRAIN_CONFIG_YAML="scripts/voicebank-demand/configs/train/resnet143_decouple_plus.yaml"

# Train & evaluate & save checkpoints.
CUDA_VISIBLE_DEVICES=0,1 python3 bytesep/train.py train \
	--dataset_dir=$DATASET_DIR \
    --workspace=$WORKSPACE \
    --gpus=2 \
    --config_yaml=$TRAIN_CONFIG_YAML