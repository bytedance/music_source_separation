#!/bin/bash
MUSDB18_DATASET_DIR=${1:-"./datasets/musdb18"}  # The first argument is dataset directory.
WORKSPACE=${2:-"./workspaces/bytesep"}  # The second argument is workspace directory.

echo "WORKSPACE=${WORKSPACE}"

# Users can modify the following config file.
TRAIN_CONFIG_YAML="scripts/musdb18/configs/train/resnet143_decouple_plus.yaml"

# Train & evaluate & save checkpoints.
CUDA_VISIBLE_DEVICES=0,1 python3 bytesep/train.py train \
    --dataset_dir=$MUSDB18_DATASET_DIR \
    --workspace=$WORKSPACE \
    --gpus=2 \
    --config_yaml=$TRAIN_CONFIG_YAML