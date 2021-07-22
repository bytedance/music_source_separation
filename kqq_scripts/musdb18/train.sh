#!/bin/bash
WORKSPACE=${1:-"./workspaces/music_source_separation"}  # Default workspace directory

echo "WORKSPACE=${WORKSPACE}"

# Users can modify the following config file.
TRAIN_CONFIG_YAML="scripts/musdb18/configs/train/resnet143_decouple_plus.yaml"

# Train & evaluate & save checkpoints.
CUDA_VISIBLE_DEVICES=0,1 python3 music_source_separation/train.py train \
    --workspace=$WORKSPACE \
    --gpus=2 \
    --config_yaml=$TRAIN_CONFIG_YAML