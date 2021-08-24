#!/bin/bash
WORKSPACE=${1:-"./workspaces/bytesep"}  # Default workspace directory

echo "WORKSPACE=${WORKSPACE}"

# Users can modify the following config file.
TRAIN_CONFIG_YAML="kqq_scripts/train/vctk-musdb18/configs/01.yaml"

# Train & evaluate & save checkpoints.
CUDA_VISIBLE_DEVICES=1 python3 bytesep/train.py train \
    --workspace=$WORKSPACE \
    --gpus=1 \
    --config_yaml=$TRAIN_CONFIG_YAML \
    --dataset_dir=""