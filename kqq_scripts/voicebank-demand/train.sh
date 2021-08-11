#!/bin/bash
WORKSPACE=${1:-"./workspaces/music_source_separation"}  # Default workspace directory

echo "WORKSPACE=${WORKSPACE}"

# Users can modify the following config file.
TRAIN_CONFIG_YAML="scripts/voicebank-demand/configs/train/resnet143_decouple_plus.yaml"

VOICEBANK_DEMAND_DATASET_DIR="/home/tiger/datasets/voicebank-demand"

# Train & evaluate & save checkpoints.
CUDA_VISIBLE_DEVICES=0,1 python3 music_source_separation/train.py train \
    --workspace=$WORKSPACE \
    --gpus=2 \
    --config_yaml=$TRAIN_CONFIG_YAML \
    --dataset_dir=$VOICEBANK_DEMAND_DATASET_DIR