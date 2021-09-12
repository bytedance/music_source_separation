#!/bin/bash
WORKSPACE=${1:-"./workspaces/bytesep"}  # The first argument is workspace directory.

echo "WORKSPACE=${WORKSPACE}"

# Users can modify the following config file.
TRAIN_CONFIG_YAML="scripts/4_train/musdb18/configs/vocals-accompaniment,unet.yaml"

CHECKPOINT_PATH="${WORKSPACE}/checkpoints/musdb18/train/config=vocals-accompaniment,unet,gpus=1/step=300000.pth"

# Inference
CUDA_VISIBLE_DEVICES=0 python3 bytesep/inference.py \
    --config_yaml=$TRAIN_CONFIG_YAML \
    --checkpoint_path=$CHECKPOINT_PATH \
    --audio_path="resources/vocals_accompaniment_10s.mp3" \
    --output_path="sep_results/vocals_accompaniment_10s_sep_vocals.mp3"
    