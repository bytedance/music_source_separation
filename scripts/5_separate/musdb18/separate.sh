#!/bin/bash
WORKSPACE=${1:-"./workspaces/bytesep"}  # The first argument is workspace directory.

echo "WORKSPACE=${WORKSPACE}"

# Config yaml path.
TRAIN_CONFIG_YAML="./scripts/4_train/musdb18/configs/vocals-accompaniment,resunet_subbandtime.yaml"

# Checkpoint path.
CHECKPOINT_PATH="${WORKSPACE}/checkpoints/musdb18/train/config=vocals-accompaniment,resunet_subbandtime,gpus=2/step=500000.pth"

# Inference.
CUDA_VISIBLE_DEVICES=0 python3 bytesep/separate.py separate_file \
    --config_yaml=$TRAIN_CONFIG_YAML \
    --checkpoint_path=$CHECKPOINT_PATH \
    --audio_path="resources/vocals_accompaniment_10s.mp3" \
    --output_path="sep_results/vocals_accompaniment_10s_sep_vocals.mp3"
    