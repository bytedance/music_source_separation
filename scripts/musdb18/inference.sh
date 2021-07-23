#!/bin/bash
CONFIG_YAML="scripts/musdb18/configs/train/01.yaml"
CHECKPOINT_PATH="./workspaces/music_source_separation/checkpoints/musdb18/train/config=01,gpus=2/step=50000.pth"

CUDA_VISIBLE_DEVICES=5 python3 music_source_separation/inference.py \
    --config_yaml=$CONFIG_YAML \
    --checkpoint_path=$CHECKPOINT_PATH \
    --audio_path='resources/vocals_accompaniment_10s.mp3' \
    --output_path='sep_results/vocals_accompaniment_10s.mp3' \
    --select='1a'
