#!/bin/bash
AUDIO_PATH=${1:-"./resources/vocals_accompaniment_10s.mp3"}  # The path of audio to be separated.
OUTPUT_PATH=${2:-"./sep_results/sep_vocals.mp3"}  # The path to write out separated audio.

MODEL_NAME="resunet_subbandtime"        # "resunet_ismir2021" | ""resunet_subbandtime""

if [ $MODEL_NAME = "resunet_ismir2021" ]; then
    TRAIN_CONFIG_YAML="./scripts/4_train/musdb18/configs/vocals-accompaniment,resunet_ismir2021.yaml"
    CHECKPOINT_PATH="./downloaded_checkpoints/resunet143_ismir2021_vocals_8.9dB_350k_steps.pth"

elif [ $MODEL_NAME = "resunet_subbandtime" ]; then
    TRAIN_CONFIG_YAML="./scripts/4_train/musdb18/configs/vocals-accompaniment,resunet_subbandtime.yaml"
    CHECKPOINT_PATH="./downloaded_checkpoints/resunet143_subbtandtime_vocals_8.8dB_350k_steps.pth"
fi

# Inference
CUDA_VISIBLE_DEVICES=0 python3 bytesep/inference.py \
    --config_yaml=$TRAIN_CONFIG_YAML \
    --checkpoint_path=$CHECKPOINT_PATH \
    --audio_path=$AUDIO_PATH \
    --output_path=$OUTPUT_PATH
