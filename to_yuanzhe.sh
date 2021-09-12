#!/bin/bash
# Download checkpoints from hdfs.
hdfs dfs -get "hdfs://haruna/home/byte_speech_sv/user/kongqiuqiang/released_models/source_separation/20210912_resunet143_ismir/resunet143_ismir2021_vocals_8.9dB_350k_steps.pth"

hdfs dfs -get "hdfs://haruna/home/byte_speech_sv/user/kongqiuqiang/released_models/source_separation/20210912_resunet143_ismir/resunet143_ismir2021_accompaniment_16.8dB_350k_steps.pth"

# config file
CONFIG_YAML="kqq_scripts/4_train/musdb18/configs/vocals-accompaniment,resunet_ismir2021.yaml"

# checkpoint paths
VOCALS_CHECKPOINT_PATH="resunet143_ismir2021_vocals_8.9dB_350k_steps.pth"
ACCOMPANIMENT_CHECKPOINT_PATH="resunet143_ismir2021_accompaniment_16.8dB_350k_steps.pth"

# Inference.
CUDA_VISIBLE_DEVICES=0 python3 bytesep/inference.py \
    --config_yaml=$CONFIG_YAML \
    --checkpoint_path=$VOCALS_CHECKPOINT_PATH \
    --audio_path='resources/vocals_accompaniment_10s.mp3' \
    --output_path='sep_results/vocals_accompaniment_10s.mp3'

# Inference many files.
CUDA_VISIBLE_DEVICES=0 python3 bytesep/inference_many.py \
    --config_yaml=$CONFIG_YAML \
    --checkpoint_path=$VOCALS_CHECKPOINT_PATH \
    --audios_dir='resources/音频物料 - 人声+bgm的内容_mp3s' \
    --output_dir='sep_results/音频物料 - 人声+bgm的内容_mp3s'