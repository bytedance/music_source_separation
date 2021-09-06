#!/bin/bash
# Download checkpoints from hdfs.
hdfs dfs -get "hdfs://haruna/home/byte_speech_sv/user/kongqiuqiang/released_models/source_separation/20210905_resnet_vocals/step=200000.pth" "20210905_resnet_vocals_8.7dB.pth"

hdfs dfs -get "hdfs://haruna/home/byte_speech_sv/user/kongqiuqiang/released_models/source_separation/20210905_resnet_accompaniment/step=200000.pth" "20210905_resnet_accompaniment_16.7dB.pth"

# config file
CONFIG_YAML="kqq_scripts/4_train/musdb18/configs/vocals-accompaniment,resunet.yaml"

# checkpoint paths
VOCALS_CHECKPOINT_PATH="20210905_resnet_vocals_8.7dB.pth"
ACCOMPANIMENT_CHECKPOINT_PATH="20210905_resnet_accompaniment_16.7dB.pth"

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