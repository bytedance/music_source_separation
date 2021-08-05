#!/bin/bash
hdfs dfs -get "hdfs://haruna/home/byte_speech_sv/user/kongqiuqiang/released_models/source_separation/20210805_resnet143_vocals_ismir2021/step\=300000.pth" "20210805_resnet143_vocals_ismir2021.pth"

CONFIG_YAML="kqq_scripts/musdb18/configs/train/resnet143_vocals_ismir2021.yaml"
CHECKPOINT_PATH="20210805_resnet143_vocals_ismir2021.pth"

CUDA_VISIBLE_DEVICES=3 python3 music_source_separation/inference.py \
    --config_yaml=$CONFIG_YAML \
    --checkpoint_path=$CHECKPOINT_PATH \
    --audio_path='resources/vocals_accompaniment_10s.mp3' \
    --output_path='sep_results/vocals_accompaniment_10s.mp3'
