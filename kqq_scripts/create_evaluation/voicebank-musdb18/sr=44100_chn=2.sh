#!/bin/bash
VOICEBANK_DATASET_DIR="/home/tiger/datasets/voicebank-demand"
MUSDB18_DATASET_DIR="/home/tiger/my_code_2019.12-/python/music_source_separation/datasets/musdb18"

SAMPLE_RATE=44100
CHANNELS=2
EVLUATION_AUDIOS_DIR="${WORKSPACE}/evaluation_audios/voicebank-musdb18/sr=${SAMPLE_RATE}_ch=${CHANNELS}"

python3 bytesep/dataset_creation/evaluation/voicebank-musdb18.py create_evaluation \
    --voicebank_dataset_dir=$VOICEBANK_DATASET_DIR \
    --musdb18_dataset_dir=$MUSDB18_DATASET_DIR \
    --evaluation_audios_dir=$EVLUATION_AUDIOS_DIR \
    --sample_rate=$SAMPLE_RATE \
    --channels=$CHANNELS
    