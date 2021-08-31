#!/bin/bash
VCTK_DATASET_DIR=${1:-"./datasets/vctk"}
MUSDB18_DATASET_DIR=${2:-"./datasets/musdb18"}
WORKSPACE=${3:-"./workspaces/bytesep"}

SAMPLE_RATE=44100
CHANNELS=2
EVALUATION_SEGMENTS_NUM=100

EVLUATION_AUDIOS_DIR="${WORKSPACE}/evaluation_audios/vctk-musdb18"

python3 bytesep/dataset_creation/create_evaluation_audios/vctk-musdb18.py \
    --vctk_dataset_dir=$VCTK_DATASET_DIR \
    --musdb18_dataset_dir=$MUSDB18_DATASET_DIR \
    --evaluation_audios_dir=$EVLUATION_AUDIOS_DIR \
    --sample_rate=$SAMPLE_RATE \
    --channels=$CHANNELS \
    --evaluation_segments_num=$EVALUATION_SEGMENTS_NUM
    