#!/bin/bash
PIANO_DATASET_DIR=${1:-"./datasets/maestro"}
SYMPHONY_DATASET_DIR=${2:-"./datasets/instruments_dataset/symphony_solo/v0.1"}
WORKSPACE=${3:-"./workspaces/bytesep"}

SAMPLE_RATE=44100
CHANNELS=2
EVALUATION_SEGMENTS_NUM=100

EVLUATION_AUDIOS_DIR="${WORKSPACE}/evaluation_audios/piano-symphony"

python3 bytesep/dataset_creation/create_evaluation_audios/piano-symphony.py \
    --piano_dataset_dir=$PIANO_DATASET_DIR \
    --symphony_dataset_dir=$SYMPHONY_DATASET_DIR \
    --evaluation_audios_dir=$EVLUATION_AUDIOS_DIR \
    --sample_rate=$SAMPLE_RATE \
    --channels=$CHANNELS \
    --evaluation_segments_num=$EVALUATION_SEGMENTS_NUM
    