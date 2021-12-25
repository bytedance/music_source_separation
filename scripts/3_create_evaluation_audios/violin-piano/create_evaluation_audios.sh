#!/bin/bash
VIOLIN_DATASET_DIR=${1:-"./datasets/instruments_solo/violin_solo/v0.1"}
PIANO_DATASET_DIR=${2:-"./datasets/maestro"}
WORKSPACE=${3:-"./workspaces/bytesep"}

SAMPLE_RATE=44100
CHANNELS=2
EVALUATION_SEGMENTS_NUM=100

EVLUATION_AUDIOS_DIR="${WORKSPACE}/instruments_dataset/violin-piano"

python3 bytesep/dataset_creation/create_evaluation_audios/violin-piano.py \
    --violin_dataset_dir=$VIOLIN_DATASET_DIR \
    --piano_dataset_dir=$PIANO_DATASET_DIR \
    --evaluation_audios_dir=$EVLUATION_AUDIOS_DIR \
    --sample_rate=$SAMPLE_RATE \
    --channels=$CHANNELS \
    --evaluation_segments_num=$EVALUATION_SEGMENTS_NUM
    