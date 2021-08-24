#!/bin/bash
SYMPHONY_DATASET_DIR="/home/tiger/datasets/instruments_dataset/dataset_root/symphony_solo/v0.1"
PIANO_DATASET_DIR="/home/tiger/datasets/maestro/dataset_root"

SAMPLE_RATE=44100
CHANNELS=2
EVLUATION_AUDIOS_DIR="${WORKSPACE}/evaluation_audios/piano-symphony/sr=${SAMPLE_RATE}_ch=${CHANNELS}"

python3 bytesep/dataset_creation/evaluation/piano_symphony.py create_evaluation \
    --symphony_dataset_dir=$SYMPHONY_DATASET_DIR \
    --piano_dataset_dir=$PIANO_DATASET_DIR \
    --evaluation_audios_dir=$EVLUATION_AUDIOS_DIR \
    --sample_rate=$SAMPLE_RATE \
    --channels=$CHANNELS
