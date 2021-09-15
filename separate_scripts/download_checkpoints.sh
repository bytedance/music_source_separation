#!/bin/bash

ZENODO_PATH="https://zenodo.org/record/5507029/files"
DOWNLOADED_CHECKPOINT_DIR="./downloaded_checkpoints"

mkdir -p $DOWNLOADED_CHECKPOINT_DIR

MODEL_NAME="resunet143_ismir2021_vocals_8.9dB_350k_steps.pth"
wget -O "${DOWNLOADED_CHECKPOINT_DIR}/${MODEL_NAME}" "${ZENODO_PATH}/${MODEL_NAME}?download=1"

MODEL_NAME="resunet143_ismir2021_accompaniment_16.8dB_350k_steps.pth"
wget -O "${DOWNLOADED_CHECKPOINT_DIR}/${MODEL_NAME}" "${ZENODO_PATH}/${MODEL_NAME}?download=1"