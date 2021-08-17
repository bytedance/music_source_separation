#!/bin/bash
DATASET_DIR=${1:-"./datasets/voicebank-demand"}  # The first argument is dataset directory.
WORKSPACE=${2:-"./workspaces/bytesep"}  # The second argument is workspace directory.

echo "DATASET_DIR=${DATASET_DIR}"
echo "WORKSPACE=${WORKSPACE}"

# Users can change the following settings.
SAMPLE_RATE=44100
CHANNELS=1

# Paths
PARENT_HDF5S_DIR="${WORKSPACE}/hdf5s/voicebank-demand/sr=${SAMPLE_RATE}_ch=${CHANNELS}"

# Pack train subset 100 pieces into hdf5 files.
HDF5S_DIR="${PARENT_HDF5S_DIR}/train"

python3 bytesep/dataset_creation/create_voicebank_demand.py pack_audios_to_hdf5s \
    --dataset_dir=$DATASET_DIR \
    --split="train" \
    --hdf5s_dir=$HDF5S_DIR \
    --sample_rate=$SAMPLE_RATE \
    --channels=$CHANNELS
    