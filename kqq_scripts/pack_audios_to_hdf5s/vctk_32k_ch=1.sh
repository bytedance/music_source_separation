#!/bin/bash
DATASET_DIR=${1:-"./datasets/vctk"}  # The first argument is dataset directory.
WORKSPACE=${2:-"./workspaces/bytesep"}  # The second argument is workspace directory.

echo "DATASET_DIR=${DATASET_DIR}"
echo "WORKSPACE=${WORKSPACE}"

# Users can change the following settings.
SAMPLE_RATE=32000
CHANNELS=1

# Paths
HDF5S_DIR="${WORKSPACE}/hdf5s/vctk/sr=${SAMPLE_RATE}_ch=${CHANNELS}/train"

python3 bytesep/dataset_creation/create_vctk.py pack_audios_to_hdf5s \
    --dataset_dir=$DATASET_DIR \
    --split="train" \
    --source_type="speech" \
    --hdf5s_dir=$HDF5S_DIR \
    --sample_rate=$SAMPLE_RATE \
    --channels=$CHANNELS
    