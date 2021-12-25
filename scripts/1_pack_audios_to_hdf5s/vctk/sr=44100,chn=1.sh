#!/bin/bash
DATASET_DIR=${1:-"./datasets/vctk"}  # The first argument is dataset directory.
WORKSPACE=${2:-"./workspaces/bytesep"}  # The second argument is workspace directory.

echo "DATASET_DIR=${DATASET_DIR}"
echo "WORKSPACE=${WORKSPACE}"

# Users can change the following settings.
SAMPLE_RATE=44100
CHANNELS=1

# Paths
SPLIT="train"
HDF5S_DIR="${WORKSPACE}/hdf5s/vctk/sr=${SAMPLE_RATE},chn=${CHANNELS}/${SPLIT}"

python3 bytesep/dataset_creation/pack_audios_to_hdf5s/vctk.py \
    --dataset_dir=$DATASET_DIR \
    --split=$SPLIT \
    --hdf5s_dir=$HDF5S_DIR \
    --sample_rate=$SAMPLE_RATE \
    --channels=$CHANNELS
    