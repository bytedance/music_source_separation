#!/bin/bash
MUSDB18_DATASET_DIR=${1:-"./datasets/musdb18"}  # The first argument is dataset directory.
WORKSPACE=${2:-"./workspaces/bytesep"}  # The second argument is workspace directory.

echo "MUSDB18_DATASET_DIR=${MUSDB18_DATASET_DIR}"
echo "WORKSPACE=${WORKSPACE}"

# Users can change the following settings.
SAMPLE_RATE=44100
CHANNELS=2

# Paths
PARENT_HDF5S_DIR="${WORKSPACE}/hdf5s/musdb18/sr=${SAMPLE_RATE}_ch=${CHANNELS}"

# Pack train subset 100 pieces into hdf5 files.
HDF5S_DIR="${PARENT_HDF5S_DIR}/train/full_train"

python3 bytesep/dataset_creation/create_musdb18.py pack_audios_to_hdf5s \
    --dataset_dir=$MUSDB18_DATASET_DIR \
    --subset="train" \
    --split="" \
    --hdf5s_dir=$HDF5S_DIR \
    --sample_rate=$SAMPLE_RATE \
    --channels=$CHANNELS


# Pack test subset 50 pieces into hdf5 files.
HDF5S_DIR="${PARENT_HDF5S_DIR}/test"

python3 bytesep/dataset_creation/create_musdb18.py pack_audios_to_hdf5s \
    --dataset_dir=$MUSDB18_DATASET_DIR \
    --subset="test" \
    --split="" \
    --hdf5s_dir=$HDF5S_DIR \
    --sample_rate=$SAMPLE_RATE \
    --channels=$CHANNELS
