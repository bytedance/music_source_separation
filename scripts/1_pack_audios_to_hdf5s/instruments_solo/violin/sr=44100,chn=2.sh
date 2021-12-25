#!/bin/bash
VIOLIN_SOLO_DATASET_DIR=${1:-"./datasets/instruments_dataset/violin_solo/v0.1"}  # The first argument is dataset directory.
WORKSPACE=${2:-"./workspaces/bytesep"}  # The second argument is workspace directory.

echo "VIOLIN_SOLO_DATASET_DIR=${VIOLIN_SOLO_DATASET_DIR}"
echo "WORKSPACE=${WORKSPACE}"

# Users can change the following settings.
SAMPLE_RATE=44100
CHANNELS=2

INSTRUMENT="violin"

# Paths
SPLIT="train"
HDF5S_DIR="${WORKSPACE}/hdf5s/instruments_dataset/violin_solo/sr=${SAMPLE_RATE},chn=${CHANNELS}/${SPLIT}"

python3 bytesep/dataset_creation/pack_audios_to_hdf5s/instruments_solo.py \
    --dataset_dir=$VIOLIN_SOLO_DATASET_DIR \
    --split=$SPLIT \
    --source_type=$INSTRUMENT \
    --hdf5s_dir=$HDF5S_DIR \
    --sample_rate=$SAMPLE_RATE \
    --channels=$CHANNELS