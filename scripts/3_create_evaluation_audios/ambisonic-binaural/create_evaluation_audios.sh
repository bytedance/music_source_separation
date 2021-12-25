#!/bin/bash
AMBISONIC_BINAURAL_DATASET_DIR=${1:-"./datasets/ambisonic-binaural"}  # The first argument is dataset directory.
WORKSPACE=${2:-"./workspaces/bytesep"}  # The first argument is workspace directory.

# Get absolute path
AMBISONIC_BINAURAL_DATASET_DIR=`readlink -f $AMBISONIC_BINAURAL_DATASET_DIR`

# Evaluation audios directory
EVALUATION_AUDIOS_DIR="${WORKSPACE}/evaluation_audios/ambisonic-binaural"

mkdir -p `dirname $EVALUATION_AUDIOS_DIR`

# Create link
ln -s $AMBISONIC_BINAURAL_DATASET_DIR $EVALUATION_AUDIOS_DIR