#!/bin/bash
MUSDB18_DATASET_DIR=${1:-"./datasets/musdb18"}  # The first argument is dataset directory.
WORKSPACE=${2:-"./workspaces/bytesep"}  # The first argument is workspace directory.

# Get absolute path
MUSDB18_DATASET_DIR=`readlink -f $MUSDB18_DATASET_DIR`

# Evaluation audios directory
EVALUATION_AUDIOS_DIR="${WORKSPACE}/evaluation_audios/musdb18"

mkdir -p `dirname $EVALUATION_AUDIOS_DIR`

# Create link
ln -s $MUSDB18_DATASET_DIR $EVALUATION_AUDIOS_DIR