#!/bin/bash
VOICEBANK_DEMAND_DATASET_DIR=${1:-"./datasets/voicebank-demand"}  # The first argument is dataset directory.
WORKSPACE=${2:-"./workspaces/bytesep"}  # The first argument is workspace directory.

# Get absolute path
VOICEBANK_DEMAND_DATASET_DIR=`readlink -f $VOICEBANK_DEMAND_DATASET_DIR`

# Evaluation audios directory
EVALUATION_AUDIOS_DIR="${WORKSPACE}/evaluation_audios/voicebank-demand"

mkdir -p `dirname $EVALUATION_AUDIOS_DIR`

# Create link
ln -s $VOICEBANK_DEMAND_DATASET_DIR $EVALUATION_AUDIOS_DIR