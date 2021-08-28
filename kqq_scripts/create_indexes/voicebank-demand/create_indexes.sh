#!/bin/bash
WORKSPACE=${1:-"./workspaces/bytesep"}  # Default workspace directory

echo "WORKSPACE=${WORKSPACE}"

# Users can modify the following config file.
INDEXES_CONFIG_YAML="scripts/create_indexes/voicebank-demand/configs/sr=44100_chn=1.yaml"

# Create indexes for training.
python3 bytesep/dataset_creation/create_indexes/create_indexes.py \
    --workspace=$WORKSPACE \
    --config_yaml=$INDEXES_CONFIG_YAML
