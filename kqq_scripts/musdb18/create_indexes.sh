#!/bin/bash
WORKSPACE=${1:-"./workspaces/music_source_separation"}  # Default workspace directory

echo "WORKSPACE=${WORKSPACE}"

# Users can modify the following config file.
INDEXES_CONFIG_YAML="scripts/musdb18/configs/create_indexes/vocals_accompaniment.yaml"

# Create indexes for training.
python3 music_source_separation/dataset_creation/create_indexes.py create_indexes \
    --workspace=$WORKSPACE \
    --config_yaml=$INDEXES_CONFIG_YAML
