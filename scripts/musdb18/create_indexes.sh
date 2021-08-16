#!/bin/bash
WORKSPACE=${1:-"./workspaces/bytesep"}  # Default workspace directory

echo "WORKSPACE=${WORKSPACE}"

# Create indexes for vocals and accompaniment.
INDEXES_CONFIG_YAML="scripts/musdb18/configs/create_indexes/vocals_accompaniment.yaml"

python3 bytesep/dataset_creation/create_indexes.py create_indexes \
    --workspace=$WORKSPACE \
    --config_yaml=$INDEXES_CONFIG_YAML

# Create indexes for vocals, bass, drums, and other.
INDEXES_CONFIG_YAML="scripts/musdb18/configs/create_indexes/vocals_bass_drums_other.yaml"

python3 bytesep/dataset_creation/create_indexes.py create_indexes \
    --workspace=$WORKSPACE \
    --config_yaml=$INDEXES_CONFIG_YAML
