#!/bin/bash
WORKSPACE=${1:-"./workspaces/bytesep"}  # Default workspace directory

echo "WORKSPACE=${WORKSPACE}"

# --- Create indexes for vocals and accompaniment ---
INDEXES_CONFIG_YAML="scripts/2_create_indexes/musdb18/configs/vocals-accompaniment,sr=44100,chn=2.yaml"

python3 bytesep/dataset_creation/create_indexes/create_indexes.py \
    --workspace=$WORKSPACE \
    --config_yaml=$INDEXES_CONFIG_YAML

# --- Create indexes for vocals, bass, drums, and other ---
INDEXES_CONFIG_YAML="scripts/2_create_indexes/musdb18/configs/vocals-bass-drums-other,sr=44100,chn=2.yaml"

python3 bytesep/dataset_creation/create_indexes/create_indexes.py \
    --workspace=$WORKSPACE \
    --config_yaml=$INDEXES_CONFIG_YAML
