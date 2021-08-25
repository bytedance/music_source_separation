#!/bin/bash
WORKSPACE=${1:-"./workspaces/bytesep"}  # Default workspace directory

echo "WORKSPACE=${WORKSPACE}"

# Users can modify the following config file.
INDEXES_CONFIG_YAML="kqq_scripts/create_indexes/vctk-audioset/configs/sr=32000_ch=1_balanced.yaml"

# Create indexes for training.
python3 bytesep/dataset_creation/create_vctk_audioset_indexes.py create_indexes \
    --workspace=$WORKSPACE \
    --config_yaml=$INDEXES_CONFIG_YAML
