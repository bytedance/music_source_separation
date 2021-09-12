#!/bin/bash
MUSDB18_DATASET_DIR=${1:-"./datasets/musdb18"}	# The first argument is dataset directory.

echo "MUSDB18_DATASET_DIR=${MUSDB18_DATASET_DIR}"

# Set up paths.
mkdir -p $MUSDB18_DATASET_DIR
cd $MUSDB18_DATASET_DIR

# Download dataset from Zenodo.
echo "The dataset link is at https://zenodo.org/record/1117372"

wget -O "musdb18.zip" "https://zenodo.org/record/1117372/files/musdb18.zip?download=1"

# Unzip dataset.
unzip "musdb18.zip"

# The downloaded MUSDB18 dataset looks like:
# ./datasets/musdb18
# ├── train (100 files)
# │   ├── 'A Classic Education - NightOwl.stem.mp4'
# │   └── ...
# ├── test (50 files)
# │   ├── 'Al James - Schoolboy Facination.stem.mp4'
# │   └── ...
# └── README.md