#!/bin/bash
MUSDB18_DATASET_DIR=${1:-"./datasets/musdb18"}	# Default dataset directory

echo "MUSDB18_DATASET_DIR=${MUSDB18_DATASET_DIR}"

# Set up paths.
mkdir -p $MUSDB18_DATASET_DIR
cd $MUSDB18_DATASET_DIR

# Download dataset from Zenodo.
wget -O "musdb18.zip" "https://zenodo.org/record/1117372/files/musdb18.zip?download=1"

# Unzip dataset.
unzip "musdb18.zip"
