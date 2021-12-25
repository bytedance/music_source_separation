#!/bin/bash
AMBISONIC_BINAURAL_DATASET_DIR=${1:-"./datasets/ambisonic-binaural"}  # The first argument is dataset directory.
WORKSPACE=${2:-"./workspaces/bytesep"}  # The second argument is workspace directory.

echo "AMBISONIC_BINAURAL_DATASET_DIR=${AMBISONIC_BINAURAL_DATASET_DIR}"
echo "WORKSPACE=${WORKSPACE}"

# Users can change the following settings.
SAMPLE_RATE=48000

SPLIT="train"

for SOURCE_TYPE in 'ambisonic' 'binaural'
do
    python3 bytesep/dataset_creation/pack_audios_to_hdf5s/ambisonic-binaural.py pack_audios_to_hdf5s \
        --audios_dir="${AMBISONIC_BINAURAL_DATASET_DIR}/${SPLIT}/${SOURCE_TYPE}" \
        --hdf5s_dir="${WORKSPACE}/hdf5s/ambisonic-binaural/sr=${SAMPLE_RATE}/${SPLIT}/${SOURCE_TYPE}" \
        --sample_rate=$SAMPLE_RATE
done