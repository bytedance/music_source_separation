#!/bin/bash

echo "The dataset can be downloaded at http://www.udialogue.org/download/VCTK-Corpus.tar.gz"

# The downloaded VCTK dataset looks like:
# ./datasets/vctk
# └── wav48
#     ├── train (100 speakers)
#     │   ├── p225 (231 files)
#     │   │   ├── p225_001_mic1.flac.wav
#     │   │   └── ...
#     │   ├── p226 (356 files)
#     │   │   ├── p226_001_mic1.flac.wav
#     │   │   └── ...
#     │   └── ...
#     └── test (8 speakers)
#         ├── p360 (424 files)
#         │   ├── p360_001_mic1.flac.wav
#         │   └── ...
#         ├── p226 (424 files)
#         │   ├── p361_001_mic1.flac.wav
#         │   └── ...
#         └── ...