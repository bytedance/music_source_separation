# Music source separation

Music source separation is a task to separate audio recordings into individual sources. This repository is an PyTorch implmementation of music source separation systems. Users can separate their favorite songs into different sources by installing this repository. Users can also train music source separation systems from scratch using this repository.

This repository can be also used for general separation tasks including 1) Speech enhancement. 2) Instrument separation. 3) Etc.

## Demos

TODO

## Installation

TODO

## Train a music source separation system from scratch

## 1. Download dataset

We use the MUSDB18 dataset to train music source separation systems. The trained system can separate vocals, accompaniments, bass, and other sources. Execute the following script to download and decompress the MUSDB18 dataset:

```bash
./scripts/download_musdb18_from_zenodo.sh
```

The dataset looks like:
<pre>
./datasets/musdb18
├── train (100 files)
│    ├── 'A Classic Education - NightOwl.stem.mp4'
│    └── ...
├── test (50 files)
│    ├── 'Al James - Schoolboy Facination.stem.mp4'
│    └── ...
└── README.md
</pre>

## 2. Pack audio files into hdf5 files

We pack audio waveforms into hdf5 files to speed up training.

```bash
./scripts/pack_musdb18_audios_to_hdf5s.sh
```

## 3. Create indexes for training

```bash
./scripts/create_indexes_musdb18.sh
```

## 4. Train & evaluate & save checkpoints
```bash
./scripts/train_musdb18.sh
```

## Reference

To appear.