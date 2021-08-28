# Music source separation

Music source separation is a task to separate audio recordings into individual sources. This repository is an PyTorch implmementation of music source separation systems. Users can separate their favorite songs into different sources by installing this repository. Users can also train music source separation systems from scratch using this repository.

This repository can be also used for music source separation, speech enhancement, instruments separation, etc.

## Demos

TODO

## Installation

TODO

## Train a music source separation system from scratch

## 1. Download dataset

We use the MUSDB18 dataset to train music source separation systems. The trained system can be used to separate vocals, accompaniments, bass, and other sources. Execute the following script to download and decompress the MUSDB18 dataset:

```bash
# ./scripts/musdb18/download_musdb18_from_zenodo.sh
./scripts/download_datasets/musdb18.sh
```

The dataset looks like:
<pre>
./datasets/musdb18
├── train (100 files)
│   ├── 'A Classic Education - NightOwl.stem.mp4'
│   └── ...
├── test (50 files)
│   ├── 'Al James - Schoolboy Facination.stem.mp4'
│   └── ...
└── README.md
</pre>

## 2. Pack audio files into hdf5 files

We pack audio waveforms into hdf5 files to speed up training.
```bash
./"scripts/pack_audios_to_hdf5s/musdb18/sr=44100_chn=2.sh"
```

## 3. Create indexes for training
```bash
./scripts/create_indexes/musdb18/musdb18.sh
```

## 3. Create evaluation audios
```bash
./scripts/create_evaluation_audios/musdb18/musdb18.sh
```

## 4. Train & evaluate & save checkpoints
```bash
./scripts/train/musdb18/train.sh
```

## 5. Inference
```bash
./scripts/musdb18/inference.sh
```

## Reference

[1] Qiuqiang Kong, Yin Cao, Haohe Liu, Keunwoo Choi, Yuxuan Wang, Decoupling Magnitude and Phase Estimation with Deep ResUet for Music Source Separation, International Society for Music Information Retrieval (ISMIR), 2021.
```
@inproceedings{kong2021decoupling,
  title={Decoupling Magnitude and Phase Estimation with Deep ResUet for Music Source Separation.},
  author={Kong, Qiuqiang and Cao, Yin and Liu, Haohe and Choi, Keunwoo and Wang, Yuxuan },
  booktitle={ISMIR},
  year={2021},
  organization={Citeseer}
}
```