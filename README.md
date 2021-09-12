# Music Source Separation

Music source separation is a task to separate audio recordings into individual sources. This repository is an PyTorch implmementation of music source separation. Users can separate their favorite songs into different sources by installing this repository. In addition, users can train their own music source separation systems using this repository. This repository also includes speech enhancement, instruments separation, etc.

## Demos

Vocals and accompaniment separation: https://www.youtube.com/watch?v=WH4m5HYzHsg

## Installation

```bash
pip install -r requirements.txt
```

## Separation

Users can easily separate their favorite audio recordings into vocals and accompaniment using the pretrained checkpoints.

## Download checkpoints
```bash
./separate_scripts/download_checkpoints.sh
```

## Separate vocals and accompaniment
```bash
./separate_scripts/separate_vocals.sh "resources/vocals_accompaniment_10s.mp3" "sep_vocals.mp3"
./separate_scripts/separate_accompaniment.sh "resources/vocals_accompaniment_10s.mp3" "sep_accompaniment.mp3"
```

## Train a music source separation system from scratch

## 1. Download dataset

We use the MUSDB18 dataset to train music source separation systems. The trained system can be used to separate vocals, accompaniments, bass, and other sources. Execute the following script to download and decompress the MUSDB18 dataset:

```bash
./scripts/0_download_datasets/musdb18.sh
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
."/scripts/1_pack_audios_to_hdf5s/musdb18/sr=44100,chn=2.sh"
```

## 3. Create indexes for training
```bash
./scripts/2_create_indexes/musdb18/create_indexes.sh
```

## 3. Create evaluation audios
```bash
./scripts/3_create_evaluation_audios/musdb18/create_evaluation_audios.sh
```

## 4. Train & evaluate & save checkpoints
```bash
./scripts/4_train/musdb18/train.sh
```

## 5. Inference
```bash
./scripts/5_inference/musdb18/inference.sh
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