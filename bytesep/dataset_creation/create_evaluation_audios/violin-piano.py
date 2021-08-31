import argparse
import os
import pathlib
import soundfile
import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from typing import List, NoReturn

import h5py
import librosa
import numpy as np

from bytesep.utils import float32_to_int16, load_audio
from bytesep.dataset_creation.pack_audios_to_hdf5s.instruments_solo import read_csv as read_instruments_solo_csv
from bytesep.dataset_creation.pack_audios_to_hdf5s.maestro import read_csv as read_maestro_csv


def get_random_segment(audio_path, random_state, segment_seconds, mono, sample_rate):
    duration = librosa.get_duration(filename=audio_path)

    start_time = random_state.uniform(0., duration - segment_seconds)

    audio = load_audio(audio_path=audio_path, mono=mono, sample_rate=sample_rate, offset=start_time, duration=segment_seconds)

    return audio


def create_evaluation(args):

    # arguments & parameters
    violin_dataset_dir = args.violin_dataset_dir
    piano_dataset_dir = args.piano_dataset_dir
    evaluation_audios_dir = args.evaluation_audios_dir
    sample_rate = args.sample_rate
    channels = args.channels
    evaluation_segments_num = args.evaluation_segments_num
    mono = True if channels == 1 else False

    split = 'test'
    segment_seconds = 10.

    random_state = np.random.RandomState(1234)

    violin_meta_csv = os.path.join(violin_dataset_dir, 'validation.csv')
    violin_names_dict = read_instruments_solo_csv(violin_meta_csv)
    violin_audio_names = violin_names_dict['{}'.format(split)]

    piano_meta_csv = os.path.join(piano_dataset_dir, 'maestro-v2.0.0.csv')
    piano_names_dict = read_maestro_csv(piano_meta_csv)
    piano_audio_names = piano_names_dict['{}'.format(split)]

    for source_type in ['violin', 'piano', 'mixture']:
        output_dir = os.path.join(evaluation_audios_dir, split, source_type)
        os.makedirs(output_dir, exist_ok=True)

    for n in range(evaluation_segments_num):
        
        print('{} / {}'.format(n, evaluation_segments_num))

        violin_audio_name = random_state.choice(violin_audio_names)
        violin_audio_path = os.path.join(violin_dataset_dir, "mp3s", violin_audio_name)
        
        violin_audio = get_random_segment(
            audio_path=violin_audio_path, 
            random_state=random_state, 
            segment_seconds=segment_seconds, 
            mono=mono, 
            sample_rate=sample_rate
        )

        output_violin_path = os.path.join(evaluation_audios_dir, split, 'violin', '{:04d}.wav'.format(n))
        soundfile.write(file=output_violin_path, data=violin_audio.T, samplerate=sample_rate)
        print("Write out to {}".format(output_violin_path))

        #
        piano_audio_name = random_state.choice(piano_audio_names)
        piano_audio_path = os.path.join(piano_dataset_dir, piano_audio_name)

        piano_audio = get_random_segment(
            audio_path=piano_audio_path, 
            random_state=random_state, 
            segment_seconds=segment_seconds, 
            mono=mono, 
            sample_rate=sample_rate
        )

        output_piano_path = os.path.join(evaluation_audios_dir, split, 'piano', '{:04d}.wav'.format(n))
        soundfile.write(file=output_piano_path, data=piano_audio.T, samplerate=sample_rate)
        print("Write out to {}".format(output_piano_path))

        #
        mixture_audio = violin_audio + piano_audio
        output_mixture_path = os.path.join(evaluation_audios_dir, split, 'mixture', '{:04d}.wav'.format(n))
        soundfile.write(file=output_mixture_path, data=mixture_audio.T, samplerate=sample_rate)
        print("Write out to {}".format(output_mixture_path))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--violin_dataset_dir",
        type=str,
        required=True,
        help="",
    )
    parser.add_argument(
        "--piano_dataset_dir",
        type=str,
        required=True,
        help="",
    )
    parser.add_argument(
        "--evaluation_audios_dir",
        type=str,
        required=True,
        help="",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        required=True,
        help="",
    )
    parser.add_argument(
        "--channels",
        type=int,
        required=True,
        help="",
    )
    parser.add_argument(
        "--evaluation_segments_num",
        type=int,
        required=True,
        help="",
    )

    # Parse arguments.
    args = parser.parse_args()

    create_evaluation(args)