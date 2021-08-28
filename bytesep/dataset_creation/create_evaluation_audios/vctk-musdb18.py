import argparse
import os
import pathlib
import soundfile
import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from typing import List, NoReturn

import h5py
import musdb
import librosa
import numpy as np

from bytesep.utils import float32_to_int16, load_audio


def get_random_segment(audio_path, random_state, segment_seconds, mono, sample_rate):
    duration = librosa.get_duration(filename=audio_path)

    start_time = random_state.uniform(0., duration - segment_seconds)

    audio = load_audio(audio_path=audio_path, mono=mono, sample_rate=sample_rate, offset=start_time, duration=segment_seconds)

    return audio


def create_evaluation(args):

    vctk_dataset_dir = args.vctk_dataset_dir
    musdb18_dataset_dir = args.musdb18_dataset_dir
    evaluation_audios_dir = args.evaluation_audios_dir
    sample_rate = args.sample_rate
    channels = args.channels
    evaluation_segments_num = args.evaluation_segments_num
    mono = True if channels == 1 else False

    split = 'test'
    evaluation_segments_num = 100

    random_state = np.random.RandomState(1234)

    # paths
    audios_dir = os.path.join(vctk_dataset_dir, "wav48", split)

    for source_type in ['speech', 'music', 'mixture']:
        output_dir = os.path.join(evaluation_audios_dir, split, source_type)
        os.makedirs(output_dir, exist_ok=True)

    # Get VCTK audio paths.
    speech_audio_paths = []
    speaker_ids = sorted(os.listdir(audios_dir))

    for speaker_id in speaker_ids:
        speaker_audios_dir = os.path.join(audios_dir, speaker_id)

        audio_names = sorted(os.listdir(speaker_audios_dir))

        for audio_name in audio_names:
            speaker_audio_path = os.path.join(speaker_audios_dir, audio_name)
            speech_audio_paths.append(speaker_audio_path)

    # Get Musdb18 audio paths.
    mus = musdb.DB(root=musdb18_dataset_dir, subsets=[split])
    track_indexes = np.arange(len(mus.tracks))

    for n in range(evaluation_segments_num):
        
        print('{} / {}'.format(n, evaluation_segments_num))

        # Randomly select and write out a clean speech segment.
        speech_audio_path = random_state.choice(speech_audio_paths) 

        speech_audio = load_audio(audio_path=speech_audio_path, mono=mono, sample_rate=sample_rate)
        # (channels_num, audio_samples)
        
        if channels == 2:
            speech_audio = np.tile(speech_audio, (2, 1))
        # (channels_num, audio_samples)

        output_speech_path = os.path.join(evaluation_audios_dir, split, 'speech', '{:04d}.wav'.format(n))
        soundfile.write(file=output_speech_path, data=speech_audio.T, samplerate=sample_rate)
        print("Write out to {}".format(output_speech_path))

        # Randomly select and write out a clean music segment.
        track_index = random_state.choice(track_indexes)
        track = mus[track_index]
        
        segment_samples = speech_audio.shape[1]
        start_sample = int(random_state.uniform(0., segment_samples - speech_audio.shape[1]))

        music_audio = track.audio[start_sample : start_sample + segment_samples, :].T
        # (channels_num, audio_samples)

        output_music_path = os.path.join(evaluation_audios_dir, split, 'music', '{:04d}.wav'.format(n))
        soundfile.write(file=output_music_path, data=music_audio.T, samplerate=sample_rate)
        print("Write out to {}".format(output_music_path))

        # Get and write out a mixture segment.
        mixture_audio = speech_audio + music_audio
        # (channels_num, audio_samples)

        output_mixture_path = os.path.join(evaluation_audios_dir, split, 'mixture', '{:04d}.wav'.format(n))
        soundfile.write(file=output_mixture_path, data=mixture_audio.T, samplerate=sample_rate)
        print("Write out to {}".format(output_mixture_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vctk_dataset_dir",
        type=str,
        required=True,
        help="",
    )
    parser.add_argument(
        "--musdb18_dataset_dir",
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