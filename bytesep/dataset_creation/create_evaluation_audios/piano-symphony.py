import argparse
import os
from typing import NoReturn

import librosa
import numpy as np
import soundfile

from bytesep.dataset_creation.pack_audios_to_hdf5s.instruments_solo import (
    read_csv as read_instruments_solo_csv,
)
from bytesep.dataset_creation.pack_audios_to_hdf5s.maestro import (
    read_csv as read_maestro_csv,
)
from bytesep.utils import load_random_segment


def create_evaluation(args) -> NoReturn:
    r"""Random mix and write out audios for evaluation.

    Args:
        piano_dataset_dir: str, the directory of the piano dataset
        symphony_dataset_dir: str, the directory of the symphony dataset
        evaluation_audios_dir: str, the directory to write out randomly selected and mixed audio segments
        sample_rate: int
        channels: int, e.g., 1 | 2
        evaluation_segments_num: int
        mono: bool

    Returns:
        NoReturn
    """

    # arguments & parameters
    piano_dataset_dir = args.piano_dataset_dir
    symphony_dataset_dir = args.symphony_dataset_dir
    evaluation_audios_dir = args.evaluation_audios_dir
    sample_rate = args.sample_rate
    channels = args.channels
    evaluation_segments_num = args.evaluation_segments_num
    mono = True if channels == 1 else False

    split = 'test'
    segment_seconds = 10.0

    random_state = np.random.RandomState(1234)

    piano_meta_csv = os.path.join(piano_dataset_dir, 'maestro-v2.0.0.csv')
    piano_names_dict = read_maestro_csv(piano_meta_csv)
    piano_audio_names = piano_names_dict[split]

    symphony_meta_csv = os.path.join(symphony_dataset_dir, 'validation.csv')
    symphony_names_dict = read_instruments_solo_csv(symphony_meta_csv)
    symphony_audio_names = symphony_names_dict[split]

    for source_type in ['piano', 'symphony', 'mixture']:
        output_dir = os.path.join(evaluation_audios_dir, split, source_type)
        os.makedirs(output_dir, exist_ok=True)

    for n in range(evaluation_segments_num):

        print('{} / {}'.format(n, evaluation_segments_num))

        # Randomly select and write out a clean piano segment.
        piano_audio_name = random_state.choice(piano_audio_names)
        piano_audio_path = os.path.join(piano_dataset_dir, piano_audio_name)

        piano_audio = load_random_segment(
            audio_path=piano_audio_path,
            random_state=random_state,
            segment_seconds=segment_seconds,
            mono=mono,
            sample_rate=sample_rate,
        )

        output_piano_path = os.path.join(
            evaluation_audios_dir, split, 'piano', '{:04d}.wav'.format(n)
        )
        soundfile.write(
            file=output_piano_path, data=piano_audio.T, samplerate=sample_rate
        )
        print("Write out to {}".format(output_piano_path))

        # Randomly select and write out a clean symphony segment.
        symphony_audio_name = random_state.choice(symphony_audio_names)
        symphony_audio_path = os.path.join(
            symphony_dataset_dir, "mp3s", symphony_audio_name
        )

        symphony_audio = load_random_segment(
            audio_path=symphony_audio_path,
            random_state=random_state,
            segment_seconds=segment_seconds,
            mono=mono,
            sample_rate=sample_rate,
        )

        output_symphony_path = os.path.join(
            evaluation_audios_dir, split, 'symphony', '{:04d}.wav'.format(n)
        )
        soundfile.write(
            file=output_symphony_path, data=symphony_audio.T, samplerate=sample_rate
        )
        print("Write out to {}".format(output_symphony_path))

        # Mix piano and symphony segments and write out a mixture segment.
        mixture_audio = symphony_audio + piano_audio
        output_mixture_path = os.path.join(
            evaluation_audios_dir, split, 'mixture', '{:04d}.wav'.format(n)
        )
        soundfile.write(
            file=output_mixture_path, data=mixture_audio.T, samplerate=sample_rate
        )
        print("Write out to {}".format(output_mixture_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--piano_dataset_dir",
        type=str,
        required=True,
        help="The directory of the piano dataset.",
    )
    parser.add_argument(
        "--symphony_dataset_dir",
        type=str,
        required=True,
        help="The directory of the symphony dataset.",
    )
    parser.add_argument(
        "--evaluation_audios_dir",
        type=str,
        required=True,
        help="The directory to write out randomly selected and mixed audio segments.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        required=True,
        help="Sample rate.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        required=True,
        help="Audio channels, e.g, 1 or 2.",
    )
    parser.add_argument(
        "--evaluation_segments_num",
        type=int,
        required=True,
        help="The number of segments to create for evaluation.",
    )

    # Parse arguments.
    args = parser.parse_args()

    create_evaluation(args)
