import datetime
import logging
import os
import pickle
from typing import Dict, NoReturn

import librosa
import numpy as np
import yaml


def create_logging(log_dir: str, filemode: str) -> logging:
    r"""Create logging to write out log files.

    Args:
        logs_dir, str, directory to write out logs
        filemode: str, e.g., "w"

    Returns:
        logging
    """
    os.makedirs(log_dir, exist_ok=True)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, "{:04d}.log".format(i1))):
        i1 += 1

    log_path = os.path.join(log_dir, "{:04d}.log".format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
        filename=log_path,
        filemode=filemode,
    )

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    return logging


def load_audio(
    audio_path: str,
    mono: bool,
    sample_rate: float,
    offset: float = 0.0,
    duration: float = None,
) -> np.array:
    r"""Load audio.

    Args:
        audio_path: str
        mono: bool
        sample_rate: float
    """
    audio, _ = librosa.core.load(
        audio_path, sr=sample_rate, mono=mono, offset=offset, duration=duration
    )
    # (audio_samples,) | (channels_num, audio_samples)

    if audio.ndim == 1:
        audio = audio[None, :]
        # (1, audio_samples,)

    return audio


def load_random_segment(
    audio_path: str,
    random_state: int,
    segment_seconds: float,
    mono: bool,
    sample_rate: int,
) -> np.array:
    r"""Randomly select an audio segment from a recording."""

    duration = librosa.get_duration(filename=audio_path)

    start_time = random_state.uniform(0.0, duration - segment_seconds)

    audio = load_audio(
        audio_path=audio_path,
        mono=mono,
        sample_rate=sample_rate,
        offset=start_time,
        duration=segment_seconds,
    )
    # (channels_num, audio_samples)

    return audio


def float32_to_int16(x: np.float32) -> np.int16:

    x = np.clip(x, a_min=-1, a_max=1)

    return (x * 32767.0).astype(np.int16)


def int16_to_float32(x: np.int16) -> np.float32:

    return (x / 32767.0).astype(np.float32)


def read_yaml(config_yaml: str) -> Dict:
    """Read config file to dictionary.

    Args:
        config_yaml: str

    Returns:
        configs: Dict
    """
    with open(config_yaml, "r") as fr:
        configs = yaml.load(fr, Loader=yaml.FullLoader)

    return configs


def check_configs_gramma(configs: Dict) -> NoReturn:
    r"""Check if the gramma of the config dictionary for training is legal."""

    paired_input_target_data = configs['train']['paired_input_target_data']

    if paired_input_target_data is False:

        input_source_types = configs['train']['input_source_types']
        augmentation_types = configs['train']['augmentations'].keys()

        for augmentation_type in list(
            set(augmentation_types)
            & set(
                [
                    'mixaudio',
                    'pitch_shift',
                    'magnitude_scale',
                    'swap_channel',
                    'flip_axis',
                ]
            )
        ):

            augmentation_dict = configs['train']['augmentations'][augmentation_type]

            for source_type in augmentation_dict.keys():
                if source_type not in input_source_types:
                    error_msg = (
                        "The source type '{}'' in configs['train']['augmentations']['{}'] "
                        "must be one of input_source_types {}".format(
                            source_type, augmentation_type, input_source_types
                        )
                    )
                    raise Exception(error_msg)


def magnitude_to_db(x: float) -> float:
    eps = 1e-10
    return 20.0 * np.log10(max(x, eps))


def db_to_magnitude(x: float) -> float:
    return 10.0 ** (x / 20)


def get_pitch_shift_factor(shift_pitch: float) -> float:
    r"""The factor of the audio length to be scaled."""
    return 2 ** (shift_pitch / 12)


class StatisticsContainer(object):
    def __init__(self, statistics_path):
        self.statistics_path = statistics_path

        self.backup_statistics_path = "{}_{}.pkl".format(
            os.path.splitext(self.statistics_path)[0],
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )

        self.statistics_dict = {"train": [], "test": []}

    def append(self, steps, statistics, split):
        statistics["steps"] = steps
        self.statistics_dict[split].append(statistics)

    def dump(self):
        pickle.dump(self.statistics_dict, open(self.statistics_path, "wb"))
        pickle.dump(self.statistics_dict, open(self.backup_statistics_path, "wb"))
        logging.info("    Dump statistics to {}".format(self.statistics_path))
        logging.info("    Dump statistics to {}".format(self.backup_statistics_path))

    '''
    def load_state_dict(self, resume_steps):
        self.statistics_dict = pickle.load(open(self.statistics_path, "rb"))

        resume_statistics_dict = {"train": [], "test": []}

        for key in self.statistics_dict.keys():
            for statistics in self.statistics_dict[key]:
                if statistics["steps"] <= resume_steps:
                    resume_statistics_dict[key].append(statistics)

        self.statistics_dict = resume_statistics_dict
    '''


def calculate_sdr(ref: np.array, est: np.array) -> float:
    s_true = ref
    s_artif = est - ref
    sdr = 10.0 * (
        np.log10(np.clip(np.mean(s_true ** 2), 1e-8, np.inf))
        - np.log10(np.clip(np.mean(s_artif ** 2), 1e-8, np.inf))
    )
    return sdr
