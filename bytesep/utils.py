import datetime
import logging
import os
import pickle

import librosa
import numpy as np
import yaml


def create_logging(log_dir, filemode):
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


def float32_to_int16(x):
    x = np.clip(x, a_min=-1, a_max=1)
    return (x * 32767.0).astype(np.int16)


def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def read_yaml(config_yaml):

    with open(config_yaml, "r") as fr:
        configs = yaml.load(fr, Loader=yaml.FullLoader)

    return configs


def check_configs_gramma(configs):

    input_source_types = configs['train']['input_source_types']

    for augmentation_type in configs['train']['augmentations'].keys():
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


def mix_audio_from_same_source(data_dict, input_sources, mixaudio):
    for source in input_sources:
        (N, segment_samples, channels_num) = data_dict[source].shape
        data_dict[source] = np.sum(
            data_dict[source].reshape(
                mixaudio, N // mixaudio, segment_samples, channels_num
            ),
            axis=0,
        )

    return data_dict


def magnitude_to_db(x):
    eps = 1e-10
    return 20.0 * np.log10(max(x, eps))


def db_to_magnitude(x):
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


def calculate_sdr(ref, est):
    s_true = ref
    s_artif = est - ref
    sdr = 10.0 * (
        np.log10(np.clip(np.mean(s_true ** 2), 1e-8, np.inf))
        - np.log10(np.clip(np.mean(s_artif ** 2), 1e-8, np.inf))
    )
    return sdr
