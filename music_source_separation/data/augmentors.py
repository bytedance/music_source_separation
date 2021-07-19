from typing import Dict
import librosa
import numpy as np

# from music_source_separation.utils import magnitude_to_db, db_to_magnitude
from music_source_separation.utils import get_pitch_shift_factor

'''
class Augmentor:
    def __init__(self, random_scale_dict, random_seed=1234):
        self.random_scale_dict = random_scale_dict

        if self.random_scale_dict:
            self.lower_db = random_scale_dict['lower_db']
            self.higher_db = random_scale_dict['higher_db']
            self.random_state = np.random.RandomState(random_seed)

    def __call__(self, waveform):
        if self.random_scale_dict:
            random_scale = self.get_random_scale(waveform)
        else:
            random_scale = 1.0

        waveform *= random_scale

        return waveform

    def get_random_scale(self, waveform):
        waveform_db = magnitude_to_db(np.max(np.abs(waveform)))
        new_waveform_db = self.random_state.uniform(
            waveform_db + self.lower_db, min(waveform_db + self.higher_db, 0)
        )
        relative_db = new_waveform_db - waveform_db
        relative_scale = db_to_magnitude(relative_db)
        return relative_scale
'''


class Augmentor:
    def __init__(self, augmentation: Dict, random_seed=1234):
        r"""Augmentor for data augmentation.

        Args:
            augmentation: Dict, e.g, {
                'pitch_shift': 4,
                ...,
            }
            random_seed: int
        """
        self.augmentation = augmentation
        self.random_state = np.random.RandomState(random_seed)

    def __call__(self, waveform: np.array) -> np.array:
        r"""Augment a waveform.

        Args:
            waveform: (channels_num, original_segments_num)

        Returns:
            new_waveform: (channels_num, segments_num)
        """
        if 'pitch_shift' in self.augmentation.keys():

            max_pitch_shift = self.augmentation['pitch_shift']
            rand_pitch = self.random_state.uniform(
                low=-max_pitch_shift, high=max_pitch_shift
            )
            pitch_shift_factor = get_pitch_shift_factor(rand_pitch)
            dummy_sample_rate = 20000   # Dummy constant.

            new_waveform = librosa.resample(
                y=waveform,
                orig_sr=dummy_sample_rate,
                target_sr=dummy_sample_rate / pitch_shift_factor,
                res_type='linear',
                axis=1,
            )

        return new_waveform
