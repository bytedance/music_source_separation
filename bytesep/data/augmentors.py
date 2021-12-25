from typing import Dict

import librosa
import numpy as np

from bytesep.utils import db_to_magnitude, get_pitch_shift_factor, magnitude_to_db


class Augmentor:
    def __init__(self, augmentations: Dict, random_seed=1234):
        r"""Augmentor for augmenting one segment.

        Args:
            augmentations: Dict, e.g, {
                'mixaudio': {'vocals': 2, 'accompaniment': 2}
                'pitch_shift': {'vocals': 4, 'accompaniment': 4},
                ...,
            }
            random_seed: int
        """
        self.augmentations = augmentations
        self.random_state = np.random.RandomState(random_seed)

    def __call__(self, waveform: np.array, source_type: str) -> np.array:
        r"""Augment a waveform.

        Args:
            waveform: (input_channels, audio_samples)
            source_type: str

        Returns:
            new_waveform: (input_channels, new_audio_samples)
        """
        if 'pitch_shift' in self.augmentations.keys():
            waveform = self.pitch_shift(waveform, source_type)

        if 'magnitude_scale' in self.augmentations.keys():
            waveform = self.magnitude_scale(waveform, source_type)

        if 'swap_channel' in self.augmentations.keys():
            waveform = self.swap_channel(waveform, source_type)

        if 'flip_axis' in self.augmentations.keys():
            waveform = self.flip_axis(waveform, source_type)

        return waveform

    def pitch_shift(self, waveform: np.array, source_type: str) -> np.array:
        r"""Shift the pitch of a waveform. We use resampling for fast pitch
        shifting, so the speed of the waveform will also be changed. The length
        of the returned waveform will be changed.

        Args:
            waveform: (input_channels, audio_samples)
            source_type: str

        Returns:
            new_waveform: (input_channels, new_audio_samples)
        """

        # maximum pitch shift in semitones
        max_pitch_shift = self.augmentations['pitch_shift'][source_type]

        if max_pitch_shift == 0:  # No pitch shift augmentations.
            return waveform

        # random pitch shift
        rand_pitch = self.random_state.uniform(
            low=-max_pitch_shift, high=max_pitch_shift
        )

        # We use librosa.resample instead of librosa.effects.pitch_shift
        # because it is 10x times faster.
        pitch_shift_factor = get_pitch_shift_factor(rand_pitch)
        dummy_sample_rate = 10000  # Dummy constant.

        input_channels = waveform.shape[0]

        if input_channels == 1:
            waveform = np.squeeze(waveform)

        new_waveform = librosa.resample(
            y=waveform,
            orig_sr=dummy_sample_rate,
            target_sr=dummy_sample_rate / pitch_shift_factor,
            res_type='linear',
            axis=-1,
        )

        if input_channels == 1:
            new_waveform = new_waveform[None, :]

        return new_waveform

    def magnitude_scale(self, waveform: np.array, source_type: str) -> np.array:
        r"""Scale the magnitude of a waveform.

        Args:
            waveform: (input_channels, audio_samples)
            source_type: str

        Returns:
            new_waveform: (input_channels, audio_samples)
        """
        lower_db = self.augmentations['magnitude_scale'][source_type]['lower_db']
        higher_db = self.augmentations['magnitude_scale'][source_type]['higher_db']

        if lower_db == 0 and higher_db == 0:  # No magnitude scale augmentation.
            return waveform

        # The magnitude (in dB) of the sample with the maximum value.
        waveform_db = magnitude_to_db(np.max(np.abs(waveform)))

        new_waveform_db = self.random_state.uniform(
            waveform_db + lower_db, waveform_db + higher_db
        )

        relative_db = new_waveform_db - waveform_db

        relative_scale = db_to_magnitude(relative_db)

        new_waveform = waveform * relative_scale

        return new_waveform

    def swap_channel(self, waveform: np.array, source_type: str) -> np.array:
        r"""Randomly swap channels.

        Args:
            waveform: (input_channels, audio_samples)
            source_type: str

        Returns:
            new_waveform: (input_channels, audio_samples)
        """
        ndim = waveform.shape[0]

        if ndim == 1:
            return waveform
        else:
            random_axes = self.random_state.permutation(ndim)
            return waveform[random_axes, :]

    def flip_axis(self, waveform: np.array, source_type: str) -> np.array:
        r"""Randomly flip the waveform along x-axis.

        Args:
            waveform: (input_channels, audio_samples)
            source_type: str

        Returns:
            new_waveform: (input_channels, audio_samples)
        """
        ndim = waveform.shape[0]
        random_values = self.random_state.choice([-1, 1], size=ndim)

        return waveform * random_values[:, None]
