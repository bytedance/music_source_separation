import numpy as np

from music_source_separation.utils import magnitude_to_db, db_to_magnitude


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
            random_scale = 1.

        waveform *= random_scale

        return waveform

    def get_random_scale(self, waveform):
        waveform_db = magnitude_to_db(np.max(np.abs(waveform)))
        new_waveform_db = self.random_state.uniform(waveform_db + self.lower_db, min(waveform_db + self.higher_db, 0))
        relative_db = new_waveform_db - waveform_db
        relative_scale = db_to_magnitude(relative_db)
        return relative_scale