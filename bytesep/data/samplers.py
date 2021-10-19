import pickle
from typing import Dict, List, NoReturn

import numpy as np
import torch.distributed as dist


class SegmentSampler:
    def __init__(
        self,
        indexes_path: str,
        segment_samples: int,
        mixaudio_dict: Dict,
        batch_size: int,
        steps_per_epoch: int,
        random_seed=1234,
    ):
        r"""Sample training indexes of sources.

        Args:
            indexes_path: str, path of indexes dict
            segment_samplers: int
            mixaudio_dict, dict, including hyper-parameters for mix-audio data
                augmentation, e.g., {'voclas': 2, 'accompaniment': 2}
            batch_size: int
            steps_per_epoch: int, #steps_per_epoch is called an `epoch`
            random_seed: int
        """
        self.segment_samples = segment_samples
        self.mixaudio_dict = mixaudio_dict
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

        self.meta_dict = pickle.load(open(indexes_path, "rb"))
        # E.g., {
        #     'vocals': [
        #         {'hdf5_path': 'songA.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 0, 'end_sample': 132300}，
        #         {'hdf5_path': 'songB.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 4410, 'end_sample': 445410},
        #         ...
        #     ],
        #     'accompaniment': [
        #         {'hdf5_path': 'songA.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 0, 'end_sample': 132300}，
        #         {'hdf5_path': 'songB.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 4410, 'end_sample': 445410},
        #         ...
        #     ]
        # }

        self.source_types = self.meta_dict.keys()
        # E.g., ['vocals', 'accompaniment']

        self.pointers_dict = {source_type: 0 for source_type in self.source_types}
        # E.g., {'vocals': 0, 'accompaniment': 0}

        self.indexes_dict = {
            source_type: np.arange(len(self.meta_dict[source_type]))
            for source_type in self.source_types
        }
        # E.g. {
        #     'vocals': [0, 1, ..., 225751],
        #     'accompaniment': [0, 1, ..., 225751]
        # }

        self.random_state = np.random.RandomState(random_seed)

        # Shuffle indexes.
        for source_type in self.source_types:
            self.random_state.shuffle(self.indexes_dict[source_type])
            print("{}: {}".format(source_type, len(self.indexes_dict[source_type])))

    def __iter__(self) -> List[Dict]:
        r"""Yield a batch of meta info.

        Returns:
            batch_meta_list: (batch_size,) e.g., when mix-audio is 2, looks like [
                {'vocals': [
                    {'hdf5_path': 'songA.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 13406400, 'end_sample': 13538700},
                    {'hdf5_path': 'songB.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 4440870, 'end_sample': 4573170}]
                'accompaniment': [
                    {'hdf5_path': 'songE.h5', 'key_in_hdf5': 'accompaniment', 'begin_sample': 14579460, 'end_sample': 14711760},
                    {'hdf5_path': 'songF.h5', 'key_in_hdf5': 'accompaniment', 'begin_sample': 3995460, 'end_sample': 4127760}]
                }
                ...
            ]
        """
        batch_size = self.batch_size

        while True:
            batch_meta_dict = {source_type: [] for source_type in self.source_types}

            for source_type in self.source_types:
                # E.g., ['vocals', 'accompaniment']

                # Loop until get a mini-batch.
                while len(batch_meta_dict[source_type]) != batch_size:

                    largest_index = (
                        len(self.indexes_dict[source_type])
                        - self.mixaudio_dict[source_type]
                    )
                    # E.g., 225750 = 225752 - 2

                    if self.pointers_dict[source_type] > largest_index:

                        # Reset pointer, and shuffle indexes.
                        self.pointers_dict[source_type] = 0
                        self.random_state.shuffle(self.indexes_dict[source_type])

                    source_metas = []
                    mix_audios_num = self.mixaudio_dict[source_type]

                    for _ in range(mix_audios_num):

                        pointer = self.pointers_dict[source_type]
                        # E.g., 1

                        index = self.indexes_dict[source_type][pointer]
                        # E.g., 12231

                        self.pointers_dict[source_type] += 1

                        source_meta = self.meta_dict[source_type][index]
                        # E.g., {
                        #     'hdf5_path': 'xx/song_A.h5',
                        #     'key_in_hdf5': 'vocals',
                        #     'begin_sample': 13406400,
                        #     'end_sample': 13538700,
                        # }

                        source_meta['end_sample'] = (
                            source_meta['begin_sample'] + self.segment_samples
                        )

                        source_metas.append(source_meta)

                    batch_meta_dict[source_type].append(source_metas)
            # When mix-audio is 2, batch_meta_dict looks like: {
            #     'vocals': [
            #         [{'hdf5_path': 'songA.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 13406400, 'end_sample': 13538700},
            #          {'hdf5_path': 'songB.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 4440870, 'end_sample': 4573170}],
            #         [{'hdf5_path': 'songC.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 1186290, 'end_sample': 1318590},
            #          {'hdf5_path': 'songD.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 8462790, 'end_sample': 8595090}]
            #     ]
            #     'accompaniment': [
            #         [{'hdf5_path': 'songE.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 24232950, 'end_sample': 24365250},
            #          {'hdf5_path': 'songF.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 1569960, 'end_sample': 1702260}],
            #         [{'hdf5_path': 'songG.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 2795940, 'end_sample': 2928240},
            #          {'hdf5_path': 'songH.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 10923570, 'end_sample': 11055870}]
            #     ]
            # }

            batch_meta_list = [
                {
                    source_type: batch_meta_dict[source_type][i]
                    for source_type in self.source_types
                }
                for i in range(batch_size)
            ]
            # When mix-audio is 2, batch_meta_list looks like: [
            #     {'vocals': [
            #         {'hdf5_path': 'songA.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 13406400, 'end_sample': 13538700},
            #         {'hdf5_path': 'songB.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 4440870, 'end_sample': 4573170}]
            #      'accompaniment': [
            #         {'hdf5_path': 'songE.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 14579460, 'end_sample': 14711760},
            #         {'hdf5_path': 'songF.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 3995460, 'end_sample': 4127760}]
            #     }
            #     ...
            # ]

            yield batch_meta_list

    def __len__(self) -> int:
        return self.steps_per_epoch

    def state_dict(self) -> Dict:
        state = {'pointers_dict': self.pointers_dict, 'indexes_dict': self.indexes_dict}
        return state

    def load_state_dict(self, state) -> NoReturn:
        self.pointers_dict = state['pointers_dict']
        self.indexes_dict = state['indexes_dict']


class DistributedSamplerWrapper:
    def __init__(self, sampler):
        r"""Distributed wrapper of sampler."""
        self.sampler = sampler

    def __iter__(self):
        num_replicas = dist.get_world_size()
        rank = dist.get_rank()

        for indices in self.sampler:
            yield indices[rank::num_replicas]

    def __len__(self) -> int:
        return len(self.sampler)
