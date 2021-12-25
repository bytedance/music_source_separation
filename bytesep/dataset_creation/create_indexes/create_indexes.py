import argparse
import os
import pickle
from typing import NoReturn

import h5py

from bytesep.utils import read_yaml


def create_indexes(args) -> NoReturn:
    r"""Create and write out training indexes into disk. The indexes may contain
    information from multiple datasets. During training, training indexes will
    be shuffled and iterated for selecting segments to be mixed. E.g., the
    training indexes_dict looks like: {
        'vocals': [
            {'hdf5_path': '.../songA.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 0}
            {'hdf5_path': '.../songB.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 4410}
            ...
        ]
        'accompaniment': [
            {'hdf5_path': '.../songA.h5', 'key_in_hdf5': 'accompaniment', 'begin_sample': 0}
            {'hdf5_path': '.../songB.h5', 'key_in_hdf5': 'accompaniment', 'begin_sample': 4410}
            ...
        ]
    }
    """

    # Arugments & parameters
    workspace = args.workspace
    config_yaml = args.config_yaml

    # Only create indexes for training, because evalution is on entire pieces.
    split = "train"

    # Read config file.
    configs = read_yaml(config_yaml)

    sample_rate = configs["sample_rate"]
    segment_samples = int(configs["segment_seconds"] * sample_rate)

    # Path to write out index.
    indexes_path = os.path.join(workspace, configs[split]["indexes"])
    os.makedirs(os.path.dirname(indexes_path), exist_ok=True)

    source_types = configs[split]["source_types"].keys()
    # E.g., ['vocals', 'accompaniment']

    indexes_dict = {source_type: [] for source_type in source_types}
    # E.g., indexes_dict will looks like: {
    #     'vocals': [
    #         {'hdf5_path': '.../songA.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 0}
    #         {'hdf5_path': '.../songB.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 4410}
    #         ...
    #     ]
    #     'accompaniment': [
    #         {'hdf5_path': '.../songA.h5', 'key_in_hdf5': 'accompaniment', 'begin_sample': 0}
    #         {'hdf5_path': '.../songB.h5', 'key_in_hdf5': 'accompaniment', 'begin_sample': 4410}
    #         ...
    #     ]
    # }

    # Get training indexes for each source type.
    for source_type in source_types:
        # E.g., ['vocals', 'bass', ...]

        print("--- {} ---".format(source_type))

        dataset_types = configs[split]["source_types"][source_type]
        # E.g., ['musdb18', ...]

        # Each source can come from mulitple datasets.
        for dataset_type in dataset_types:

            hdf5s_dir = os.path.join(
                workspace, dataset_types[dataset_type]["hdf5s_directory"]
            )

            hop_samples = int(dataset_types[dataset_type]["hop_seconds"] * sample_rate)

            key_in_hdf5 = dataset_types[dataset_type]["key_in_hdf5"]
            # E.g., 'vocals'

            hdf5_names = sorted(os.listdir(hdf5s_dir))
            print("Hdf5 files num: {}".format(len(hdf5_names)))

            count = 0

            # Traverse all packed hdf5 files of a dataset.
            for n, hdf5_name in enumerate(hdf5_names):

                print(n, hdf5_name)
                hdf5_path = os.path.join(hdf5s_dir, hdf5_name)

                with h5py.File(hdf5_path, "r") as hf:

                    bgn_sample = 0
                    while bgn_sample + segment_samples < hf[key_in_hdf5].shape[-1]:
                        meta = {
                            'hdf5_path': hdf5_path,
                            'key_in_hdf5': key_in_hdf5,
                            'begin_sample': bgn_sample,
                        }
                        indexes_dict[source_type].append(meta)

                        bgn_sample += hop_samples
                        count += 1

                    # If the audio length is shorter than the segment length,
                    # then use the entire audio as a segment.
                    if bgn_sample == 0:
                        meta = {
                            'hdf5_path': hdf5_path,
                            'key_in_hdf5': key_in_hdf5,
                            'begin_sample': 0,
                        }
                        indexes_dict[source_type].append(meta)

            print("{} indexes: {}".format(dataset_type, count))

        print(
            "Total indexes for {}: {}".format(
                source_type, len(indexes_dict[source_type])
            )
        )

    pickle.dump(indexes_dict, open(indexes_path, "wb"))
    print("Write index dict to {}".format(indexes_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workspace", type=str, required=True, help="Directory of workspace."
    )
    parser.add_argument(
        "--config_yaml", type=str, required=True, help="User defined config file."
    )

    # Parse arguments.
    args = parser.parse_args()

    # Create training indexes.
    create_indexes(args)
