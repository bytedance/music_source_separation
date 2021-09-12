import os
import sys
import numpy as np
import argparse
import h5py
import math
import time
import logging
import pickle
import matplotlib.pyplot as plt


def load_sdrs(workspace, task_name, filename, config, gpus):

    stat_path = os.path.join(
        workspace,
        "statistics",
        task_name,
        filename,
        "config={},gpus={}".format(config, gpus),
        "statistics.pkl",
    )

    stat_dict = pickle.load(open(stat_path, 'rb'))

    median_sdrs = [e['sdr'] for e in stat_dict['test']]

    return median_sdrs


def plot_statistics(args):

    # arguments & parameters
    workspace = args.workspace
    select = args.select
    task_name = "vctk-musdb18"
    filename = "train"

    # paths
    fig_path = os.path.join('results', task_name, "sdr_{}.pdf".format(select))
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    linewidth = 1
    lines = []
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ylim = 30
    expand = 1

    if select == '1a':
        sdrs = load_sdrs(workspace, task_name, filename, config='unet', gpus=1)
        (line,) = ax.plot(sdrs, label='UNet,l1_wav', linewidth=linewidth)
        lines.append(line)

    else:
        raise Exception('Error!')

    eval_every_iterations = 10000
    total_ticks = 50
    ticks_freq = 10

    ax.set_ylim(0, ylim)
    ax.set_xlim(0, total_ticks)
    ax.xaxis.set_ticks(np.arange(0, total_ticks + 1, ticks_freq))
    ax.xaxis.set_ticklabels(
        np.arange(
            0,
            total_ticks * eval_every_iterations + 1,
            ticks_freq * eval_every_iterations,
        )
    )
    ax.yaxis.set_ticks(np.arange(ylim + 1))
    ax.yaxis.set_ticklabels(np.arange(ylim + 1))
    ax.grid(color='b', linestyle='solid', linewidth=0.3)
    plt.legend(handles=lines, loc=4)

    plt.savefig(fig_path)
    print('Save figure to {}'.format(fig_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--select', type=str, required=True)

    args = parser.parse_args()

    plot_statistics(args)
