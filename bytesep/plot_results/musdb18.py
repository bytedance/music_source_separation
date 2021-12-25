import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


def load_sdrs(workspace, task_name, filename, config, gpus, source_type):

    stat_path = os.path.join(
        workspace,
        "statistics",
        task_name,
        filename,
        "config={},gpus={}".format(config, gpus),
        "statistics.pkl",
    )

    stat_dict = pickle.load(open(stat_path, 'rb'))

    median_sdrs = [e['median_sdr_dict'][source_type] for e in stat_dict['test']]

    return median_sdrs


def plot_statistics(args):

    # arguments & parameters
    workspace = args.workspace
    select = args.select
    task_name = "musdb18"
    filename = "train"

    # paths
    fig_path = os.path.join('results', task_name, "sdr_{}.pdf".format(select))
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    linewidth = 1
    lines = []
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    eval_every_iterations = 10000

    if select == '1a':
        sdrs = load_sdrs(
            workspace,
            task_name,
            filename,
            config='vocals-accompaniment,resunet_subbandtime',
            gpus=2,
            source_type='vocals',
        )
        (line,) = ax.plot(sdrs, label='ResUNet_Subbandtime,vocals', linewidth=linewidth)
        lines.append(line)

        sdrs = load_sdrs(
            workspace,
            task_name,
            filename,
            config='accompaniment-vocals,resunet_subbandtime',
            gpus=2,
            source_type='accompaniment',
        )
        (line,) = ax.plot(
            sdrs, label='ResUNet_Subbandtime,accompaniment', linewidth=linewidth
        )
        lines.append(line)

        sdrs = load_sdrs(
            workspace,
            task_name,
            filename,
            config='vocals-accompaniment,mobilenet_subbandtime',
            gpus=2,
            source_type='vocals',
        )
        (line,) = ax.plot(
            sdrs, label='MobileNet_Subbandtime,vocals', linewidth=linewidth
        )
        lines.append(line)

        sdrs = load_sdrs(
            workspace,
            task_name,
            filename,
            config='accompaniment-vocals,mobilenet_subbandtime',
            gpus=2,
            source_type='accompaniment',
        )
        (line,) = ax.plot(
            sdrs, label='MobileNet_Subbandtime,accompaniment', linewidth=linewidth
        )
        lines.append(line)

        ylim = 20
        eval_every_iterations = 10000

    else:
        raise Exception('Error!')

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
    ax.set_xlabel("Steps")
    ax.set_ylabel("SDR (dB)")
    ax.yaxis.set_ticks(np.arange(ylim + 1))
    ax.yaxis.set_ticklabels(np.arange(ylim + 1))
    ax.grid(color='b', linestyle='solid', linewidth=0.3)
    plt.legend(handles=lines, loc=4)
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)

    plt.savefig(fig_path)
    print('Save figure to {}'.format(fig_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--select', type=str, required=True)

    args = parser.parse_args()

    plot_statistics(args)
