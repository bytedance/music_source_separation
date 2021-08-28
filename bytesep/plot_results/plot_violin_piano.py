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

def plot_statistics(args):

    def load_sdrs(config, gpus, filename='train', source_type='vocals'):

        stat_path = './workspaces/bytesep/statistics/violin-piano/train/config={},gpus={}/statistics.pkl'.format(config, gpus)

        stat_dict = pickle.load(open(stat_path, 'rb'))
        
        median_sdrs = [e['sdr'] for e in stat_dict['test']]

        return median_sdrs

    select = args.select

    linewidth = 1
    lines = []
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ylim = 20
    expand = 1

    if select == '1a':
        sdrs = load_sdrs(config='01_violin', gpus=1)
        line, = ax.plot(sdrs, label='UNet,l1_wav', linewidth=linewidth)
        lines.append(line)

        sdrs = load_sdrs(config='01_piano', gpus=1)
        line, = ax.plot(sdrs, label='UNet,l1_wav', linewidth=linewidth)
        lines.append(line)

    elif select == '1b':
        sdrs = load_sdrs(config='02_violin', gpus=1)
        line, = ax.plot(sdrs, label='UNet,l1_wav', linewidth=linewidth)
        lines.append(line)

        sdrs = load_sdrs(config='02_piano', gpus=1)
        line, = ax.plot(sdrs, label='UNet,l1_wav', linewidth=linewidth)
        lines.append(line)

    else:
        raise Exception('Error!')
    
    max_plot_iteration = 500001
    iterations = np.arange(0, max_plot_iteration, 10000 * expand)
    ax.set_ylim(0, ylim)
    # ax.set_xlim(0, len(iterations))
    # ax.xaxis.set_ticks(np.arange(0, len(iterations), 5))
    # ax.xaxis.set_ticklabels(np.arange(0, max_plot_iteration, 50000 * expand))
    ax.yaxis.set_ticks(np.arange(ylim))
    ax.yaxis.set_ticklabels(np.arange(ylim))
    ax.grid(color='b', linestyle='solid', linewidth=0.3)
    plt.legend(handles=lines, loc=4)

    os.makedirs('results', exist_ok=True)
    save_out_path = 'results/violin_piano_sdr_{}.pdf'.format(select)
    plt.savefig(save_out_path)
    print('Save figure to {}'.format(save_out_path))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_plot = subparsers.add_parser('plot')
    parser_plot.add_argument('--select', type=str, required=True)
 
    args = parser.parse_args()

    if args.mode == 'plot':
        plot_statistics(args)

    else:
        raise Exception('Error argument!')
