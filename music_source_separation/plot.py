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

    def load_sdrs(config, gpus, filename='train'):

        # stat_path = '/home/tiger/workspaces/byte_separation/statistics/{}/config={},gpus={}/statistics.pkl'.format(filename, config, gpus)
        stat_path = './workspaces/music_source_separation/statistics/musdb18/train/config={},gpus={}/statistics.pkl'.format(config, gpus)

        stat_dict = pickle.load(open(stat_path, 'rb'))
        
        median_sdrs = [e['median_sdr_dict']['vocals'] for e in stat_dict['test']]

        return median_sdrs

    def load_sdrs_multi(config, gpus, filename='train'):

        stat_path = './workspaces/music_source_separation/statistics/musdb18/{}/config={},gpus={}/statistics.pkl'.format(filename, config, gpus)

        stat_dict = pickle.load(open(stat_path, 'rb'))

        vocal_sdrs = [e['median_sdr_dict']['vocals'] for e in stat_dict['test']]
        bass_sdrs = [e['median_sdr_dict']['bass'] for e in stat_dict['test']]
        drums_sdrs = [e['median_sdr_dict']['drums'] for e in stat_dict['test']]
        other_sdrs = [e['median_sdr_dict']['other'] for e in stat_dict['test']]

        return vocal_sdrs, bass_sdrs, drums_sdrs, other_sdrs

    select = args.select

    linewidth = 1
    lines = []
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ylim = 15
    expand = 1

    if select == '1a':
        sdrs = load_sdrs(config='01', gpus=2)
        line, = ax.plot(sdrs, label='UNet,l1_wav', linewidth=linewidth)
        lines.append(line)

    elif select == '2a':
        sdrs = load_sdrs(config='resnet143_vocals_ismir2021', gpus=2)
        line, = ax.plot(sdrs, label='resnet143_vocals_ismir2021,l1_wav', linewidth=linewidth)
        lines.append(line)

        sdrs = load_sdrs(config='resnet143_decouple_plus_vocals', gpus=2)
        line, = ax.plot(sdrs, label='resnet143_decouple_plus_vocals,l1_wav', linewidth=linewidth)
        lines.append(line)

    elif select == '2b':
        vocal_sdrs, bass_sdrs, drums_sdrs, other_sdrs = load_sdrs_multi(config='resnet143_decouple_plus_4outputs', gpus=2)
        line, = ax.plot(vocal_sdrs, label='vocals,resnet143_decouple_plus_4outputs,l1_wav', linewidth=linewidth)
        lines.append(line)
        line, = ax.plot(bass_sdrs, label='bass,resnet143_decouple_plus_4outputs,l1_wav', linewidth=linewidth)
        lines.append(line)
        line, = ax.plot(drums_sdrs, label='drums,resnet143_decouple_plus_4outputs,l1_wav', linewidth=linewidth)
        lines.append(line)
        line, = ax.plot(other_sdrs, label='other,resnet143_decouple_plus_4outputs,l1_wav', linewidth=linewidth)
        lines.append(line)

    else:
        raise Exception('Error!')
    
    max_plot_iteration = 500001
    iterations = np.arange(0, max_plot_iteration, 10000 * expand)
    ax.set_ylim(0, ylim)
    ax.set_xlim(0, len(iterations))
    ax.xaxis.set_ticks(np.arange(0, len(iterations), 5))
    ax.xaxis.set_ticklabels(np.arange(0, max_plot_iteration, 50000 * expand))
    ax.yaxis.set_ticks(np.arange(ylim))
    ax.yaxis.set_ticklabels(np.arange(ylim))
    ax.grid(color='b', linestyle='solid', linewidth=0.3)
    plt.legend(handles=lines, loc=4)

    os.makedirs('results', exist_ok=True)
    save_out_path = 'results/sdr_{}.pdf'.format(select)
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
