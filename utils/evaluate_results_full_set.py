from __future__ import print_function
import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
from enum import Enum

ABLATION_DIR = '/home/tpatten/Data/ICAS2020/Experiments/ablation'
HAND_INPUT_DIR = '/home/tpatten/Data/ICAS2020/Experiments/hand_input'

ADDS_CODE = 'adds'
TR_CODE = 'tr'
TRXYZ_CODE = 'trxyz'

DPI = 100 # 200
FIG_SIZE = (8, 6)
LINE_WIDTH = 2

def plot_ablation_architecture():
    targets = ['abf', 'bb', 'gpmf', 'gsf']
    metrics = [ADDS_CODE]
    nets = [[0, 1, 3], [3, 2, 4, 5, 6]]

    net_names = ['baseline', 'baseline (no pool)', 'split heads', 'sorted + MLP', 'w/o dropout', 'w/o aug', 'w/o sym']
    net_colors = [[0.00, 0.00, 0.00],  # PointNet
                  [0.90, 0.60, 0.00],  # PointNet No Pool
                  [0.00, 0.60, 0.50],  # PointNet Split
                  [0.35, 0.70, 0.90],  # PointNet Flat
                  [0.95, 0.90, 0.25],  # w/o dropout
                  [0.80, 0.40, 0.00],  # w/o aug
                  [0.80, 0.60, 0.70]   # w/o sym
                 ]
    net_styles = ['-', '-', '-', '-', '-', '-', '-']

    for m in metrics:
        metric_vals = []
        for t in targets:
            filename = os.path.join(ABLATION_DIR, m, t + '.txt')
            vals = np.loadtxt(filename)
            metric_vals.append(vals)

        # For each set of networks to compare
        for net_set in nets:
            # For each network variation
            fig = plt.figure(figsize=FIG_SIZE, dpi=DPI, facecolor='w', edgecolor='k')
            ax = fig.add_subplot(111)

            for n in net_set:
                # Collect the values
                net_vals = np.zeros((metric_vals[0].shape[0], len(metric_vals)))
                for i in range(len(metric_vals)):
                    net_vals[:, i] = metric_vals[i][:, n]

                # Get the mean
                mean = np.mean(net_vals, axis=1)
                std = np.std(net_vals, axis=1)

                # Add to plot
                x = np.linspace(0, 50, mean.shape[0])
                ax.plot(x, mean, linewidth=LINE_WIDTH, color=net_colors[n], linestyle=net_styles[n], label=net_names[n])
                #ax.fill_between(x, mean - std, mean + std, interpolate=True, alpha=0.2,
                #                facecolor=net_colors[n], edgecolor=net_colors[n])

            # Add legend and axes labels
            handles, labels = ax.get_legend_handles_labels()
            #plt.figlegend(handles, labels, loc='upper right', ncol=1,
            #              labelspacing=0.8, fontsize=14, bbox_to_anchor=(0.9, 0.9))
            plt.figlegend(handles, labels, loc='lower right', ncol=1,
                          labelspacing=0.1, fontsize=16, bbox_to_anchor=(0.9, 0.1))

            plt.ylabel(r'Accuracy', fontsize=18)
            if m == ADDS_CODE:
                plt.xlabel(r'ADD threshold (\% diameter)', fontsize=18)
            else:
                plt.xlabel(r'Translation/rotation threshold (cm/degree)', fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=16)


def plot_ablation_hand_input():
    targets = ['abf']
    metrics = [ADDS_CODE]
    joints = [[0, 1, 4, 7, 10], [0, 2, 3, 5, 6, 8, 9, 11, 12]]
    # 0 all
    # 1 w/o TIPs, 2 TIPs, 3 TIPs + W
    # 4 w/o DIPs, 5 DIPs, 6 DIPs + W
    # 7 w/o PIPs, 8 PIPs, 9 PIPs + W
    # 10 w/o MCPs, 11 MCPs, 12 MCPs + W

    joint_names = ['All',
                   'w/o TIPs', 'TIPs', 'TIPs + W',
                   'w/o DIPs', 'DIPs', 'DIPs + W',
                   'w/o PIPs', 'PIPs', 'PIPs + W',
                   'w/o MCPs', 'MCPs', 'MCPs + W']
    joint_colors = [[0.00, 0.00, 0.00],  # All
                    [0.90, 0.60, 0.00],  # \TIPs
                    [0.90, 0.60, 0.00],  # TIPs
                    [0.90, 0.60, 0.00],  # TIPs + W
                    [0.00, 0.60, 0.50],  # \DIPs
                    [0.00, 0.60, 0.50],  # DIPs
                    [0.00, 0.60, 0.50],  # DIPs + W
                    [0.35, 0.70, 0.90],  # \PIPs
                    [0.35, 0.70, 0.90],  # PIPs
                    [0.35, 0.70, 0.90],  # PIPs + W
                    [0.80, 0.40, 0.00],  # \MCPs
                    [0.80, 0.40, 0.00],  # MCPs
                    [0.80, 0.40, 0.00]   # MCPs + W
                   ]
    joint_styles = ['-', '-', '--', ':', '-', '--', ':', '-', '--', ':', '-', '--', ':']

    for m in metrics:
        metric_vals = []
        for t in targets:
            filename = os.path.join(HAND_INPUT_DIR, m, t + '.txt')
            vals = np.loadtxt(filename)
            metric_vals.append(vals)

        # For each set of joints to compare
        for joint_set in joints:
            # For each network variation
            fig = plt.figure(figsize=FIG_SIZE, dpi=DPI, facecolor='w', edgecolor='k')
            ax = fig.add_subplot(111)

            for j in joint_set:
                # Collect the values
                joint_vals = np.zeros((metric_vals[0].shape[0], len(metric_vals)))
                for i in range(len(metric_vals)):
                    joint_vals[:, i] = metric_vals[i][:, j]

                # Get the mean
                mean = np.mean(joint_vals, axis=1)

                # Add to plot
                x = np.linspace(0, 50, mean.shape[0])
                ax.plot(x, mean, linewidth=LINE_WIDTH, color=joint_colors[j], linestyle=joint_styles[j],
                        label=joint_names[j])

            # Add legend and axes labels
            handles, labels = ax.get_legend_handles_labels()
            # plt.figlegend(handles, labels, loc='upper right', ncol=1,
            #              labelspacing=0.8, fontsize=14, bbox_to_anchor=(0.9, 0.9))
            plt.figlegend(handles, labels, loc='lower right', ncol=1,
                          labelspacing=0.1, fontsize=16, bbox_to_anchor=(0.9, 0.1))

            plt.ylabel(r'Accuracy', fontsize=18)
            if m == ADDS_CODE:
                plt.xlabel(r'ADD threshold (\% diameter)', fontsize=18)
            else:
                plt.xlabel(r'Translation/rotation threshold (cm/degree)', fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=16)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=int, default=0, choices=[0, 1],
                        help='type of results to plot: 0 (architectures), 1 (hand input)')
    opt = parser.parse_args()
    print(opt)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    if opt.type == 0:
        plot_ablation_architecture()
    elif opt.type == 1:
        plot_ablation_hand_input()

    plt.show()