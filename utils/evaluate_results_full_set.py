from __future__ import print_function
import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
from enum import Enum

ABLATION_DIR = '/home/tpatten/Data/ICAS2020/Experiments/ablation'

ADDS_CODE = 'adds'
TR_CODE = 'tr'
TRXYZ_CODE = 'trxyz'

DPI = 100 # 200

def plot_ablation_architecture():
    targets = ['abf', 'bb', 'gpmf', 'gsf']
    metrics = [ADDS_CODE]
    nets = [[0, 1, 2, 3], [3, 4, 5, 6]]

    net_names = ['PointNet', 'No Pool', 'Split', 'Flat', 'w/o dropout', 'w/o aug', 'w/o sym']
    net_colors = [[0., 0., 0.],     # PointNet
                  [0.38, 0.66, 0.34],  # PointNet No Pool
                  [0.98, 0.74, 0.27],  # PointNet Split
                  [0.40, 0.53, 0.94],  # PointNet Flat
                  [0.40, 0.53, 0.94],  # w/o dropout
                  [0.40, 0.53, 0.94],  # w/o aug
                  [0.40, 0.53, 0.94],  # w/o sym
    ]
    net_styles = ['-', '-', '-', '-', '--', '-.', ':']

    for m in metrics:
        metric_vals = []
        for t in targets:
            filename = os.path.join(ABLATION_DIR, m, t + '.txt')
            vals = np.loadtxt(filename)
            metric_vals.append(vals)

        # For each set of networks to compare
        for net_set in nets:
            # For each network variation
            fig = plt.figure(figsize=(7, 5), dpi=DPI, facecolor='w', edgecolor='k')
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
                ax.plot(x, mean, linewidth=3, color=net_colors[n], linestyle=net_styles[n], label=net_names[n])
                #ax.fill_between(x, mean - std, mean + std, interpolate=True, alpha=0.2,
                #                facecolor=net_colors[n], edgecolor=net_colors[n])

            # Add legend and axes labels
            handles, labels = ax.get_legend_handles_labels()
            #plt.figlegend(handles, labels, loc='upper right', ncol=1,
            #              labelspacing=0.8, fontsize=14, bbox_to_anchor=(0.9, 0.9))
            plt.figlegend(handles, labels, loc='lower right', ncol=1,
                          labelspacing=0.8, fontsize=14, bbox_to_anchor=(0.9, 0.1))

            plt.ylabel(r'Accuracy', fontsize=16)
            if m == ADDS_CODE:
                plt.xlabel(r'ADDS threshold (\% diameter)', fontsize=16)
            else:
                plt.xlabel(r'Translation/rotation threshold (cm/degree)', fontsize=16)


def plot_ablation_hand_input():
    pass

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