#!/usr/bin/env python

import matplotlib.pyplot as plt

import os
import numpy as np

import corner

def _create_fig_folder():
    # Chek the existence of the fig folders and creates it in case it is missing, so to store the plots
    path = os.getcwd()
    if os.path.isdir('figs'):
        print("Directory figs already exists")
    else:
        print("Creating the directory: figs")
        os.mkdir(path + '/figs')

def cornerplots(real_data, test_data, labels, bins, ranges, legend, name):
    """Making comparison corner plots of the test set.

    Args:
        real_data (narray): the real noisy data set.
        test_data (narray): the NN's sampled set (it could be noisy or noisless).
        labels (list): labels of each band (dimension).
        bins (int): the bins of the 2Dhist.
        ranges (list, shape nx2): the ranges of each band for plotting.
        legend (string): legend of the sampled set on the plot.
        name (string): the specific name of the plot of relative fluxes.
    """
    _create_fig_folder()
    fig = corner.corner(real_data, labels=labels, label_kwargs={"fontsize": 25}, bins=bins, range=ranges)
    corner.corner(test_data, fig=fig, color='r', labels=labels, label_kwargs={"fontsize": 25}, bins=bins,
                  range=ranges, alpha=0.7)
    axes = np.array(fig.axes).reshape((real_data.shape[-1], real_data.shape[-1]))
    for ax in fig.get_axes():
        ax.tick_params(axis='both', which='major', direction='in', length=5, labelsize=15, width=2)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
    axes[0, -3].text(0.6, 0.4, 'Real data', c='k', fontsize=25, horizontalalignment='center', weight='bold')
    axes[1, -3].text(0.6, 1.0, legend, c='r', fontsize=25, horizontalalignment='center', weight='bold')
    fig.savefig('figs/{}_relative_flux.png'.format(name))
    

def make_gif(name, save_name):
    # Create a gif of the relative flux contours
    import imageio
    folder='figs/'
    files = [folder+f'{_}_relative_flux.png' for _ in name]
    images = [imageio.imread(file) for file in files]
    imageio.mimwrite(folder+save_name+'.gif', images, fps=1.5)