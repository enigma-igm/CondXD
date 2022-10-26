#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import os
import numpy as np

import corner

from IPython import embed


def create_fig_folder():
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
    fig = corner.corner(real_data, labels=labels, label_kwargs={"fontsize": 25}, bins=bins, range=ranges,
                        quiet=True, levels=(0.382, 0.682, 0.866,))
    corner.corner(test_data, fig=fig, color='r', labels=labels, label_kwargs={"fontsize": 25}, bins=bins,
                  range=ranges, alpha=0.7, quiet=True, levels=(0.382, 0.682, 0.866,))
    axes = np.array(fig.axes).reshape((real_data.shape[-1], real_data.shape[-1]))
    for ax in fig.get_axes():
        ax.tick_params(axis='both', which='major', direction='in', length=5, labelsize=15, width=2)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
    axes[0, -3].text(0.6, 0.4, 'Real data', c='k', fontsize=25, horizontalalignment='center', weight='bold')
    axes[1, -3].text(0.6, 1.0, legend, c='r', fontsize=25, horizontalalignment='center', weight='bold')
    fig.savefig('figs/{}_relative_flux.png'.format(name))
    plt.close()


def make_gif(name, save_name):
    # Create a gif of the relative flux contours
    import imageio
    folder = 'figs/'
    files = [folder + f'{_}_relative_flux.png' for _ in name]
    images = [imageio.imread(file) for file in files]
    imageio.mimwrite(folder + save_name + '.gif', images, fps=1.5)


def plot_mag_distribution_data(bin_mp, hist, area, popt, xc, spl, limits, ylimits, extension_name='data'):
    """ Plot the mag distribution

        Args:
            bin_mp (narray): the bin mid points of the distribution
            hist (narray): the number counts
            area (float): the area of the survey
            popt (list, shape nx2): the parameters of the power-law that accounts for the incompleteness
            xc (float): the incompleteness magnitude value
            spl (spline function): the spline used to interpolate the distribution
            limits (list, shape nx2): the ranges to plot
            ylimits (list, shape nx2): the range to plot in the y-axis
            extension_name (string): the specific name of the plot
        """
    create_fig_folder()

    def plaw(x, popt, spl, xc):
        if (x > xc):
            return np.power(10, popt[0]) * x ** popt[1]
        else:
            return spl(x)

    fig = plt.figure(num=None, figsize=(15, 10))
    ax = plt.subplot(1, 1, 1)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='minor', direction='in', length=5, labelsize=30, width=4)
    ax.tick_params(axis='both', which='major', direction='in', length=10, labelsize=30, width=4)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.set_ylabel("N [deg$^{-2}$]", fontsize=35)
    ax.set_xlabel("$J$ [mag]", fontsize=35)
    ax.errorbar(bin_mp, hist / area, yerr=np.sqrt(hist) / area, c='k', marker='o', markersize=10, lw=0)
    x_inter = np.linspace(min(bin_mp), max(bin_mp), 100)
    powerl = np.array([plaw(x_inter[i], popt, spl, xc) for i in range(len(x_inter))])
    spline = np.array([spl(x_inter[i]) for i in range(len(x_inter))])
    ax.plot(x_inter, powerl, 'r-', markersize=5, linewidth=4, label=extension_name)
    ax.set_xlim(limits)
    ax.set_ylim(ylimits)
    ax.legend(loc='upper left', fontsize='35')
    plt.tight_layout()
    fig.savefig('figs/' + extension_name + '_mag_distribution.pdf')
    plt.close()


def plot_prob_hist(prob, bins=100, label='', extension_name='data'):
    """ Plot the total probability distribution

        Args:ax.pl
            prob (narray): the total probability
            bins (int): the number of bins to use in the histogram
            extension_name (string): the label to show in the histogram
            extension_name (string): the specific name of the plot
        """
    fig = plt.figure(num=None, figsize=(15, 10))
    ax = plt.subplot(1, 1, 1)
    ax.tick_params(axis='both', which='minor', direction='in', length=5, labelsize=30, width=4)
    ax.tick_params(axis='both', which='major', direction='in', length=10, labelsize=30, width=4)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    ax.set_ylabel("N", fontsize=35)
    ax.set_xlabel("P", fontsize=35)
    ax.hist(prob, bins=bins, range=[0, 1], color='k', histtype='step', lw=3, label=label)
    ax.set_xlim([-0.1, 1.1])
    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize='35')
    plt.tight_layout()
    fig.savefig('figs/' + extension_name + '_prob_hist.pdf')
    plt.close()

def plot_prob_red_dist(prob, z=[], index=[0], zrange=[6,8], zstep=0.01):
    """ Plot the probability density vs redshift

        Args:
            bin_mp (narray): the bin mid points of the distribution
            hist (narray): the number counts
            area (float): the area of the survey
            popt (list, shape nx2): the parameters of the power-law that accounts for the incompleteness
            xc (float): the incompleteness magnitude value
            spl (spline function): the spline used to interpolate the distribution
            limits (list, shape nx2): the ranges to plot
            ylimits (list, shape nx2): the range to plot in the y-axis
            extension_name (string): the specific name of the plot
        """
    create_fig_folder()

    redshift = np.arange(zrange[0], zrange[1] + zstep, zstep)

    for i, idx in enumerate(index):
        fig = plt.figure(num=None, figsize=(15, 10))
        ax = plt.subplot(1, 1, 1)
        #ax.set_yscale('log')
        ax.tick_params(axis='both', which='minor', direction='in', length=5, labelsize=30, width=4)
        ax.tick_params(axis='both', which='major', direction='in', length=10, labelsize=30, width=4)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.set_ylabel("ln(P(z))", fontsize=35)
        ax.set_xlabel("z", fontsize=35)

        if len(z) != 0:
            ax.axvline(x=z[i], linewidth=4, color='r', linestyle='--')

        ax.plot(redshift, np.log(prob[i]), 'k-', markersize=5, linewidth=4, label=str(i))
        ax.set_xlim(zrange[0]-0.1, zrange[1]+0.1)
        #ax.set_ylim(ylimits)
        ax.legend(loc='upper left', fontsize='35')
        plt.tight_layout()
        fig.savefig('figs/' + str(i) + '_prob_red_distribution.png')
        plt.close()