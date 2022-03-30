#!/usr/bin/env python

import os

import numpy as np

from scipy.optimize import curve_fit
from scipy import interpolate

from astropy.table import Table

import torch

from gmmnet.GMM_training import GMM
from gmmnet.GMM_plots import plot_mag_distribution_data

from models import model

from IPython import embed

def _create_prob_folder():
    # Chek the existence of the probabilities folders and creates it in case it is missing, so to store the plots
    path = os.getcwd()
    if os.path.isdir('probabilities'):
        print("Directory probabilities already exists")
    else:
        print("Creating the directory: probabilities")
        os.mkdir(path + '/probabilities')

def interpolate_number_counts(ref_mag, prior_mag, area, bins=10, range=[17, 23], x_incom=[20, 20.5], ylimits=[1e-1,5e3],
                              name='data'):
    """ Fit the number count distribution using a broken power-law

        Args:
            ref_mag (nparray): this is the mag array used to evaluate the prior curve
            prior_mag (nparray): this is the mag array used to compute the priors
            area: area of the survey used to normalize
            bins (int): number of bins to compute the number counts
            range (list, shape nx2): the interpolation range
            x_incom (list, shape nx2): the range from where to derive the incompleteness
            ylimits (list, shape nx2): the range to plot in the y-axis
            name (string): the name of the plot of the distribution
        """
    hist, bin_edges = np.histogram(ref_mag, bins=bins,range=range)
    bin_mp = (bin_edges[1:] + bin_edges[:-1]) / 2

    # Define the power-law function that extrapolate the incompleteness
    def pl(x, b, a):
        y = []
        for xx in x:
            if (xx > np.log10(x_incom[0])):
                y.append(b + xx * a)
            else:
                y.append(0)
        return y

    hist_pl = hist[(bin_mp <= x_incom[1])] / area
    bin_mp_pl = bin_mp[(bin_mp <= x_incom[1])]
    log_bin_mp = np.log10(bin_mp_pl)
    log_hist = np.log10(hist_pl)
    hist_err = np.log(10) / np.sqrt(hist_pl * area)
    popt, pcov = curve_fit(pl, log_bin_mp, log_hist, sigma=hist_err)

    # Here it computes the incompleteness curve
    spl = interpolate.interp1d(bin_mp, hist/area, kind='cubic', fill_value="extrapolate")

    # Plot the distribution
    plot_mag_distribution_data(bin_mp, hist, area, popt, x_incom[0], spl, range, ylimits=ylimits, extension_name=name)

    prior_prob = np.array([_prior(mag, popt, spl, x_incom[0]) for mag in prior_mag])

    return prior_prob

def log_prob_computation(GMM_params, hyper_params, model_name, table_name, overwrite=True):
    """ Compute the probability density of the provided sample wrt the specified model

        :param GMM_params:
        :param hyper_params:
        :param table_name:
        :param overwrite:
        """

    _create_prob_folder()

    # Initialize the NN and sample the noisy and noiseless data
    XD = GMM(GMM_params, hyper_params)

    best_model = model.GMMNet(XD.n_gauss, XD.rel_flux.shape[1], conditional_dim=1)
    best_model.load_state_dict(torch.load('XD_fit_models/{}.pkl'.format(model_name)))
    best_model.eval()
    log_prob_dens = best_model.log_prob_b(XD.rel_flux, XD.ref_f, noise=XD.rel_flux_err).detach().numpy()

    # Save the computed probabilities
    idx = np.linspace(1, len(log_prob_dens), len(log_prob_dens))
    catalog = np.vstack((idx, np.transpose(log_prob_dens)))
    dat = Table(np.transpose(catalog), names=['idx', 'logP'])
    dat.write('probabilities/{}_probabilities.fits'.format(table_name), format='fits', overwrite=overwrite)

    return log_prob_dens

def _prior(x, popt, spl, xc):
    """ Compute the priors of the distribution

    :param x:
    :param popt:
    :param spl:
    :param xc:
    """
    if (x > xc):
        return np.power(10, popt[0]) * x ** popt[1]
    else:
        return spl(x)