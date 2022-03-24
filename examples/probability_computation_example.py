#!/usr/bin/env python

import time

import numpy as np

from astropy.io import fits
from astropy.table import Table

from gmmnet.GMM_probabilities import log_prob_computation, interpolate_number_counts

from IPython import embed

# Set the hyperparameters
hyper_params = {
    "learning_rate": 1e-3,
    "batch_size": 500,
    "schedule_factor": 0.4,
    "patience": 2,
    "num_epoch": 100,
    "weight_decay": 0.001,
    "size_training": 80,
    "size_validation": 20,
    "size_testing": 0,
    "n_gauss": 20
}

GMM_params = {
    # Set the folder path to read the input table
    'path': 'data',
    # Set the input table name with the data
    'table_name': 'VIKING_catalog.fits',
    # Set the reference flux and its error
    'ref_flux': 'f_J',
    'ref_flux_err': 'f_J_err',
    # Set the fluxes for the real data
    'fluxes': ['f_z', 'f_Y', 'f_H', 'f_Ks', 'f_w1', 'f_w2'],
    # Set the relative data errors
    'fluxes_err': ['f_z_err', 'f_Y_err', 'f_H_err', 'f_Ks_err', 'f_w1_err', 'f_w2_err'],
}

start_time = time.time()

#logp = log_prob_computation(GMM_params, hyper_params, '20G_contaminants', 'contaminants')

hdu_list = fits.open(GMM_params['path']+'/'+GMM_params['table_name'], memmap=True)
output = Table(hdu_list[1].data)
ref_f=output[GMM_params['ref_flux']]
ref_mag = 22.5 - 2.5 * np.log10(ref_f)

prior_cont = interpolate_number_counts(ref_mag, area=1076, bins=25, range=[17,22.5], x_incom = [20.8, 21.5],
                                       name='contaminants')

elapsed_time = time.time() - start_time
print("{:.1f} s: ".format(elapsed_time))