#!/usr/bin/env python

import time

import numpy as np

from astropy.io import fits
from astropy.table import Table

from gmmnet.GMM_training import GMM
from gmmnet.GMM_plots import cornerplots, make_gif

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

# Define the ploting parameters
labels = ['$f_z/f_J$','$f_Y/f_J$', '$f_H/f_J$','$f_{Ks}/f_J$','$f_{W1}/f_J$','$f_{W2}/f_J$']
bins = 50
ranges = [(-0.5,1.),(-0.7,1.4),(-1.2,3.1),(-1.6,5.),(-2.5,8.3),(-4,12)]

hdu_list = fits.open(GMM_params['path']+'/'+GMM_params['table_name'], memmap=True)
output = Table(hdu_list[1].data)
J_mag = output[GMM_params['ref_flux']].astype('float').reshape(-1, 1)

# Initialize the NN and sample the noisy and noiseless data
XD = GMM(GMM_params, hyper_params)

real_data, noiseless_fluxes, noisy_fluxes = XD.sample_data('contaminants')

# Produce and save the corner plots with the comparison between the real and the deconvolved/convolved data
J_mag_bin = np.linspace(17,22.3, 54)
name_fig=[]
embed()
for i in range(1,len(J_mag_bin)):
    idx = np.where((J_mag <= J_mag_bin[i]) & (J_mag > J_mag_bin[i-1]))
    cornerplots(real_data[idx], noiseless_fluxes[idx], labels, bins, ranges, legend='Deconvolved data',
            name='deconvolved_contaminants_{}_J_{}'.format(str(round(J_mag_bin[i-1],2)),str(round(J_mag_bin[i],2))))
    name_fig.append('deconvolved_contaminants_{}_J_{}'.format(str(round(J_mag_bin[i-1],2)),str(round(J_mag_bin[i],2))))

# Produce a gif with all the figures produces above
make_gif(name_fig, 'deconvolved_contaminants')

elapsed_time = time.time() - start_time
print("{:.1f} s: ".format(elapsed_time))