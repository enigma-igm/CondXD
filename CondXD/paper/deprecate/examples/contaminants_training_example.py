#!/usr/bin/env python

import time

from gmmnet.GMM_training import GMM

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
    'table_name': 'VIKING_catalog_clean_nobright.fits',
    # Set the reference flux and its error
    'ref_flux': 'J_flux_aper_3p0',
    'ref_flux_err': 'J_flux_aper_3p0',
    # Set the fluxes for the real data
    'fluxes': ['flux_z', 'Y_flux_aper_3p0', 'H_flux_aper_3p0', 'K_flux_aper_3p0', 'flux_w1', 'flux_w2'],
    # Set the relative data errors
    'fluxes_err': ['flux_z_err', 'Y_flux_aper_err_3p0', 'H_flux_aper_err_3p0', 'K_flux_aper_err_3p0', 'flux_w1_err', 'flux_w2_err'],
}

start_time = time.time()

XD = GMM(GMM_params, hyper_params)


data_train, data_err_train, ref_train = XD.sample_splitting('training')
data_val, data_err_val, ref_val = XD.sample_splitting('validation')
data_test, data_err_test, ref_test = XD.sample_splitting('testing')

XD.gmm_fit(ref_train, data_train, data_err_train, ref_val, data_val, data_err_val, model_name='contaminants')
#XD.test_sample(ref_test, data_test, data_err_test, model_name='contaminants')

elapsed_time = time.time() - start_time
print("{:.1f} s: ".format(elapsed_time))