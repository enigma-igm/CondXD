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
    'path': '/home/riccardo/Software/high_z_qso_project-master/GMM_UKIDSS/data',
    # Set the input table name with the data
    'table_name': 'UKIDSS_catalog_z7_training_subsample_clean.fits',
    # Set the reference flux and its error
    'ref_flux': 'flux_J',
    'ref_flux_err': 'flux_J_err',
    # Set the fluxes for the real data
    'fluxes': ['flux_z', 'i_flux_aper_3p0', 'z_flux_aper_3p0', 'y_flux_aper_3p0', 'Y_flux_aper_3p0', 'H_flux_aper_3p0', 'K_flux_aper_3p0', 'flux_W1', 'flux_W2'],
    # Set the relative data errors
    'fluxes_err': ['flux_err_z', 'i_flux_aper_err_3p0', 'z_flux_aper_err_3p0', 'y_flux_aper_err_3p0', 'Y_flux_aper_err_3p0', 'H_flux_aper_err_3p0',
                   'K_flux_aper_err_3p0', 'flux_err_W1', 'flux_err_W2'],
}

start_time = time.time()

XD = GMM(GMM_params, hyper_params, low_SN_mag=20.5)


data_train, data_err_train, ref_train = XD.sample_splitting('training')
data_val, data_err_val, ref_val = XD.sample_splitting('validation')
data_test, data_err_test, ref_test = XD.sample_splitting('testing')

XD.gmm_fit(ref_train, data_train, data_err_train, ref_val, data_val, data_err_val, model_name='contaminants_z67')
#XD.test_sample(ref_test, data_test, data_err_test, model_name='contaminants')

elapsed_time = time.time() - start_time
print("{:.1f} s: ".format(elapsed_time))