from condxd import qsoutil, CondXD
from condxd.path import datpath
from astropy.io import fits
from astropy.table import Table
import os

cont_table = Table.read(os.path.join(datpath, 'JulienCat_candidates_z6.fits'))
GMM_params = {'fluxes': ['f_1p2_VIS', 'f_1p2_Y', 'f_1p2_H'],
              'fluxes_err': ['f_1p2_VIS_err', 'f_1p2_Y_err', 'f_1p2_H_err'],
              'ref_flux': 'f_1p2_J', 'ref_flux_err': 'f_1p2_J_err',
              'conditions': ['f_1p2_J',]}

# get the flux ratio and its covariance
flux_ratio = qsoutil.get_flux_ratio_from_table(cont_table, GMM_params['fluxes'], GMM_params['ref_flux'])
mag_cutoff = 23.5 # AB magnitude
flux_cutoff = qsoutil.mag_to_flux(mag_cutoff, zero_point=30.) # zero point for ERO catalog
flux_ratio_covar = qsoutil.get_flux_ratio_covar_from_table(
    cont_table, GMM_params['fluxes'], GMM_params['fluxes_err'], GMM_params['ref_flux'], flux_cutoff)

# get the conditions
conditions = qsoutil.get_conditions_from_table(cont_table, GMM_params['conditions'])