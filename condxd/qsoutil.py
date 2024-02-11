"""
TODO DMY - Functions to 
           1) handle input table (containing fluxes and redshifts) and
              convert it to a format that can be used by the condXD class
           2) compute the covariance matrix of the flux ratios
           3) TBD
"""

import numpy as np

from astropy.table import Table
from astropy.io import fits

from IPython import embed

def get_flux_ratio_covar(rel_flux, flux_err, ref_flux, flux_cutoff):
    """
    Compute the covariance matrix for flux ratios.

    Parameters:
    -----------
    rel_flux : numpy.ndarray
        Array of relative flux values with shape (N_sample, N_dim).
    flux_err : numpy.ndarray
        Array of flux error values with shape (N_sample, N_dim).
    ref_flux : numpy.ndarray
        Array of reference flux values with shape (N_sample, 1).
    flux_cutoff : float
        Flux cutoff value (e.g. 10 sigma value for ref flux).

    Returns:
    --------
    covar : numpy.ndarray
        Covariance matrix with shape (N_sample, N_dim, N_dim).
    """
    N_sample, N_dim = rel_flux.shape
    covar = np.zeros((N_sample, N_dim, N_dim))

    # 1. off-diagonal elements
    high_SNR_mask = ref_flux > flux_cutoff

    for i in range(1, N_dim):
        for j in range(i):
            covar[high_SNR_mask, i, j] = (rel_flux[high_SNR_mask, i] * rel_flux[high_SNR_mask, j] / ref_flux[high_SNR_mask, 0]**2 * flux_err[high_SNR_mask, 0]**2)

    covar = covar + covar.transpose(0, 2, 1)

    # 2. diagonal elements
    for i in range(N_dim):
        covar[:, i, i] = 1 / ref_flux[:, 0]**2 * flux_err[:, i]**2 + rel_flux[:, i]**2 / ref_flux[:, 0]**2 * flux_err[:, 0]**2

    return covar