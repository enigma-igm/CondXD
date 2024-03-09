import numpy as np

from astropy.table import Table
from astropy.io import fits

from IPython import embed

def table2array(table):
    return np.lib.recfunctions.structured_to_unstructured(np.array(table))

def flux_to_mag(flux, zero_point):
    """
    Convert flux to magnitude

    Useful Zero Points:
    ---------
    nanoMaggies: 22.5
    AB (jansky): 8.9
    AB (erg/s/cm^2/Hz): 48.6
    Euclid ERO: 30.0 for NISP, 30.132 for VIS
    """
    return zero_point - 2.5 * np.log10(flux)

def mag_to_flux(mag, zero_point):
    """
    Convert magnitude to flux

    Useful Zero Points:
    ---------
    nanoMaggies: 22.5
    AB (jansky): 8.9
    AB (erg/s/cm^2/Hz): 48.6
    Euclid ERO: 30.0 for NISP, 30.132 for VIS
    """
    return 10**((zero_point - mag) / 2.5)

def get_flux_ratio(flux, ref_flux):
    """
    Calculate the flux ratio of each flux value in the given list of fluxes
    with respect to the reference flux.

    Parameters:
    -----------
    flux : numpy.ndarray
        Array of flux values with shape (N_sample, N_dim).
    ref_flux : numpy.ndarray
        Reference flux (N_sample, 1)

    Returns:
    --------
    flux_ratio: numpy.ndarray
        Array of flux ratio with shape (N_sample, N_dim).
    """
    ref_flux = ref_flux.reshape(-1, 1)
    flux_ratio = flux / ref_flux
    return flux_ratio

def get_flux_ratio_from_table(table, flux_keyword, ref_keyword):
    """
    Calculate the flux ratio of each flux value in the given list of fluxes
    with respect to the reference flux. Values are taken from the given table.

    Parameters:
    -----------
    table : astropy.table.Table
        Table containing the flux values.
    flux_keyword : str
        Keyword for the flux values in the table.
    ref_keyword : str
        Keyword for the reference flux values in the table.

    Returns:
    --------
    flux_ratio: numpy.ndarray
        Array of flux ratio with shape (N_sample, N_dim).
    """
    flux = table2array(table[flux_keyword])
    ref_flux = table2array(table[[ref_keyword]])
    flux_ratio = get_flux_ratio(flux, ref_flux)
    return flux_ratio

def get_conditions_from_table(table, condition_keyword):
    """
    Retrieve conditions from a table based on a given condition keyword.

    Parameters
    ----------
    table : astropy.table.Table
        The table containing the conditions.

    condition_keyword : list of str
        The keyword(s) used to filter the conditions.

    Returns
    -------
    conditions : numpy.ndarray
        Array of conditions with shape (N_sample, N_dim).
    """
    conditions = table2array(table[condition_keyword])
    return conditions

def get_flux_ratio_covar(flux, flux_err, ref_flux, ref_fluxerr, flux_cutoff):
    """
    Compute the covariance matrix for flux ratios.

    Parameters:
    -----------
    flux : numpy.ndarray
        Array of flux values with shape (N_sample, N_dim).
    flux_err : numpy.ndarray
        Array of flux error values with shape (N_sample, N_dim).
    ref_flux : numpy.ndarray
        Array of reference flux values with shape (N_sample, 1).
    ref_fluxerr : numpy.ndarray
        Array of reference flux error values with shape (N_sample, 1).
    flux_cutoff : float
        Flux cutoff value (e.g. 10 sigma value for ref flux).

    Returns:
    --------
    covar : numpy.ndarray
        Covariance matrix with shape (N_sample, N_dim, N_dim).
    """
    flux_ratio = get_flux_ratio(flux, ref_flux)
    N_sample, N_dim = flux_ratio.shape
    covar = np.zeros((N_sample, N_dim, N_dim))
    ref_flux = ref_flux.reshape(-1, 1)
    ref_fluxerr = ref_fluxerr.reshape(-1, 1)

    # 1. off-diagonal elements
    high_SNR_mask = ref_flux > flux_cutoff
    high_SNR_mask = high_SNR_mask.reshape(-1)

    for i in range(1, N_dim):
        for j in range(i):
            covar[high_SNR_mask, i, j] = (flux_ratio[high_SNR_mask, i] * flux_ratio[high_SNR_mask, j] / 
                                          ref_flux[high_SNR_mask][:,0]**2 * ref_fluxerr[high_SNR_mask][:,0]**2)

    covar = covar + covar.transpose(0, 2, 1)

    # 2. diagonal elements
    for i in range(N_dim):
        covar[:, i, i] = (flux_err[:, i] / ref_flux[:,0])**2 + flux_ratio[:, i]**2 * (ref_fluxerr[:,0] / ref_flux[:,0])**2

    return covar

def get_flux_ratio_covar_from_table(table, flux_keyword, fluxerr_keyword, ref_keyword, referr_keyword, flux_cutoff):
    """
    Compute the covariance matrix for flux ratios from a given table.

    Parameters:
    -----------
    table : astropy.table.Table
        Table containing the flux values.
    flux_keyword : str
        Keyword for the flux values in the table.
    fluxerr_keyword : str
        Keyword for the flux error values in the table.
    ref_keyword : str
        Keyword for the reference flux values in the table.
    referr_keyword : str
        Keyword for the reference flux error values in the table.
    flux_cutoff : float
        Flux cutoff value (e.g. 10 sigma value for ref flux).

    Returns:
    --------
    covar : numpy.ndarray
        Covariance matrix with shape (N_sample, N_dim, N_dim).
    """
    flux = table2array(table[flux_keyword])
    fluxerr = table2array(table[fluxerr_keyword])
    ref_flux = table2array(table[[ref_keyword]])
    ref_fluxerr = table2array(table[[referr_keyword]])

    covar = get_flux_ratio_covar(flux, fluxerr, ref_flux, ref_fluxerr, flux_cutoff)
    return covar