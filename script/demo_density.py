from condxd import qsoutil, CondXD
from condxd.path import datpath
from astropy.io import fits
from astropy.table import Table
import os
import numpy as np
import matplotlib.pyplot as plt
import corner
import smplotlib

""" 1. Load the data, make it suitable for CondXD """
cont_table = Table.read(os.path.join(datpath, 'JulienCat_candidates_z6.fits'))
GMM_params = {'fluxes': ['f_1p2_VIS', 'f_1p2_Y', 'f_1p2_H'],
              'fluxes_err': ['f_1p2_VIS_err', 'f_1p2_Y_err', 'f_1p2_H_err'],
              'ref_flux': 'f_1p2_J', 'ref_flux_err': 'f_1p2_J_err',
              'conditions': ['f_1p2_J',]}

# get the flux ratio and its covariance
flux_ratio = qsoutil.get_flux_ratio_from_table(cont_table, GMM_params['fluxes'], GMM_params['ref_flux'])
mag_cutoff = 23.5 # AB magnitude, the assumption of Gaussian noise is not valid for fainter sources
flux_cutoff = qsoutil.mag_to_flux(mag_cutoff, zero_point=30.) # zero point for ERO catalog
flux_ratio_covar = qsoutil.get_flux_ratio_covar_from_table(
    cont_table, GMM_params['fluxes'], GMM_params['fluxes_err'], GMM_params['ref_flux'], flux_cutoff)

# get the conditions
conditions = qsoutil.get_conditions_from_table(cont_table, GMM_params['conditions'])
conditions = np.log(conditions)

""" 2. Fit the model """
n_sample, sample_dim = flux_ratio.shape
_, conditional_dim = conditions.shape
cont_condxd = CondXD(n_Gaussians=20, sample_dim=sample_dim, conditional_dim=conditional_dim)
cont_condxd.load_data(conditions, flux_ratio, flux_ratio_covar, 
                      tra_val_tes_size=(90,10,0), batch_size=10)
cont_condxd.deconvolve(num_epoch=100)
flux_ratio_sample = cont_condxd.sample(conditions, n_per_conditional=50)
flux_ratio_sample = flux_ratio_sample.reshape(-1, sample_dim).detach().numpy()

cont_condxd.save(filename='ero_cont.pkl')


fig, ax = plt.subplots(3, 3, figsize=(8, 8))
ranges = [(-0.08,0.12), (-0.5,1.5), (-0.5,3)]
smooth = 0.6
levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.6, 0.5) ** 2)

fig = corner.corner(
    flux_ratio_sample,
    bins=50,
    range=ranges,
    labels=[
        r"$f_O/f_J$",
        r"$f_Y/f_J$",
        r"$f_H/f_J$",
        # r"$f_{W1}/f_J$",
        # r"$f_{W2}/f_J$",
    ],
    show_titles=False,
    plot_datapoints=False,
    hist_kwargs={"density": True},
    levels=levels,
    fig=fig,
    smooth=smooth,
    color='green',
    contour_kwargs={"linewidths":0.8, 'alpha':0.7},
    label_kwargs={'fontsize':20},
    data_kwargs={'alpha':0.5,},
    plot_density=False,
    no_fill_contours=True,
)

corner.overplot_points(fig, flux_ratio, color='navy', alpha=0.5, markersize=1, marker='s')
corner.overplot_hist(fig, flux_ratio, color='navy', alpha=0.8, 
                     ranges=ranges, bins=50, density=True, histtype='step', linewidth=1)

ax[0][2].text(0.9, 0.7, r'Contaminant Model', c='green',
                fontsize=25, horizontalalignment='right', weight='bold')
ax[0][2].text(0.9, 0.4, r'Contaminant Data', c='navy',
                fontsize=25, horizontalalignment='right', weight='bold')

plt.show()