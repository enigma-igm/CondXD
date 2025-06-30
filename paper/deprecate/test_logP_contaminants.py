import torch

from paper.model import GMMNet
from paper.diagnostics.plots_QSO import *

import seaborn as sns

from astropy.io import fits
from astropy.table import Table, Column
from astropy import units as u
from astropy.coordinates import SkyCoord

import numpy as np

import copy
import os

from IPython import embed

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# read file
file = fits.open('data/VIKING_catalog.fits')
data = copy.deepcopy(file[1].data)
file.close()

# the RA and DEC
ra0  = data['RA']
dec0 = data['DEC']
coord0 = SkyCoord(ra=ra0*u.degree, dec=dec0*u.degree, frame='icrs')

# load reference band and error
f_J = data['f_J'].astype('float').reshape(-1, 1)
f_J_err = data['f_J_err'].astype('float').reshape(-1, 1)
# transform to tensor
f_J = torch.Tensor(f_J)
f_J_err = torch.Tensor(f_J_err)

# load relative flux
rf_z = data['f_z'] / data['f_J']
rf_Y = data['f_Y'] / data['f_J']
rf_H = data['f_H'] / data['f_J']
rf_Ks = data['f_Ks'] / data['f_J']
rf_w1 = data['f_w1'] / data['f_J']
rf_w2 = data['f_w2'] / data['f_J']
data_set = torch.Tensor(np.array([rf_z, rf_Y, rf_H, rf_Ks, rf_w1, rf_w2]))
data_set = data_set.transpose(1, 0)


# load errors
f_z_err = data['f_z_err']
f_Y_err = data['f_Y_err']
f_H_err = data['f_H_err']
f_Ks_err = data['f_Ks_err']
f_w1_err = data['f_w1_err']
f_w2_err = data['f_w2_err']
err_set  = torch.Tensor(np.array([f_z_err, f_Y_err, f_H_err, f_Ks_err, f_w1_err, f_w2_err]))
err_set  = err_set.transpose(1, 0)


# setup parameter: length of data and data dimension
len_data, D = data_set.shape
# Gaussian components
K = 20


def get_noise_covar(len_data, D, f_J, f_J_err, data_set, err_set):
    # new covariance matrix
    err_r_set = torch.zeros((len_data, D, D))
    # off-diagonal element
    high_SN_bln = ((22.5 - 2.5*torch.log10(f_J)) <= 21).flatten()
    for i in range(1, D):
        for j in range(i):
            err_r_set[high_SN_bln,i,j] = (data_set[:,i] * data_set[:,j] / f_J[:,0]**2 * f_J_err[:,0]**2)[high_SN_bln]
    err_r_set = err_r_set + err_r_set.transpose(2, 1)
    # diagonal element
    for i in range(D):
        err_r_set[:,i,i] = 1/f_J[:,0]**2 * err_set[:,i]**2 + data_set[:,i]**2 / f_J[:,0]**2 * f_J_err[:,0]**2

    return err_r_set

err_r_set = get_noise_covar(len_data, D, f_J, f_J_err, data_set, err_set)


f_J_err = f_J_err/f_J
f_J = torch.log(f_J)

del err_set



# NN initialization
gmm = GMMNet(K, D, conditional_dim=1)

gmm.load_state_dict(torch.load(f'params/params_d_K{K:d}.pkl'))



# random sample from the data set
size = 10000
#ii = np.random.choice(np.arange(len_data), size=size, replace=False)

file_sample = fits.open('data/random_contaminants.fits')
ra_sample  = copy.deepcopy(file_sample[1].data['RA'])
dec_sample = copy.deepcopy(file_sample[1].data['DEC'])
file_sample.close()

coord_sample = SkyCoord(ra=ra_sample*u.degree, dec=dec_sample*u.degree, frame='icrs')

ii, _, _  = coord_sample.match_to_catalog_sky(coord0)

data_set = data_set[ii]
err_r_set = err_r_set[ii]
ranges = np.array([(-0.5,1.),(-0.7,1.4),(-1.2,3.1),(-1.6,5.),(-2.5,8.3),(-4,12)])
in_range = ((data_set.numpy() > ranges[:,0]) & (data_set.numpy() < ranges[:,1])).all(-1)


len_J = 50
f_J = torch.linspace(0, 6, len_J).reshape(-1, 1)
log_prob_set = torch.zeros((size, len_J))
for i in range(size):
    data_i = data_set[i].reshape(1, -1).repeat(len_J, 1)
    err_r  = err_r_set[i].reshape(1, D, D).repeat(len_J, 1, 1)
    log_prob, loss = gmm.score(data_i, f_J, noise=err_r)
    log_prob_set[i] = log_prob.detach()

f_J = f2mag(torch.exp(f_J).numpy()).flatten()
log_prob_set = log_prob_set.numpy()

def edges(name):
    table = fits.open(name, memmap=True)
    table_data = Table(table[1].data)
    bin_edges = np.zeros((len(table_data['N_obj']),2))
    bin_edges[:,0] = table_data['ed_l']
    bin_edges[:,1] = table_data['ed_h']
    return bin_edges
bin_edges_star = edges(name='data/bin_edges_stars.fits')


valid_idx = [i for i in range(18)]
log_prob_set_Riccardo = np.zeros((10000, len(valid_idx)))
for i,idx in enumerate(valid_idx):
    table_name = 'data_mag_' + str(round(bin_edges_star[idx, 0], 2)) + '_' + str(round(bin_edges_star[idx, 1], 2))
    table_file = fits.open('data/probabilities/' + table_name + '_probabilities.fits')
    table_data = copy.deepcopy(table_file[1].data['logP'])
    ra_Ric  = copy.deepcopy(table_file[1].data['RA'])
    dec_Ric = copy.deepcopy(table_file[1].data['DEC'])
    table_file.close()
    log_prob_set_Riccardo[:,i] = table_data

coord_Ric = SkyCoord(ra=ra_Ric*u.degree, dec=dec_Ric*u.degree, frame='icrs')
idx, _, _  = coord_sample.match_to_catalog_sky(coord_Ric)


bin_center = bin_edges_star.mean(-1)[valid_idx]
for i in range(50):
    j = np.random.randint(10000)
    fig = plt.figure()
    p1, = plt.plot(f_J, log_prob_set[j], label='Yi')
    p2, = plt.plot(bin_center, log_prob_set_Riccardo[idx][j], label='Riccardo')
    plt.xlabel('J band magnitude', fontsize=14)
    plt.ylabel('log Probability', fontsize=14)
    customs = [p1, p2, 
              Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='k', markersize=3)]
    plt.legend(customs, [p1.get_label(), p2.get_label(), f'In plotting range:{in_range[j]}'],
            fontsize=13)
    plt.savefig(f'figs/logP_contaminant/sample_{j}')
    plt.close()
