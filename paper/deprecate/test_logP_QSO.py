import torch

# from CondXD.model import *
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
file = fits.open('data/noiseless_QSO_test.fits')
data = copy.deepcopy(file[1].data)
file.close()



# load reference band and error
f_J = data['VISTA-VISTA-J'].astype('float').reshape(-1, 1)
# transform to tensor
f_J = torch.Tensor(f_J)

# load relative flux
rf_z = data['VISTA-VISTA-Z'] / data['VISTA-VISTA-J']
rf_Y = data['VISTA-VISTA-Y'] / data['VISTA-VISTA-J']
rf_H = data['VISTA-VISTA-H'] / data['VISTA-VISTA-J']
rf_Ks = data['VISTA-VISTA-K'] / data['VISTA-VISTA-J']
rf_w1 = data['WISE-unWISE-W1'] / data['VISTA-VISTA-J']
rf_w2 = data['WISE-unWISE-W2'] / data['VISTA-VISTA-J']
data_set = torch.Tensor(np.array([rf_z, rf_Y, rf_H, rf_Ks, rf_w1, rf_w2]))
data_set = data_set.transpose(1, 0)

del data

# setup parameter: length of data and data dimension
size_data, D = data_set.shape
# Gaussian components
K = 20


# manual noise covariance
err_r_set = torch.zeros((size_data, D, D))
err_r_set[:, torch.eye(D).to(torch.bool)] = 0.001


# transformation of f_J
f_J = torch.log(f_J)



# NN initialization
gmm = GMMNet(K, D, conditional_dim=1)
gmm.load_state_dict(torch.load(f'params/params_d_K{K:d}.pkl'))


# plotting range
ranges = np.array([(-0.5,1.),(-0.7,1.4),(-1.2,3.1),(-1.6,5.),(-2.5,8.3),(-4,12)])
in_range = ((data_set.numpy() > ranges[:,0]) & (data_set.numpy() < ranges[:,1])).all(-1)


# measure possibility
len_J = 50
f_J = torch.linspace(0, 6, len_J).reshape(-1, 1)
log_prob_set = torch.zeros((size_data, len_J))
for i in range(size_data):
    data_i = data_set[i].reshape(1, -1).repeat(len_J, 1)
    err_r  = err_r_set[i].reshape(1, D, D).repeat(len_J, 1, 1)
    log_prob, loss = gmm.score(data_i, f_J, noise=err_r)
    log_prob_set[i] = log_prob.detach()


# customize
f_J = f2mag(torch.exp(f_J).numpy()).flatten()
log_prob_set = log_prob_set.numpy()


# load bin edges
def edges(name):
    table = fits.open(name, memmap=True)
    table_data = Table(table[1].data)
    bin_edges = np.zeros((len(table_data['N_obj']),2))
    bin_edges[:,0] = table_data['ed_l']
    bin_edges[:,1] = table_data['ed_h']
    return bin_edges
bin_edges_star = edges(name='data/bin_edges_stars.fits')


# load Riccardo's possibilities
valid_idx = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 17, 19]
log_prob_set_Riccardo = np.zeros((size_data, len(valid_idx)))
for i,idx in enumerate(valid_idx):
    table_name = 'data_mag_' + str(round(bin_edges_star[idx, 0], 2)) + '_' + str(round(bin_edges_star[idx, 1], 2))
    table_file = fits.open('data/probabilities2/' + table_name + '_probabilities.fits')
    table_data = copy.deepcopy(table_file[1].data['logP'])
    table_file.close()
    log_prob_set_Riccardo[:,i] = table_data


# plots
bin_center = bin_edges_star.mean(-1)[valid_idx]
for i in range(50):
    j = np.random.randint(size_data)
    fig = plt.figure()
    p1, = plt.plot(f_J, log_prob_set[j], label='Yi')
    p2, = plt.plot(bin_center, log_prob_set_Riccardo[j], label='Riccardo')
    plt.xlabel('J band magnitude', fontsize=14)
    plt.ylabel('log Probability', fontsize=14)
    customs = [p1, p2, 
              Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='k', markersize=3)]
    plt.legend(customs, [p1.get_label(), p2.get_label(), f'In plotting range:{in_range[j]}'],
            fontsize=13)
    #plt.legend(fontsize=14)
    plt.savefig(f'figs/logP_QSO/sample_{j}')
    plt.close()
