import torch
from torch.utils.data import DataLoader, TensorDataset

from deprecate.model import *
from diagnostics.plots_QSO import *

import seaborn as sns

from astropy.io import fits

import numpy as np

import copy
import os


os.environ['KMP_DUPLICATE_LIB_OK']='True'

# read file
file = fits.open('data/VIKING_catalog.fits')
data = copy.deepcopy(file[1].data)
file.close()

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

del data

# setup parameter: length of data and data dimension
len_data, D = data_set.shape
# Gaussian components
K = 20
# size of training / validation /test set
size_tra = 0
size_val = 0
size_tes = 100


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


# divide to training and validation set
def real_size(size_tra, size_val, size_tes, len_data):
    sum_size = size_tra + size_val + size_tes
    real_size_tra = np.round(len_data*size_tra/sum_size).astype('int')
    real_size_val = np.round(len_data*size_val/sum_size).astype('int')
    real_size_tes = np.round(len_data*size_tes/sum_size).astype('int')
    return (real_size_tra, real_size_val, real_size_tes)
def get_set(data_set, err_r_set, id_sep, asign):
    data  = data_set[id_sep==asign]
    err_r = err_r_set[id_sep==asign]
    return (data, err_r)


size_tra, size_val, size_tes = real_size(size_tra, size_val, size_tes, len_data)
id_sep = np.append(np.ones(size_tra), np.append(np.ones(size_val)*2, np.ones(size_tes)*3)).astype('int')
np.random.seed()
np.random.shuffle(id_sep)
f_J_tes, f_J_err_tes = get_set(f_J, f_J_err, id_sep, 3)
data_tes, err_r_tes = get_set(data_set, err_r_set, id_sep, 3)

del data_set, err_set

# put data into batches
batch_size = 500
train_loader_tes = DataLoader(TensorDataset(f_J_tes, data_tes, err_r_tes), batch_size=batch_size, shuffle=False)


# NN initialization
gmm = GMMNet(K, D, conditional_dim=1)

gmm.load_state_dict(torch.load(f'params/params_d_K{K:d}.pkl'))

# test process
gmm.eval()
tes_loss = 0
for i, (f_J_i, data_i, err_r_i) in enumerate(train_loader_tes):
    log_prob_b, loss = gmm.score(data_i, f_J_i, noise=err_r_i)

    tes_loss += -log_prob_b.sum().item()
tes_loss = tes_loss / size_tes
print('\nTest loss:', tes_loss)


all_figures(K, D, train_loader_tes, gmm)