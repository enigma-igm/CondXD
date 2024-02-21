import torch
from torch.utils.data import DataLoader, TensorDataset

from data.toy import *
from deprecate.model import *
from diagnostics.toy import *

import seaborn as sns

from astropy.io import fits

from sklearn.cluster import KMeans

import copy
import os

from IPython import embed

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# read file
file = fits.open('../../4Master/Research1/VIKING_catalog.fits')
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

# kmeans to classify each data point. The means0 serves as the origin of the means.
#kmeans_t = KMeans(n_clusters=K, random_state=0).fit(data_tes.numpy())
#means0_t = kmeans_t.cluster_centers_
means0_t = np.zeros((K, D))

# NN initialization
gmm = GMMNet(K, D, 1, torch.FloatTensor(means0_t))

gmm.load_state_dict(torch.load(f'params/params_d_K{K:d}.pkl'))

# test process
gmm.eval()
tes_loss = 0
for i, (f_J_i, data_i, err_r_i) in enumerate(train_loader_tes):
    log_prob_b, loss = gmm.fit(data_i, f_J_i, noise=err_r_i)

    tes_loss += -log_prob_b.sum().item()
tes_loss = tes_loss / size_tes
print('\nTest loss:', tes_loss)


all_figures(K, D, train_loader_tes, gmm)

"""
#All figures to show the performances of our network.

# sampling from trained model with noise
f_J_tes = torch.Tensor([])
data_tes = torch.Tensor([])
output_tes = torch.Tensor([])
for i, (f_J_i, data_i, err_r_i) in enumerate(train_loader_tes):
    f_J_tes = torch.cat((f_J_tes, f_J_i))
    data_tes = torch.cat((data_tes, data_i))
    output_tes = torch.cat((output_tes, gmm.sample(f_J_i, 1, err_r_i)))
data_tes = data_tes.numpy()
output_tes = output_tes.reshape(-1, D).numpy()

# the corner plot of the relative fluxes
name = f'd_K{K:d}'
cornerplots(data_tes, output_tes, labels, bins, ranges, '', name, noisy=True)

# the corner plot of the relative fluxes in each J band flux bin
for i in range(Jbin_len):
    bln = (torch.exp(f_J_tes)>=Jbin_l[i]) & (torch.exp(f_J_tes)<Jbin_r[i])
    bln = bln.numpy().flatten()
    data_tes_i = data_tes[bln]
    output_tes_i = output_tes[bln]
    anno = f'{Jbin_l[i]:.1f}<$f_J$<{Jbin_r[i]:.1f}'
    name = f'd_J{i:d}_K{K:d}'
    cornerplots(data_tes_i, output_tes_i, labels, bins, ranges, anno, name, noisy=True)

tag='noisy'
save_name = f'noisy_relative_f_d_K{K:d}'
make_gif(Jbin_len, K, tag, save_name)
plt.close()


# shuffle the test set
f_J_tes = torch.Tensor([])
data_tes = torch.Tensor([])
# sampling from trained model without noise
output_tes = torch.Tensor([])
for i, (f_J_i, data_i, _) in enumerate(train_loader_tes):
    f_J_tes = torch.cat((f_J_tes, f_J_i))
    data_tes = torch.cat((data_tes, data_i))
    output_tes = torch.cat((output_tes, gmm.sample(f_J_i, 1)))
data_tes = data_tes.numpy()
output_tes = output_tes.reshape(-1, D).numpy()

# the corner plot of the relative fluxes
name = f'd_K{K:d}'
cornerplots(data_tes, output_tes, labels, bins, ranges, '', name, noisy=False)
# the corner plot of the relative fluxes in each J band flux bin
for i in range(Jbin_len):
    bln = (np.exp(f_J_tes)>=Jbin_l[i]) & (np.exp(f_J_tes)<Jbin_r[i])
    bln = bln.numpy().flatten()
    data_tes_i = data_tes[bln]
    output_tes_i = output_tes[bln]
    anno = f'{Jbin_l[i]:.1f}<$f_J$<{Jbin_r[i]:.1f}'
    name = f'd_J{i:d}_K{K:d}'
    cornerplots(data_tes_i, output_tes_i, labels, bins, ranges, anno, name, noisy=False)


tag='clean'
save_name = f'clean_relative_f_d_K{K:d}'
make_gif(Jbin_len, K, tag, save_name)
plt.close()

embed()
"""

