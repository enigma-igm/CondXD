import os
import copy
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import matplotlib.pyplot as plt
import torch

from condxd.CondXD import CondXD
from paper.data.experiment import data_load
from diagnostics.plots_exp import exp_figures


# components, dimension of data, and dimension of conditional
K, D, D_cond = 10, 7, 1

# size of training sample and validation sample
N = 100000

# load training and validation data and parameters
# set up random seed
seed = 15

# load training and validation data without normalization
# noise is noise matrix
cond, _, _, _, data, noise, _ = data_load(
    N, K, D, D_cond,
    noisy=True, seed0=seed
)


# initialization the deconvolution model
K_GMM = K
batch_size = 250

# NN parameter (weights) save directory
param_path = f'params/experiment/seed{seed}/'
if not os.path.exists(param_path):
    os.mkdir(param_path)
    print("Directory ", param_path , " created ")

# start
condxd_example = CondXD(
    n_Gaussians=K_GMM, 
    sample_dim=D, 
    conditional_dim=D_cond
)

# load data into the condxd_example
condxd_example.load_data(
    cond, data, noise,
    tra_val_tes_size=(9, 1, 0),
    batch_size=batch_size
)

# deconvolve the data
condxd_example.deconvolve(num_epoch=100)

# sample from the deconvolved distribution
samples = condxd_example.sample(cond, n_per_conditional=50)

# condxd.save_NN(filename='condxd_weights.pkl')



# visualization of the results
# conditioning variable bins for KL divergence plot
n_bin = 10
cond_bin_edges_l = np.linspace(0, 1, num=n_bin, endpoint=False)
cond_bin_edges_r = np.linspace(0, 1, num=n_bin, endpoint=False) + 1/n_bin
cond_bin_edges = np.array([cond_bin_edges_l, cond_bin_edges_r]).transpose()

fig_path = f'figs/experiment/seed{seed}/'
if not os.path.exists(fig_path):
    os.mkdir(fig_path)
    print("Directory " , fig_path ,  " Created ")

KL_div, cond = exp_figures(
    D_cond, K, D, 
    condxd_example.train_loss_list, 
    condxd_example.valid_loss_list,
    condxd_example, 
    condxd_example.data_avg, 
    condxd_example.data_std,
    cond_bin_edges, 
    fig_path, 
    seed,
    save_sample=True
)