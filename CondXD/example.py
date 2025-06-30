import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import matplotlib.pyplot as plt
import torch

from condxd.CondXD import CondXD
from paper.data.experiment import data_load
from paper.diagnostics.plots_exp import exp_figures


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
param_path = f'paper/params/experiment/seed{seed}/'
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
cond_sample = np.random.randn(50, D_cond)
samples = condxd_example.sample(
    cond_sample, 
    n_per_conditional=10
)

# condxd.save(filename='condxd_model.pkl')