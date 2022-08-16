import torch
from torch.utils.data import DataLoader, TensorDataset

from models.model import GMMNet
from diagnostics.plots_QSO import all_figures

from astropy.io import fits
import numpy as np

import copy
import os

from IPython import embed

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#experiment = Experiment(
#    api_key="3OL9Cr63OABpmGEDoiLeXFzhj",
#    project_name="GMMnet",
#    workspace="protesticon",
#)

hyper_params = {
    "learning_rate": 1e-3,
    "batch_size": 500,
    "schedule_factor": 0.4,
    "patience": 2,
    "num_epoch": 100,
    "weight_decay": 0.001,
    "size_tra": 70,
    "size_val": 15,
    "size_tes": 15,
    "num_Gauss": 20
}

#experiment.log_parameters(hyper_params)

# read file
file = fits.open('data/VIKING_catalog.fits')
data = copy.deepcopy(file[1].data)
file.close()

# load reference band and error
f_J = data['f_J'].astype('float').reshape(-1, 1)
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
K = hyper_params['num_Gauss']
# size of training / validation / test set
size_tra = hyper_params['size_tra']
size_val = hyper_params['size_val']
size_tes = hyper_params['size_tes']


def get_noise_covar(len_data, D, f_J, f_J_err, data_set, err_set):
    """Generating the covariant noisy matrix of the relative fluxes.

    Args:
        len_data (int): length, or size of the data.
        D (int): number of dimensions of the data.
        f_J (tensor): 1D tensor of J band fluxes.
        f_J_err (tensor): error of the J band fluxes.
        data_set (tensor): dataset of relative fluxes, shape (len_data, D).
        err_set ([type]): error of original fluxes, shape(len_data, D).

    Returns:
        tensor: covariant noisy matrix of the relative fluxes.
    """
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


# log to prevent exploding
f_J_err = f_J_err/f_J
f_J = torch.log(f_J)


# divide to training and validation set
def real_size(size_tra, size_val, size_tes, len_data):
    sum_size = size_tra + size_val + size_tes
    real_size_tra = np.round(len_data*size_tra/sum_size).astype('int')
    real_size_val = np.round(len_data*size_val/sum_size).astype('int')
    real_size_tes = np.round(len_data*size_tes/sum_size).astype('int')
    if (real_size_tra+real_size_val+real_size_tes !=len_data):
        delta = (real_size_tra+real_size_val+real_size_tes - len_data).astype('int')
        real_size_tes -= delta
    return (real_size_tra, real_size_val, real_size_tes)
def get_set(data_set, err_r_set, id_sep, asign):
    data  = data_set[id_sep==asign]
    err_r = err_r_set[id_sep==asign]
    return (data, err_r)

size_tra, size_val, size_tes = real_size(size_tra, size_val, size_tes, len_data)
id_sep = np.append(np.ones(size_tra), np.append(np.ones(size_val)*2, np.ones(size_tes)*3)).astype('int')
np.random.seed()
np.random.shuffle(id_sep)

f_J_tra, f_J_err_tra = get_set(f_J, f_J_err, id_sep, 1)
f_J_val, f_J_err_val = get_set(f_J, f_J_err, id_sep, 2)
f_J_tes, f_J_err_tes = get_set(f_J, f_J_err, id_sep, 3)
data_tra, err_r_tra = get_set(data_set, err_r_set, id_sep, 1)
data_val, err_r_val = get_set(data_set, err_r_set, id_sep, 2)
data_tes, err_r_tes = get_set(data_set, err_r_set, id_sep, 3)

del data_set, err_set

# put data into batches
batch_size = hyper_params['batch_size']
train_loader = DataLoader(TensorDataset(f_J_tra, data_tra, err_r_tra), batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(TensorDataset(f_J_val, data_val, err_r_val), batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(TensorDataset(f_J_tes, data_tes, err_r_tes), batch_size=batch_size, shuffle=False)




# NN initialization
gmm = GMMNet(K, D, conditional_dim=1)
optimizer = torch.optim.Adam(gmm.parameters(),
                            lr=hyper_params["learning_rate"], 
                            weight_decay=hyper_params["weight_decay"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    factor=hyper_params["schedule_factor"], 
                                                    patience=hyper_params["patience"])



# training process
# training loop
epoch = hyper_params["num_epoch"]
lowest_loss = 9999
best_model  = copy.deepcopy(gmm)
for n in range(epoch):
    try:
        # training
        gmm.train()
        train_loss = 0
        for i, (f_J_i, data_i, err_r_i) in enumerate(train_loader):
            size_batch_i = f_J_i.shape[0]
            optimizer.zero_grad()
            log_prob_b, loss = gmm.score(data_i, f_J_i, noise=err_r_i, regression=True)
            train_loss += loss
            # backward and update parameters
            loss.backward()
            optimizer.step()

            #experiment.log_metric('batch_tra_loss', loss/size_batch_i, step=i)
        
        train_loss = train_loss / size_tra
        print('\nEpoch', (n+1), 'Training loss:', train_loss.item())
        scheduler.step(train_loss)

        # validating
        gmm.eval()
        val_loss = 0
        for i, (f_J_i, data_i, err_r_i) in enumerate(valid_loader):
            size_batch_i = f_J_i.shape[0]
            optimizer.zero_grad()
            log_prob_b, loss = gmm.score(data_i, f_J_i, noise=err_r_i)
            val_loss += loss

            #experiment.log_metric('batch_val_loss', loss/size_batch_i, step=i)
        
        val_loss = val_loss / size_val
        print('Epoch', (n+1), 'Validation loss:', val_loss.item())
        if val_loss < lowest_loss:
            lowest_loss = val_loss
            best_model  = copy.deepcopy(gmm)
            torch.save(best_model.state_dict(),
                f'params/params_d_K{K:d}.pkl')


    except KeyboardInterrupt:
        break
   
#all_figures(K, D, test_loader, best_model)

best_model.eval()
tes_loss = 0
for i, (f_J_i, data_i, err_r_i) in enumerate(test_loader):
    log_prob_b, loss = best_model.score(data_i, f_J_i, noise=err_r_i)
    tes_loss += loss

tes_loss = tes_loss / size_tes
print('\nTest loss:', tes_loss.item())