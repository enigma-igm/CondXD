import torch
from torch.utils.data import DataLoader, TensorDataset

from data.toy import *
from models.model import *
from diagnostics.toy import all_figures, KLdiv_figure

import seaborn as sns

from astropy.io import fits

from sklearn.cluster import KMeans

import copy
import os

from IPython import embed

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# read file
file = fits.open('../../../4Master/Research1/VIKING_catalog.fits')
data = copy.deepcopy(file[1].data)
file.close()

# load reference band and error
f_J = data['f_J'].astype('float').reshape(-1, 1)
f_J_err = data['f_J_err'].astype('float').reshape(-1, 1)
# transform to tensor
f_J = torch.Tensor(f_J)
f_J_err = torch.Tensor(f_J_err)

# load relative flux
nf_Y = data['f_Y'] / data['f_J']
nf_H = data['f_H'] / data['f_J']
nf_Ks = data['f_Ks'] / data['f_J']
nf_g = data['f_g'] / data['f_J']
nf_r = data['f_r'] / data['f_J']
nf_z = data['f_z'] / data['f_J']
nf_w1 = data['f_w1'] / data['f_J']
nf_w2 = data['f_w2'] / data['f_J']
data_set = torch.Tensor(np.array([nf_Y, nf_H, nf_Ks, nf_g, nf_r, nf_z, nf_w1, nf_w2]))
data_set = data_set.transpose(1, 0)

# load errors
f_Y_err = data['f_Y_err']
f_H_err = data['f_H_err']
f_Ks_err = data['f_Ks']
f_g_err = data['f_g_err']
f_r_err = data['f_r_err']
f_z_err = data['f_z_err']
f_w1_err = data['f_w1_err']
f_w2_err = data['f_w2_err']
err_set  = torch.Tensor(np.array([f_Y_err, f_H_err, f_Ks_err, f_g_err, f_r_err, f_z_err, f_w1_err, f_w2_err]))
err_set  = err_set.transpose(1, 0)


# setup parameter: length of data and data dimension
len_data, D = data_set.shape
# Gaussian components
K = D
# size of training / validation /test set
size_tra = 80
size_val = 5
size_tes = 15

# new covariance matrix
cov_set = torch.zeros((len_data, D, D))
# off-diagonal element
for i in range(1, D):
    for j in range(i):
        cov_set[:,i,j] = data_set[:,i] * data_set[:,j] / f_J[:,0]**2 * f_J_err[:,0]**2
cov_set = cov_set + cov_set.transpose(2, 1)
# diagonal element
for i in range(D):
    cov_set[:,i,i] = 1/f_J[:,0]**2 * err_set[:,i]**2 + data_set[:,i]**2 / f_J[:,0]**2 * f_J_err[:,0]**2

del data


# divide to training and validation set
def real_size(size_tra, size_val, size_tes, len_data):
    sum_size = size_tra + size_val + size_tes
    real_size_tra = np.round(len_data*size_tra/sum_size).astype('int')
    real_size_val = np.round(len_data*size_val/sum_size).astype('int')
    real_size_tes = np.round(len_data*size_tes/sum_size).astype('int')
    return (real_size_tra, real_size_val, real_size_tes)
def get_set(data_set, cov_set, id_sep, asign):
    data = data_set[id_sep==asign]
    cov  = cov_set[id_sep==asign]
    return (data, cov)

size_tra, size_val, size_tes = real_size(size_tra, size_val, size_tes, len_data)
id_sep = np.append(np.ones(size_tra), np.append(np.ones(size_val)*2, np.ones(size_tes)*3)).astype('int')
np.random.seed()
np.random.shuffle(id_sep)
f_J_tra, f_J_err_tra = get_set(f_J, f_J_err, id_sep, 1)
f_J_val, f_J_err_val = get_set(f_J, f_J_err, id_sep, 2)
f_J_tes, f_J_err_tes = get_set(f_J, f_J_err, id_sep, 3)
data_tra, cov_tra = get_set(data_set, cov_set, id_sep, 1)
data_val, cov_val = get_set(data_set, cov_set, id_sep, 2)
data_tes, cov_tes = get_set(data_set, cov_set, id_sep, 3)

del data_set, err_set

# put data into batches
batch_size = 200
train_loader_tra = DataLoader(TensorDataset(f_J_tra, data_tra, cov_tra), batch_size=batch_size, shuffle=True)
train_loader_val = DataLoader(TensorDataset(f_J_val, data_val, cov_val), batch_size=batch_size, shuffle=True)
train_loader_tes = DataLoader(TensorDataset(f_J_tes, data_tes, cov_tes), batch_size=batch_size, shuffle=True)

# training process

# kmeans to classify each data point. The means0 serves as the origin of the means.
kmeans_t = KMeans(n_clusters=K, random_state=0).fit(data_tra.numpy())
means0_t = kmeans_t.cluster_centers_

kmeans_v = KMeans(n_clusters=K, random_state=0).fit(data_val.numpy())
means0_v = kmeans_v.cluster_centers_



# NN initialization
gmm = GMMNet(K, D, 1, torch.FloatTensor(means0_t))

learning_rate = 1e-3
optimizer = torch.optim.Adam(gmm.parameters(), lr=learning_rate, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.4, patience=2)

def reg_loss(self, covars):
    '''
    regression loss.
    '''
    l = self.w / torch.diagonal(covars, dim1=-1, dim2=-2)
    return l.sum(axis=(-1,-2))



# training process
# training loop
epoch = 250
lowest_loss = 9999
best_model  = copy.deepcopy(gmm)
for n in range(epoch):
    try:
        # training
        gmm.train()
        train_loss = 0
        for i, (f_J_i, data_i, cov_i) in enumerate(train_loader_tra):
            optimizer.zero_grad()
            log_prob_b, loss = gmm.fit(data_i, f_J_i, noise=cov_i, regression=True)

            train_loss += -log_prob_b.sum().item()
            # backward and update parameters
            loss.backward()
            optimizer.step()
        
        train_loss = train_loss / size_tra
        print('Epoch', (n+1), 'Training loss', train_loss)
        scheduler.step(train_loss)

        # validating
        gmm.eval()
        val_loss = 0
        for i, (f_J_i, data_i, cov_i) in enumerate(train_loader_val):
            optimizer.zero_grad()
            log_prob_b, loss = gmm.fit(data_i, f_J_i, noise=cov_i)

            val_loss += -log_prob_b.sum().item()
        val_loss = val_loss / size_val
        print('Epoch', (n+1), 'Validation loss', val_loss)
        if val_loss < lowest_loss:
            lowest_loss = val_loss
            best_model  = copy.deepcopy(gmm)


    except KeyboardInterrupt:
        break
    

#all_figures(K, D, weight_func, means_func, covar_func, sample_func, best_model, data_t, means0_t)
#Ldiv_figure(param_cond_t, weights_t, means_t, covars_t, data_t, noise_t, best_model)