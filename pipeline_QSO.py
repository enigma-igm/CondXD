import torch
from torch.utils.data import DataLoader, TensorDataset

from data.toy import *
from models.model import *
from diagnostics.toy import all_figures

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
#print('Check data_set transposed correctly.')
#embed()

# load errors
f_z_err = data['f_z_err']
f_Y_err = data['f_Y_err']
f_H_err = data['f_H_err']
f_Ks_err = data['f_Ks']
f_w1_err = data['f_w1_err']
f_w2_err = data['f_w2_err']
err_set  = torch.Tensor(np.array([f_z_err, f_Y_err, f_H_err, f_Ks_err, f_w1_err, f_w2_err]))
err_set  = err_set.transpose(1, 0)

del data
'''
# test: a narrow J band flux bin
bln = ((f_J>=1) & (f_J<=1.5)).flatten()
f_J = f_J[bln]
f_J_err = f_J_err[bln]
data_set = data_set[bln]
err_set = err_set[bln]
'''
# setup parameter: length of data and data dimension
len_data, D = data_set.shape
# Gaussian components
K = 25
# size of training / validation /test set
size_tra = 80
size_val = 5
size_tes = 15


# new covariance matrix
err_r_set = torch.zeros((len_data, D, D))
# off-diagonal element
for i in range(1, D):
    for j in range(i):
        err_r_set[:,i,j] = data_set[:,i] * data_set[:,j] / f_J[:,0]**2 * f_J_err[:,0]**2
err_r_set = err_r_set + err_r_set.transpose(2, 1)
# diagonal element
for i in range(D):
    err_r_set[:,i,i] = 1/f_J[:,0]**2 * err_set[:,i]**2 + data_set[:,i]**2 / f_J[:,0]**2 * f_J_err[:,0]**2
#print('Check err_r_set setup correctly.')
#embed()
f_J_err = f_J_err/f_J
f_J = torch.log(f_J)
#f_J_err = f_J_err/f_J.std()
#f_J = (f_J-f_J.mean()) / f_J.std()

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
#print('check id_sep.')
#embed()
f_J_tra, f_J_err_tra = get_set(f_J, f_J_err, id_sep, 1)
f_J_val, f_J_err_val = get_set(f_J, f_J_err, id_sep, 2)
f_J_tes, f_J_err_tes = get_set(f_J, f_J_err, id_sep, 3)
data_tra, err_r_tra = get_set(data_set, err_r_set, id_sep, 1)
data_val, err_r_val = get_set(data_set, err_r_set, id_sep, 2)
data_tes, err_r_tes = get_set(data_set, err_r_set, id_sep, 3)
'''
#embed()
data_tra_mean = data_tra.mean(axis=0)
data_tra_std  = data_tra.std(axis=0)

data_tra  = (data_tra - data_tra_mean)/data_tra_std
err_r_tra = err_r_tra/np.outer(data_tra_std, data_tra_std)

data_val  = (data_val - data_tra_mean)/data_tra_std
err_r_val = err_r_val/np.outer(data_tra_std, data_tra_std)

data_tes  = (data_tes - data_tra_mean)/data_tra_std
err_r_tes = err_r_tes/np.outer(data_tra_std, data_tra_std)
'''
del data_set, err_set

# put data into batches
batch_size = 500
train_loader_tra = DataLoader(TensorDataset(f_J_tra, data_tra, err_r_tra), batch_size=batch_size, shuffle=True)
train_loader_val = DataLoader(TensorDataset(f_J_val, data_val, err_r_val), batch_size=batch_size, shuffle=True)
train_loader_tes = DataLoader(TensorDataset(f_J_tes, data_tes, err_r_tes), batch_size=batch_size, shuffle=True)


# kmeans to classify each data point. The means0 serves as the origin of the means.
kmeans_t = KMeans(n_clusters=K, random_state=0).fit(data_tra.numpy())
means0_t = kmeans_t.cluster_centers_

#kmeans_v = KMeans(n_clusters=K, random_state=0).fit(data_val.numpy())
#means0_v = kmeans_v.cluster_centers_



# NN initialization
gmm = GMMNet(K, D, 1, torch.FloatTensor(means0_t))

learning_rate = 1e-3
optimizer = torch.optim.Adam(gmm.parameters(), lr=learning_rate, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.4, patience=2)



# training process
# training loop
epoch = 100
lowest_loss = 9999
best_model  = copy.deepcopy(gmm)
for n in range(epoch):
    try:
        # training
        gmm.train()
        train_loss = 0
        for i, (f_J_i, data_i, err_r_i) in enumerate(train_loader_tra):
            optimizer.zero_grad()
            log_prob_b, loss = gmm.fit(data_i, f_J_i, noise=err_r_i, regression=True)
            train_loss += loss
            # backward and update parameters
            loss.backward()
            optimizer.step()
        
        train_loss = train_loss / size_tra
        print('\nEpoch', (n+1), 'Training loss:', train_loss.item())
        scheduler.step(train_loss)

        # validating
        gmm.eval()
        val_loss = 0
        for i, (f_J_i, data_i, err_r_i) in enumerate(train_loader_val):
            optimizer.zero_grad()
            log_prob_b, loss = gmm.fit(data_i, f_J_i, noise=err_r_i)
            val_loss += loss
        
        val_loss = val_loss / size_val
        print('Epoch', (n+1), 'Validation loss:', val_loss.item())
        if val_loss < lowest_loss:
            lowest_loss = val_loss
            best_model  = copy.deepcopy(gmm)
            torch.save(best_model.state_dict(),
                f'params/params_d_K{K:d}.pkl')


    except KeyboardInterrupt:
        break
embed()    
all_figures(K, D, train_loader_tes, gmm)

gmm.eval()
tes_loss = 0
for i, (f_J_i, data_i, err_r_i) in enumerate(train_loader_tes):
    log_prob_b, loss = gmm.fit(data_i, f_J_i, noise=err_r_i)
    tes_loss += loss

tes_loss = tes_loss / size_tes
print('\nTest loss:', tes_loss.item())