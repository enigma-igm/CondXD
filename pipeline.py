from torch.utils import data
from data.toy import *
from models.model import *
from diagnostics.toy import all_figures, KLdiv_figure

import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

from more_itertools import chunked

from sklearn.cluster import KMeans

import copy
import os

from IPython import embed

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# components, dimension of data, and dimension of conditional
K, D, D_cond = 7, 7, 1

# size of training sample and validation sample (no validation yet)
N_t = 10000
N_v = N_t//10

# load training and validation data and parameters
param_cond_t, weights_t, means_t, covars_t, data_t, noise_t, draw_t = data_load(N_t, K, D, D_cond, noisy=True)
param_cond_v, weights_v, means_v, covars_v, data_v, noise_v, draw_v = data_load(N_v, K, D, D_cond, noisy=True)


# kmeans to classify each data point. The means0 serves as the origin of the means.
kmeans_t = KMeans(n_clusters=K, random_state=0).fit(data_t)
means0_t = kmeans_t.cluster_centers_

kmeans_v = KMeans(n_clusters=K, random_state=0).fit(data_v)
means0_v = kmeans_v.cluster_centers_


# initialization
gmm = GMMNet(K, D, D_cond, torch.FloatTensor(means0_t))

learning_rate = 1e-3
optimizer = torch.optim.Adam(gmm.parameters(), lr=learning_rate, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.4, patience=2)

# calculate the log-likelihood on the real model, i.e. subtrahend in the KL divergence
log_true = 0
for i, data_i in enumerate(data_t):
    
    log_resp = mvn(loc=torch.from_numpy(means_t[i][None, :]),
                   covariance_matrix=torch.from_numpy(covars_t[i][None, :])
                   ).log_prob(torch.from_numpy(data_i[None, None, :]))

    log_resp += torch.log(torch.from_numpy(weights_t[i]))

    log_prob = torch.logsumexp(log_resp, dim=1)
    
    log_true += log_prob
    
log_true = log_true / data_t.shape[0]



# training process
# put data into batches
batch_size = 250

def chunck(array, bsize):
    # chunck
    array = list(chunked(array, bsize)) 
    array = torch.FloatTensor(array)

    return array

data_t = chunck(data_t, batch_size)
data_v = chunck(data_v, batch_size)

param_cond_t = chunck(param_cond_t, batch_size)
param_cond_v = chunck(param_cond_v, batch_size)

noise_t = chunck(noise_t, batch_size)
noise_v = chunck(noise_v, batch_size)


# training loop
epoch = 250
lowest_loss = 9999
best_model  = copy.deepcopy(gmm)
for n in range(epoch):
    try:
        # training
        gmm.train()
        train_loss = 0
        for i, data_i in enumerate(data_t):
            optimizer.zero_grad()
            loss = -gmm.log_prob_b(data_i, param_cond_t[i], noise=noise_t[i]).mean()
            train_loss += loss.item()

            # backward and update parameters
            loss.backward()
            optimizer.step()
        
        train_loss = train_loss / data_t.shape[0]
        print('Epoch', (n+1), 'Training loss', train_loss)
        scheduler.step(train_loss)

        # validating
        gmm.eval()
        val_loss = 0
        for i, data_i in enumerate(data_v):
            optimizer.zero_grad()
            loss = -gmm.log_prob_b(data_i, param_cond_v[i], noise=noise_v[i]).mean()
            val_loss += loss.item()
        val_loss = val_loss / data_v.shape[0]
        print('Epoch', (n+1), 'Validation loss', val_loss)
        if val_loss < lowest_loss:
            lowest_loss = val_loss
            best_model  = copy.deepcopy(gmm)


    except KeyboardInterrupt:
        break
    

KL_Div = train_loss + log_true.numpy()
print(f'KL divergense = {KL_Div}')
all_figures(K, D, weight_func, means_func, covar_func, sample_func, best_model, data_t, means0_t)
KLdiv_figure(param_cond_t, weights_t, means_t, covars_t, data_t, noise_t, best_model)