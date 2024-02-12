import os
import copy
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.experiment import data_load
from condxd.model import GMMNet
from diagnostics.plots_exp import exp_figures
# export PYTHONPATH="${PYTHONPATH}:mypath/extreme-deconvolution/py"
# to install extreme-deconvolution: run `export LIBRARY_PATH=/usr/local/Cellar/gsl/2.7.1/lib/` 
# in the terminal before `make`
from xdgmm import XDGMM



# hyper_params = {
#     "learning_rate": 1e-3,
#     "batch_size": 500,
#     "schedule_factor": 0.4,
#     "patience": 2,
#     "num_epoch": 100,
#     "weight_decay": 0.001,
#     "size_training": 80,
#     "size_validation": 20,
#     "size_testing": 0,
#     "n_gauss": 20
# }

# components, dimension of data, and dimension of conditional
K, D, D_cond = 10, 7, 1

# size of training sample and validation sample
N_t = 90000
N_v = N_t//9

# old method parameters
n_bin = 10
cond_bin_edges_l = np.linspace(0, 1, num=n_bin, endpoint=False)
cond_bin_edges_r = np.linspace(0, 1, num=n_bin, endpoint=False) + 1/n_bin
cond_bin_edges = np.array([cond_bin_edges_l, cond_bin_edges_r]).transpose()

# load training and validation data and parameters
seed_list = [9, 11, 13, 15, 17, 19, 21, 23, 25, 27]
KL_div_list = np.array([])
for seed in seed_list:
    cond_t, _, _, _, data_t, noise_t, _ = data_load(N_t, K, D, D_cond,
                                                    noisy=True, seed0=seed)
    cond_v, _, _, _, data_v, noise_v, _ = data_load(N_v, K, D, D_cond, 
                                                    noisy=True, seed0=seed)

    cond_t = torch.FloatTensor(cond_t)
    data_t = torch.FloatTensor(data_t)
    noise_t = torch.FloatTensor(noise_t)
    cond_v = torch.FloatTensor(cond_v)
    data_v = torch.FloatTensor(data_v)
    noise_v = torch.FloatTensor(noise_v)


    # put data into batches
    batch_size = 250

    train_loader = DataLoader(TensorDataset(cond_t, data_t, noise_t), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(cond_v, data_v, noise_v), batch_size=batch_size, shuffle=True)

    # initialization
    K_GMM = K
    CondXD = GMMNet(K, D, D_cond)

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(CondXD.parameters(), lr=learning_rate, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.4, patience=2)


    # NN parameter (weights) save directory
    param_path = f'params/experiment/seed{seed}/'
    if not os.path.exists(param_path):
        os.mkdir(param_path)
        print("Directory ", param_path , " created ")
    
    # training loop
    epoch = 100
    lowest_loss = 9999
    best_model  = copy.deepcopy(CondXD)
    train_loss_list = np.ones(epoch) * np.nan
    valid_loss_list = np.ones(epoch) * np.nan
    for n in range(epoch):
        try:
            # training
            CondXD.train()
            train_loss = 0
            for i, (cond_i, data_i, noise_i) in enumerate(train_loader):
                optimizer.zero_grad()
                loss = CondXD.loss(data_i, cond_i, noise=noise_i, regression=True)
                train_loss += loss

                # backward and update parameters
                loss.backward()
                optimizer.step()
            
            train_loss = train_loss / N_t
            train_loss_list[n] = train_loss
            print('Epoch', (n+1), 'Training loss', train_loss.item())
            scheduler.step(train_loss)

            # validation
            CondXD.eval()
            val_loss = 0
            for i, (cond_i, data_i, noise_i) in enumerate(valid_loader):
                optimizer.zero_grad()
                loss = CondXD.loss(data_i, cond_i, noise=noise_i, regression=True)
                val_loss += loss
            
            val_loss = val_loss / N_v
            valid_loss_list[n] = val_loss
            print('Epoch', (n+1), 'Validation loss', val_loss.item())
            if val_loss < lowest_loss:
                lowest_loss = val_loss
                best_model  = copy.deepcopy(CondXD)
                torch.save(best_model.state_dict(),
                            param_path+'params.pkl')

        except KeyboardInterrupt:
            break
    
    
    # old binning method
    print('Old binning method.')
    xd_list = np.array([])
    weights_last = mu_last = V_last = None
    for i in range(n_bin):
        print(f'Bin-XD: processing bin {i+1}/{n_bin}...')
        xd = XDGMM(K, method='Bovy', weights=weights_last, mu=mu_last, V=V_last)

        bin_filter = (cond_t <= cond_bin_edges[i,1]) & (cond_t > cond_bin_edges[i,0])
        bin_filter = bin_filter.flatten()
        data_t_bin = data_t[bin_filter].numpy()
        noise_t_bin = noise_t[bin_filter].numpy()
        
        xd.fit(data_t_bin, noise_t_bin)
        xd_list = np.append(xd_list, xd)
        weights_last = xd.weights
        mu_last = xd.mu
        V_last = xd.V
    print('Finished.')

    del cond_t, data_t, noise_t
    del cond_v, data_v, noise_v
    del data_t_bin, noise_t_bin

    fig_path = f'figs/experiment/seed{seed}/'
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
        print("Directory " , fig_path ,  " Created ")

    KL_div, cond = exp_figures(D_cond, K, D, 
                            train_loss_list, valid_loss_list,
                            best_model, cond_bin_edges, fig_path, seed,
                            binXD=xd_list)
    KL_div_list = np.append(KL_div_list, KL_div)



# calculating the mean KL divergence and their std when conditional fixed
n_type = 3
if xd_list is not None:
    n_type = 6
KL_div_np = np.zeros((len(seed_list), n_type, len(cond)))
for i, KL in enumerate(KL_div_list):
    KL_div_np[i][0] = KL['clean']
    KL_div_np[i][1] = KL['noisy']
    KL_div_np[i][2] = KL['maxim']
    if xd_list is not None:
        KL_div_np[i][3] = KL['clean_binXD']
        KL_div_np[i][4] = KL['noisy_binXD']
KL_div_mean = KL_div_np.mean(axis=0)
KL_div_std  = KL_div_np.std(axis=0)

# plotting KL divergence vs conditional
import matplotlib.pyplot as plt
plt.close('all')

# from IPython import embed
# embed(header='Now adjust the KL-Div figure.')
fig, ax = plt.subplots()

'''
linetypes = ['$D_\mathrm{KL}$(underlying | deconvolved fitted)',
            '$D_\mathrm{KL}$(noisy underlying | reconvolved fitted)',
            'Estimated Max $D_\mathrm{KL}$']
'''
linetypes = ['$D_\mathrm{KL, deconvolved}$',
            '$D_\mathrm{KL, noisy}$']
linestyles = ['solid', 'dashed', '-.']
methods = ['', '']
if xd_list is not None:
    methods = [', CondXD', ', bin-XD']

for i in [0, 1]:
    ax.plot(cond, KL_div_mean[i], label=linetypes[i]+methods[0], linestyle=linestyles[i], color='tab:red')
    ax.fill_between(cond, KL_div_mean[i]-KL_div_std[i], KL_div_mean[i]+KL_div_std[i],
                    facecolor='tab:red', alpha=0.5)
i = 2
ax.plot(cond, KL_div_mean[i], linestyle=linestyles[i], color='tab:red')
ax.fill_between(cond, KL_div_mean[i]-KL_div_std[i], KL_div_mean[i]+KL_div_std[i],
                facecolor='tab:red', alpha=0.5)
ax.annotate('Estimated Max $D_\mathrm{KL}$'+methods[0], xy=(0.27,2.55), color='tab:red',fontsize=12.5)

if xd_list is not None:
    for i in [3, 4]: # not play maximum estimation of bin-XD
        ax.plot(cond, KL_div_mean[i], label=linetypes[i-3]+methods[1], linestyle=linestyles[i-3], color='tab:blue')
        ax.fill_between(cond, KL_div_mean[i]-KL_div_std[i], KL_div_mean[i]+KL_div_std[i],
                        facecolor='tab:blue', alpha=0.35)
    ax.legend(fontsize=12.5, frameon=False)

ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xlabel('Conditional c', fontsize=14)
ax.set_ylabel('$D_\mathrm{KL}$', fontsize=14)
fig.savefig('figs/experiment/KLDiv_mean.pdf')
plt.close()

np.save('random_seed', KL_div_list)
