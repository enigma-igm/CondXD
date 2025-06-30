import os
import copy
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np

from paper.data.experiment import data_load
from condxd.CondXD import CondXD
from paper.diagnostics.plots_exp import exp_figures
# export PYTHONPATH="${PYTHONPATH}:mypath/extreme-deconvolution/py"
# to install extreme-deconvolution: run `export LIBRARY_PATH=/usr/local/Cellar/gsl/2.7.1/lib/` 
# in the terminal before `make`
from xdgmm import XDGMM

# components, dimension of data, and dimension of conditional
K, D, D_cond = 10, 7, 1

n_samples = 10000

seed = 10

batch_size = 250

# old method parameters
n_bin = 10
cond_bin_edges_l = np.linspace(0, 1, num=n_bin, endpoint=False)
cond_bin_edges_r = np.linspace(0, 1, num=n_bin, endpoint=False) + 1/n_bin
cond_bin_edges = np.array([cond_bin_edges_l, cond_bin_edges_r]).transpose()

seed_list = [12, 10, 14]
KL_div_list = np.array([])
for seed in seed_list:

    cond, _, _, _, sample, noise, _ = data_load(n_samples, K, D, D_cond,
                                                    noisy=True, seed0=seed)

    # test part:
    condxd = CondXD(K, D, D_cond)

    condxd.load_data(cond, sample, noise, (9,1,0), batch_size)

    condxd.deconvolve(5)

    print('Old binning method.')
    xd_list = np.array([])
    weights_last = mu_last = V_last = None
    for i in range(n_bin):
        print(f'Bin-XD: processing bin {i+1}/{n_bin}...')
        xd = XDGMM(K, method='Bovy', weights=weights_last, mu=mu_last, V=V_last)

        from IPython import embed
        print('still some bug.')
        embed()
        cond_t, data_t, noise_t = condxd.dataloader_tra.dataset.tensors[0:3]

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

    fig_path = f'figs/experiment/seed{seed}/'
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
        print("Directory " , fig_path ,  " Created ")

    KL_div, cond = exp_figures(D_cond, K, D, 
                            condxd.train_loss_list, condxd.valid_loss_list,
                            condxd, cond_bin_edges, fig_path, seed,
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
fig.savefig('figs/experiment/KLDiv_mean_test.pdf')
plt.close()

np.save('random_seed_test', KL_div_list)