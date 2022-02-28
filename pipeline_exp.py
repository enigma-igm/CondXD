from data.experiment import *
from models.model import *
from diagnostics.plots_exp import exp_figures

from more_itertools import chunked

from xdgmm import XDGMM

from IPython import embed

import copy
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# components, dimension of data, and dimension of conditional
K, D, D_cond = 10, 7, 1

# size of training sample and validation sample
N_t = 60000
N_v = N_t//6

# old method parameters
n_bin = 10
cond_bin_edges_l = np.linspace(0, 1, num=n_bin, endpoint=False)
cond_bin_edges_r = np.linspace(0, 1, num=n_bin, endpoint=False) + 1/n_bin
cond_bin_edges = np.array([cond_bin_edges_l, cond_bin_edges_r]).transpose()

# load training and validation data and parameters
seed_list = [9, 11, 13, 15, 17, 19, 21, 23, 25, 26]
KL_div_list = np.array([])
for seed in seed_list:
    cond_t, weights_t, means_t, covars_t, data_t, noise_t, draw_t = data_load(N_t, K, D, D_cond, noisy=True, seed0=seed)
    cond_v, weights_v, means_v, covars_v, data_v, noise_v, draw_v = data_load(N_v, K, D, D_cond, noisy=True, seed0=seed)


    # initialization
    K_GMM = K
    gmm = GMMNet(K, D, D_cond)

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

    def chunk(array, bsize):
        # chunk the data set into mini-batches
        array = list(chunked(array, bsize)) 
        array = torch.FloatTensor(array)

        return array

    data_t_unbatch = data_t.copy()
    data_t = chunk(data_t, batch_size)
    data_v = chunk(data_v, batch_size)

    cond_t_unbatch = cond_t.copy()
    cond_t = chunk(cond_t, batch_size)
    cond_v = chunk(cond_v, batch_size)

    noise_t_unbatch = noise_t.copy()
    noise_t = chunk(noise_t, batch_size)
    noise_v = chunk(noise_v, batch_size)


    # NN parameter (weights) save directory
    param_path = f'params/experiment/seed{seed}/'
    if not os.path.exists(param_path):
        os.mkdir(param_path)
        print("Directory " , param_path ,  " Created ")
    
    # training loop
    epoch = 200
    lowest_loss = 9999
    best_model  = copy.deepcopy(gmm)
    for n in range(epoch):
        try:
            # training
            gmm.train()
            train_loss = 0
            for i, data_i in enumerate(data_t):
                optimizer.zero_grad()
                log_prob_b, loss = gmm.score(data_i, cond_t[i], noise=noise_t[i])
                train_loss += loss

                # backward and update parameters
                loss.backward()
                optimizer.step()
            
            train_loss = train_loss / N_t
            print('Epoch', (n+1), 'Training loss', train_loss.item())
            scheduler.step(train_loss)

            # validating
            gmm.eval()
            val_loss = 0
            for i, data_i in enumerate(data_v):
                optimizer.zero_grad()
                log_prob_b, loss = gmm.score(data_i, cond_v[i], noise=noise_v[i])
                val_loss += loss
            val_loss = val_loss / N_v
            print('Epoch', (n+1), 'Validation loss', val_loss.item())
            if val_loss < lowest_loss:
                lowest_loss = val_loss
                best_model  = copy.deepcopy(gmm)
                torch.save(best_model.state_dict(),
                            param_path+'params.pkl')

        except KeyboardInterrupt:
            break
    
    
    # old binning method
    print('Old binning method.')
    xd_list = np.array([])
    weights_last = mu_last = V_last = None
    for i in range(n_bin):
        print(f'Bin-GMM: processing bin {i+1}/{n_bin}...')
        xd = XDGMM(K, method='Bovy', weights=weights_last, mu=mu_last, V=V_last)

        bin_filter = (cond_t_unbatch <= cond_bin_edges[i,1]) & (cond_t_unbatch > cond_bin_edges[i,0])
        bin_filter = bin_filter.flatten()
        data_t_bin = data_t_unbatch[bin_filter]
        noise_t_bin = noise_t_unbatch[bin_filter]

        xd.fit(data_t_bin, noise_t_bin)
        xd_list = np.append(xd_list, xd)
        weights_last = xd.weights
        mu_last = xd.mu
        V_last = xd.V
    print('Finished.')

    del cond_t, weights_t, means_t, covars_t, data_t, noise_t, draw_t
    del cond_v, weights_v, means_v, covars_v, data_v, noise_v, draw_v
    del data_t_bin, cond_t_unbatch, noise_t_unbatch, noise_t_bin

    fig_path = f'figs/experiment/seed{seed}/'
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
        print("Directory " , fig_path ,  " Created ")

    KL_div, cond = exp_figures(D_cond, K, D,
                                weight_func, means_func, covar_func, noise_func, sample_func,
                                best_model, cond_bin_edges, fig_path, seed,
                                binGMM=xd_list)
    KL_div_list = np.append(KL_div_list, KL_div)



# calculating the mean KL divergence and their std when conditional fixed
n_type = 3
if xd_list is not None:
    n_type = 6
KL_div_np = np.zeros((len(seed_list), n_type, len(cond)))
for i, KL in enumerate(KL_div_list):
    KL_div_np[i][0] = KL['noisy']
    KL_div_np[i][1] = KL['clean']
    KL_div_np[i][2] = KL['cross']
    if xd_list is not None:
        KL_div_np[i][3] = KL['noisy_binGMM']
        KL_div_np[i][4] = KL['clean_binGMM']
        KL_div_np[i][5] = KL['cross_binGMM']
KL_div_mean = KL_div_np.mean(axis=0)
KL_div_std  = KL_div_np.std(axis=0)
embed()
# plotting KL divergence vs conditional
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
fig, ax = plt.subplots()

linetypes = ['Noise Convovled', 'Noise Deconvovled', 'Cross']
linestyles = ['solid', 'dashed', '-.']
for i in [0, 1, 2]:
    ax.plot(cond, KL_div_mean[i], label=linetypes[i], linestyle=linestyles[i], color='tab:red')
    ax.fill_between(cond, KL_div_mean[i]-KL_div_std[i], KL_div_mean[i]+KL_div_std[i],
                    facecolor='tab:red', alpha=0.5)
ax.legend(fontsize=10)

if xd_list is not None:
    methods = ['condGMM', 'bin-GMM']
    for i in [3, 4, 5]:
        ax.plot(cond, KL_div_mean[i], label=linetypes[i], linestyle=linestyles[i], color='tab:blue')
        ax.fill_between(cond, KL_div_mean[i]-KL_div_std[i], KL_div_mean[i]+KL_div_std[i],
                        facecolor='tab:blue', alpha=0.5)
    customs1 = [Line2D([0], [0], linestyle='solid', color='tab:red'),
                Line2D([0], [0], linestyle='solid', color='tab:blue')]
    customs2 = [Line2D([0], [0], linestyle='solid', color='k'),
                Line2D([0], [0], linestyle='dashed', color='k'),
                Line2D([0], [0], linestyle='-.', color='k')]
    customs = customs1 + customs2
    labels  = methods  + linetypes
    ax.legend(customs, labels, fontsize=10)

ax.set_xlabel('Conditional Parameter c', fontsize=14)
ax.set_ylabel('K-L Divergence', fontsize=14)
ax.set_title('K-L Divergence vs Conditional, $n_\mathrm{seed}='+f'{len(seed_list)}$', fontsize=16)
fig.savefig('figs/experiment/KLDiv_mean.pdf')
plt.show()

np.save('random_seed', KL_div_list)
