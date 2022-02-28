import matplotlib.pyplot as plt
import numpy as np
import torch

from models.model import mvn

torch.set_num_threads(1)
from IPython import embed
# weights comparison plot
def weights_comp(cond, weights_r, weights_p, path):
    K = weights_r.shape[-1]
    fig, ax = plt.subplots()
    pw_r = ax.plot(cond.flatten(), weights_r, color='tab:blue', label='Real')
    pw_p = ax.plot(cond.flatten(), weights_p, color='tab:orange', label='Predicted')
    ax.set_xlabel(r'Conditional $\mathbf{c}$', fontsize=14)
    ax.set_ylabel(r'Weights $\mathbf{c}$', fontsize=14)
    ax.set_title(f'Weights of {K} Components', fontsize=16)
    customs = [pw_r[0], pw_p[0]]
    ax.legend(customs, [pw_r[0].get_label(), pw_p[0].get_label()], fontsize=10)
    plt.savefig(path+'weights.pdf')


# means comparison plot
def means_comp(cond, means_r, means_p, path):

    from matplotlib.collections import LineCollection
    K = means_r.shape[-2]
    D = means_r.shape[-1]

    fig, ax = plt.subplots()
    norm = plt.Normalize(cond.min(), cond.max())

    # means of the real simulated model
    for i in range(K):
        points = means_r[:,i,:].reshape(-1, 1, D)[:,:,0:2]
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='Blues', norm=norm)
        # Set the values used for colormapping
        lc.set_array(cond.flatten())
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
    cbar = plt.colorbar(line, ax=ax, aspect=15)
    cbar.set_label(r'Real vs Conditional $\mathbf{c}$', fontsize=14)

    # means of the NN predicted model
    for i in range(K):
        points = means_p[:,i,:].reshape(-1, 1, D)[:,:,0:2]
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='Oranges', norm=norm)
        # Set the values used for colormapping
        lc.set_array(cond.flatten())
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
        
    cbar = plt.colorbar(line, ax=ax, aspect=15)
    cbar.set_label(r'Predicted vs Conditional $\mathbf{c}$', fontsize=14)
    ax.set_xlim(means_p[:,:,0].min()-0.5, means_p[:,:,0].max()+0.5)
    ax.set_ylim(means_p[:,:,1].min()-0.5, means_p[:,:,1].max()+0.5)
    ax.set_xlabel('Dimension 1', fontsize=14)
    ax.set_ylabel('Dimension 2', fontsize=14)
    ax.set_title(f'Means of {K} Gaussians', fontsize=16)
    fig.savefig(path+'means.pdf')


# covariance comparison plot
def covars_comp(cond, covars_r, covars_p, num, path):

    K = covars_r.shape[-3]

    fig, ax = plt.subplots(num, 1, figsize=(6,num*1.7), sharex=True)
    fig.subplots_adjust(hspace=0)
    for i in range(num):
        pde_r = ax[i].plot(cond, covars_r[:,:,i,i], color='tab:blue', label='Real')
        pde_p = ax[i].plot(cond, covars_p[:,:,i,i], color='tab:orange', label='Predicted')
        ax[i].set_ylabel(f'{i+1, i+1} Element')
        ax[i].set_ylim([0, 0.4])
        ax[i].set_yticks(np.arange(0, 0.4, 0.1))
        customs = [pde_r[0], pde_p[0]]
        ax[i].legend(customs, [pde_r[0].get_label(), pde_p[0].get_label()], fontsize=10)
    ax[num-1].set_xlabel(r'Conditional $\mathbf{c}$', fontsize=14)
    ax[0].set_title(f'Elements of Covariances of {K} Gaussians', fontsize=16)
    fig.savefig(path+'/covars.pdf')


# density comparison plot
def density_comp(data_r, data_p, label, bins, conditional, noisy, path):
    """[summary]

    Args:
        data_r ([type]): [description]
        data_p ([type]): [description]
        label ([type]): [description]
        bins ([type]): [description]
        conditional ([type]): [description]
        noisy (bool): [description].
    """

    import corner
    if noisy == False:
        name = 'clean'
        tag = 'Deconvolved'
    elif noisy == True:
        name = 'noisy'
        tag = 'Noise Convolved'
    D = data_r.shape[-1]

    figure = corner.corner(data_r, color='tab:blue', labels=label, label_kwargs=dict(fontsize=16), bins=bins)
    corner.corner(data_p, fig=figure, color='tab:orange', labels=label, label_kwargs=dict(fontsize=16), bins=bins)
    axes = np.array(figure.axes).reshape(D, D)
    for ax in figure.get_axes():
        ax.tick_params(axis='both', direction='in', labelsize=12)
    annotate = 'Samples on NN vs Samples on Real Model,\n cond $\mathbf{c}$'+f'={conditional:.2f}'
    axes[1, -3].text(0.6, 0.8, annotate, fontsize=25, horizontalalignment='center', c='k', weight='bold')
    axes[1, -3].text(0.6, 0.5, tag+f' Real Model', fontsize=25, horizontalalignment='center', c='tab:blue', weight='bold')
    axes[1, -3].text(0.6, 0.2, tag+f' NN Prediction', fontsize=25, horizontalalignment='center', c='tab:orange', weight='bold')
    figure.savefig(path+name+f'Comp_{conditional:.2f}.pdf')


# K-L divergence plot
def KLdiv_figure(D_cond, cond, data_r_clean, data_r_noisy,
                weights_r, means_r, covars_r, noise,
                condGMM, cond_bin_edges, path,
                binGMM=None):
    """Plotting the Kullback-Leibler divergence vs Conditional. Output plots to figs/experiment/.

    Args:
        D_cond (int): Number of dimensions of the conditional.
        cond (array): The conditional parameters. Shaped 1D, length L.
        weights_r (narray): The weights of the real simulated model. Shaped LxK.
        means_r (narray): The means of the real simulated model. Shaped LxKxD.
        covars_r (narray): The covariances of the real simulated model. Shaped LxKxDxD. Must be positive definite.
        model (class): The GMM neural network.
        path (string): output path of the figure.

    Returns:
        KL_div.
    """

    if D_cond != 1:
        print('KL divergence plots canceled, because D_cond != 1.')
        return None
    
    L = len(cond_bin_edges)
    D = data_r_clean.shape[-1]

    # 
    KL_div_clean = np.zeros(L)
    KL_div_noisy = np.zeros(L)
    KL_div_cross = np.zeros(L)

    if binGMM is not None:
        KL_div_bin_clean = np.zeros(L)
        KL_div_bin_noisy = np.zeros(L)
        KL_div_bin_cross = np.zeros(L)
    
    weights_r = torch.FloatTensor(weights_r) 
    means_r   = torch.FloatTensor(means_r) 
    covars_r  = torch.FloatTensor(covars_r)
    noise     = torch.FloatTensor(noise)
    weights_p, means_p, covars_p = condGMM(torch.FloatTensor(cond).reshape(-1,1))
    
    
    for i in range(L):

        bin_filter = (cond > cond_bin_edges[i,0]) & (cond <= cond_bin_edges[i,1])
        data_r_clean_i = data_r_clean[bin_filter]
        data_r_noisy_i = data_r_noisy[bin_filter]
        noise_i        = noise[bin_filter]

        weights_r_i = weights_r[bin_filter]
        means_r_i   = means_r[bin_filter]
        covars_r_i  = covars_r[bin_filter]

        weights_p_i = weights_p[bin_filter]
        means_p_i   = means_p[bin_filter]
        covars_p_i  = covars_p[bin_filter]

        # for the deconvolved model
        logp_r_clean = condGMM.log_prob_b(torch.FloatTensor(data_r_clean_i), 
                                        weights_r_i, means_r_i, covars_r_i).detach().numpy()
        logp_p_clean = condGMM.log_prob_b(torch.FloatTensor(data_r_clean_i),
                                        weights_p_i, means_p_i, covars_p_i).detach().numpy()
        KL_div_clean[i] = (logp_r_clean - logp_p_clean).mean()

        # for the noise convolved model
        logp_r_noisy = condGMM.log_prob_b(torch.FloatTensor(data_r_noisy_i), 
                                        weights_r_i, means_r_i, covars_r_i, noise=noise_i).detach().numpy()
        logp_p_noisy = condGMM.log_prob_b(torch.FloatTensor(data_r_noisy_i), 
                                        weights_p_i, means_p_i, covars_p_i, noise=noise_i).detach().numpy()
        KL_div_noisy[i] = (logp_r_noisy - logp_p_noisy).mean()

        # probability of deconvolved samples on noise convolved model
        logp_p_cross = condGMM.log_prob_b(torch.FloatTensor(data_r_clean_i),
                                        weights_p_i, means_p_i, covars_p_i, noise=noise_i).detach().numpy()
        KL_div_cross[i] = (logp_r_clean - logp_p_cross).mean()


        # KL divergence plot for the old bin GMM method
        if binGMM is not None:
            binGMM_i = binGMM[i]
            weights_p_i = torch.FloatTensor(binGMM_i.weights)
            means_p_i   = torch.FloatTensor(binGMM_i.mu)
            covars_p_i  = torch.FloatTensor(binGMM_i.V)
            
            # for the deconvolved model
            logp_p_clean = condGMM.log_prob_b(torch.FloatTensor(data_r_clean_i),
                                            weights_p_i, means_p_i, covars_p_i).detach().numpy()
            KL_div_bin_clean[i] = (logp_r_clean - logp_p_clean).mean()

            # for the noise convolved model
            logp_p_noisy = condGMM.log_prob_b(torch.FloatTensor(data_r_noisy_i), 
                                            weights_p_i, means_p_i, covars_p_i, noise=noise_i).detach().numpy()
            KL_div_bin_noisy[i] = (logp_r_noisy - logp_p_noisy).mean()

            # probability of deconvolved samples on noise convolved model
            logp_p_cross = condGMM.log_prob_b(torch.FloatTensor(data_r_clean_i),
                                        weights_p_i, means_p_i, covars_p_i, noise=noise_i).detach().numpy()
            KL_div_bin_cross[i] = (logp_r_clean - logp_p_cross).mean()
            


    # figure. KL divergence vs conditional
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots()
    cond_axis = cond_bin_edges.mean(axis=-1)
    linetypes = ['Noise Convovled', 'Noise Deconvovled', 'Cross']
    ax.plot(cond_axis, KL_div_noisy, label=linetypes[0], linestyle='solid', color='tab:red')
    ax.plot(cond_axis, KL_div_clean, label=linetypes[1], linestyle='dashed', color='tab:red')
    ax.plot(cond_axis, KL_div_cross, label=linetypes[2], linestyle='-.', color='tab:red')
    ax.legend(fontsize=10)

    if binGMM is not None:
        methods = ['condGMM', 'bin-GMM']
        ax.plot(cond_axis, KL_div_bin_noisy, label=linetypes[0], linestyle='solid', color='tab:blue')
        ax.plot(cond_axis, KL_div_bin_clean, label=linetypes[1], linestyle='dashed', color='tab:blue')
        ax.plot(cond_axis, KL_div_bin_cross, label=linetypes[2], linestyle='-.', color='tab:blue')
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
    ax.set_title('K-L Divergence vs Conditional', fontsize=16)
    fig.savefig(path+'KLDiv.pdf')

    plt.close()

    KL_div = {}
    KL_div['noisy'] = KL_div_noisy
    KL_div['clean'] = KL_div_clean
    KL_div['cross'] = KL_div_cross
    if binGMM is not None:
        KL_div['noisy_binGMM'] = KL_div_bin_noisy
        KL_div['clean_binGMM'] = KL_div_bin_clean
        KL_div['cross_binGMM'] = KL_div_bin_cross

    return KL_div


# main function
def exp_figures(D_cond, K, D, weight_func, means_func, covar_func,
                noise_func, sample_func, model, cond_bin_edges, path, seed, binGMM=None):
    """Visualization of the experiment result. Only valid when D_cond=1. Output plots to figs/experiment/.

    Args:
        D_cond (int): Number of dimensions of the conditional.
        K (int): Number of Gaussian components in the model (how many Gaussians).
        D (int): Dimension of data.
        weight_func (function): Deriving weights from conditional. See data/experiment.py.
        means_func (function): Deriving means from conditional. See data/experiment.py.
        covar_func (function): Deriving covariance from conditional. See data/experiment.py.
        noise_func (function): Deriving noise from conditional. See data/experiment.py.
        sample_func (function): Random sampling from a Gaussian mixture distribution. See data/experiment.py.
        model (class): The GMM neural network.
        path (string): output path of the figure.
        seed (int): Random seed.
    """

    if D_cond != 1:
        print('Plotting canceled, because D_cond != 1.')
        return None
    
    cond = np.linspace(0.02, 1, 50).reshape(-1, 1)
    # r for real. real GMM parameters.
    weights_r = np.zeros((len(cond), K))
    means_r   = np.zeros((len(cond), K, D))
    covars_r  = np.zeros((len(cond), K, D, D))
    for i in range(len(cond)):
        weights_r[i] = weight_func(cond[i], K, seed=seed)
        means_r[i] = means_func(cond[i], K, D, seed=seed+4)
        covars_r[i] = covar_func(cond[i], K, D, seed=seed+12)

    # p for predicted. predicted GMM parameters.
    weights_p, means_p, covars_p = model(torch.FloatTensor(cond))
    weights_p = weights_p.detach().numpy()
    means_p   = means_p.detach().numpy()
    covars_p  = covars_p.detach().numpy()


    # global comparison
    # comparing weights
    weights_comp(cond.flatten(), weights_r, weights_p, path)


    # comparing means
    means_comp(cond.flatten(), means_r, means_p, path)


    # comparing covars
    num = 4
    covars_comp(cond, covars_r, covars_p, num, path)


    # density comparison at specific conditionals. corner plots.
    j_list = [4, 24, 44] # cond = 0.1, 0.5, 0.9
    for j in j_list:
        N_data = 10000
        data_r_clean = sample_func(weights_r[j], means_r[j], covars_r[j], N=N_data)[0]
        data_p_clean = model.sample(torch.FloatTensor(cond[j]).unsqueeze(0), N_data).squeeze().numpy()
        noise = np.zeros((N_data, D, D))
        data_r_noisy = np.zeros((N_data, D))
        data_p_noisy = np.zeros((N_data, D))
        for i in range(N_data):
            noise[i]  = noise_func(cond[j], D, sigma_d=1., sigma_l=0.5)
            data_r_noisy[i] = sample_func(weights_r[j], means_r[j], covars_r[j], noise=noise[i])[0]
            data_p_noisy[i] = model.sample(torch.FloatTensor(cond[j]).unsqueeze(0), 1, 
                                            torch.FloatTensor(noise[i]).unsqueeze(0)).squeeze().numpy()

        # corner plots parameters
        label = [f'Dim. {i+1}' for i in range(D)]
        bins = 25

        # cornerplots clean
        conditional = cond[j][0]
        density_comp(data_r_clean, data_p_clean, label, bins, conditional, noisy=False, path=path)
        # conerplots nosiy
        density_comp(data_r_noisy, data_p_noisy, label, bins, conditional, noisy=True, path=path)




    # plotting Kullback-Leibler divergence
    N_data = 25000
    cond = np.random.rand(N_data, D_cond)
    # r for real. real GMM parameters.
    weights_r  = np.zeros((N_data, K))
    means_r    = np.zeros((N_data, K, D))
    covars_r   = np.zeros((N_data, K, D, D))
    noise      = np.zeros((N_data, D, D))

    data_r_noisy = np.zeros((N_data, D))
    data_r_clean = np.zeros((N_data, D))

    for i in range(N_data):
        weights_r[i] = weight_func(cond[i], K, seed=seed)
        means_r[i] = means_func(cond[i], K, D, seed=seed+4)
        covars_r[i] = covar_func(cond[i], K, D, seed=seed+12)
        noise[i] = noise_func(cond[i], D, sigma_d=1., sigma_l=0.5)

        data_r_clean[i] = sample_func(weights_r[i], means_r[i], covars_r[i], N=1)[0]
        data_r_noisy[i] = sample_func(weights_r[i], means_r[i], covars_r[i], noise=noise[i])[0]
        
    KL_div = KLdiv_figure(D_cond, cond.flatten(),
                        data_r_clean, data_r_noisy,
                        weights_r, means_r, covars_r, noise,
                        model, cond_bin_edges, path,
                        binGMM=binGMM)

    plt.close()

    cond_axis = cond_bin_edges.mean(axis=-1)

    return KL_div, cond_axis