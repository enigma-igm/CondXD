from GMMNet.data.toy import noise_func
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
import numpy as np
import torch
import corner

from IPython import embed
from models.model import mvn

def covars_ellipse(covars):
    """The parameters of the ellipse of the covariance matrix. Only valid when dimension=2.

    Args:
        covars (narray): The covariance matrix. Shaped in 2x2.

    Returns:
        lambda1, lambda2, theta(float, float, float): the 2 eigenvalues(semi-major axis, semi-minor axis) and the rotating angle in degree.
    """
    a = covars[:,0,0]
    b = covars[:,0,1]
    c = covars[:,1,1]
    lambda1 = ((a+c) + np.sqrt((a-c)**2+4*b**2)) / 2
    lambda2 = ((a+c) - np.sqrt((a-c)**2+4*b**2)) / 2
    theta   = np.arctan2(lambda1-a, b) * 180/np.pi

    return (lambda1, lambda2, theta)


def add_ellipse(ax, data, lambda1, lambda2, theta, color='none'):
    """Add covariance ellipses to the plot.

    Args:
        ax ([type]): [description]
        data (narray): The data points. Shaped in nx2.
        lambda1 (array): One semi axis. Length n.
        lambda2 (array): Another semi axis. length n.
        theta (array): Rotation angle of lambda1 with respect to (0,1). In degree. Length n.
        color (str, optional): The color of the ellipses. Defaults to 'none'.
    """
    ells = [Ellipse(xy=data[i],
                    width=lambda1[i], height=lambda2[i],
                    angle=theta[i],
                    facecolor='none',
                    edgecolor=color,
                    alpha=0.2)
            for i in range(len(data))]
    
    for e in ells:
        ax.add_artist(e)
    
    


def all_figures(K,
                D,
                weight_func,
                means_func,
                covar_func,
                sample_func,
                gmm,              
                data_t, 
                means0_t
                ):
    """ All figures to show the performances of our network.

    Args:
        K (int): Number of Gaussian components (how many Gaussians).
        D (int): Dimension of data.
        sample_func (function): Random sampling from a Gaussian mixture distribution. See data/toy.py.
        gmm (class): Neural network for Gaussian Mixture Model. See model/model.py
        data_t ([type]): [description]
        means0_t ([type]): [description]
    """


    # global check
    Nr = 10000
    sigma_d = 1
    sigma_l = 0.5
    # conditional parameter covering the training range
    param_cond_tes = np.arange(0, 1, 0.01) + 0.01
    param_cond_tes = torch.FloatTensor(param_cond_tes.reshape(-1,1))

    # derive the trained parameters
    weight_tes, means_tes, covars_tes = gmm(param_cond_tes)
    weight_tes = weight_tes.detach().numpy()
    means_tes  = means_tes.detach().numpy()
    covars_tes = covars_tes.detach().numpy()

    param_cond_tes = param_cond_tes.numpy()

    # derive the true paramters
    weight_r   = np.zeros((len(param_cond_tes), K))
    means_r    = np.zeros((len(param_cond_tes), K, D))
    covars_r   = np.zeros((len(param_cond_tes), K, D, D))
    for i in range(len(param_cond_tes)):
        weight_r[i]   = weight_func(param_cond_tes[i], K)
        means_r[i]    = means_func(param_cond_tes[i], K, D)
        covars_r[i]   = covar_func(param_cond_tes[i], K, D)
    
    # figure. all samples and initial guess of the means
    data_t = data_t.numpy().reshape(-1, D)
    fig, ax = plt.subplots()
    pd_t = ax.scatter(*data_t[:,0:2].transpose(), marker='.', color='grey', alpha=0.5, label='Training Set')
    pd_k = ax.scatter(*means0_t[:,0:2].transpose(), s=80, marker='.', color='tab:orange', label='kmeans centroids')
    ax.set_title('Training Set', fontsize=14)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.legend(fontsize=14)
    fig.savefig('figs/trainingset.pdf')

    # figure. plot means vs conditional parameter
    fig, ax = plt.subplots()
    norm = plt.Normalize(param_cond_tes.min(), param_cond_tes.max())
    for i in range(K):
        points = means_r[:,i,:].reshape(-1, 1, D)[:,:,0:2]
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='Blues', norm=norm)
        # Set the values used for colormapping
        lc.set_array(param_cond_tes.flatten())
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
    cbar = plt.colorbar(line, ax=ax, aspect=15)
    cbar.set_label('True vs Conditional Parameter z', fontsize=10)
    for i in range(gmm.n_components):
        points = means_tes[:,i,:].reshape(-1, 1, D)[:,:,0:2]
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='Oranges', norm=norm)
        # Set the values used for colormapping
        lc.set_array(param_cond_tes.flatten())
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
    cbar = plt.colorbar(line, ax=ax, aspect=15)
    cbar.set_label('Predicted vs Conditional Parameter z', fontsize=10)
    ax.set_xlim(means_tes[:,:,0].min()-0.5, means_tes[:,:,0].max()+0.5)
    ax.set_ylim(means_tes[:,:,1].min()-0.5, means_tes[:,:,1].max()+0.5)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Means of Each Component', fontsize=14)
    fig.savefig('figs/means.pdf')


    # figure. plot weights vs conditional paramters
    fig, ax = plt.subplots()
    pw_r = ax.plot(param_cond_tes, weight_r, color='tab:blue', label='True')
    pw_tes = ax.plot(param_cond_tes, weight_tes, color='tab:orange', label='Predicted')
    ax.set_xlabel('Conditional Parameter z', fontsize=14)
    ax.set_ylabel('weight', fontsize=14)
    ax.set_title(f'Weight of {K} Components', fontsize=14)
    customs = [pw_r[0], pw_tes[0]]
    ax.legend(customs, [pw_r[0].get_label(), pw_tes[0].get_label()], fontsize=10)
    fig.savefig('figs/weights.pdf')


    # figure. Diagonal Element vs conditional parameters
    fig, ax = plt.subplots(D, figsize=(6,D*1.7))
    for i in range(D):
        pde_r = ax[i].plot(param_cond_tes, covars_r[:,:,i,i], color='tab:blue', label='True')
        pde_tes = ax[i].plot(param_cond_tes, covars_tes[:,:,i,i], color='tab:orange', label='Predicted')
        ax[i].set_title(f'{i+1, i+1} Element of the Covariance Matrix', fontsize=12)
        ax[i].set_xlabel('Conditional Parameter z')
        customs = [pde_r[0], pde_tes[0]]
        ax[i].legend(customs, [pde_r[0].get_label(), pde_tes[0].get_label()], fontsize=10)
    plt.tight_layout() 
    fig.savefig('figs/covars.pdf')
    plt.show()



    # specific check
    # GMM parameters at a certain conditional parameter

    # figure. means, and some samples, and learned means at certain condtional parameter
    Nr = 10000
    cond_array = [9, 49, 89] # cond = 0.1, 0.5, 0.9
    param_cond_tes = torch.from_numpy(param_cond_tes)
    step = 50
    bins = 30
    label = [f'Dim. {i+1}' for i in range(D)]

    # noisy observation vs predition + noise
    for i in cond_array:
        
        noise  = np.zeros((Nr, D, D))
        data_r = np.zeros((Nr, D))
        data_t = np.zeros((Nr, D))
        
        for j in range(Nr):
            noise[j]     = noise_func(param_cond_tes[i], means_tes.shape[-1], sigma_d=sigma_d, sigma_l=sigma_l)
            data_r[j], _ = sample_func(weight_tes[i], means_tes[i], covars_tes[i], noise[j], 1)
            data_t[j]    = gmm.sample(param_cond_tes[i].unsqueeze(0), 1, torch.FloatTensor(noise[j]).unsqueeze(0)).squeeze().detach().numpy()
        '''
        noise     = noise_func(param_cond_tes[i], means_tes.shape[-1], sigma_d=sigma_d, sigma_l=simga_l)
        data_r, _ = sample_func(weight_tes[i], means_tes[i], covars_tes[i], noise, Nr)
        data_t    = gmm.sample(param_cond_tes[i].unsqueeze(0), Nr, torch.FloatTensor(noise).unsqueeze(0)).squeeze().detach().numpy()
        '''
        weight_t, means_t, covars_t = gmm(param_cond_tes[i].unsqueeze(0))
        '''
        fig, ax = plt.subplots()

        pm_r = ax.scatter(*means_tes[i][:,0:2].transpose(), label='True Means')
        pd_r = ax.scatter(*data_r[::step][:,0:2].transpose(), marker='.', color='tab:blue', label='Observations', alpha=0.2)
        #noise = noise[None,:].repeat(Nr,axis=0)
        lambda1, lambda2, theta = covars_ellipse(noise)
        add_ellipse(ax, data_r[:,0:2][::step], lambda1[::step], lambda2[::step], theta[::step], color='tab:blue')
        sns.kdeplot(x=data_r[:,0], y=data_r[:,1], color='tab:blue', alpha=0.5)
        
        pm_t = ax.scatter(*means_t.detach().numpy()[0][:,0:2].transpose(), label='Predicted Means')
        pd_t = ax.scatter(*data_t[::step][:,0:2].transpose(), marker='.', color='tab:orange', label='Predictions + Noise', alpha=0.2)
        lambda1, lambda2, theta = covars_ellipse(noise)
        add_ellipse(ax, data_t[:,0:2][::step], lambda1[::step], lambda2[::step], theta[::step], color='tab:orange')
        sns.kdeplot(x=data_t[:,0], y=data_t[:,1], color='tab:orange', alpha=0.5)

        ax.set_title('Noisy Observation vs Prediction with Noise')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        customs = [pm_r, pd_r, pm_t, pd_t,
                    Line2D([0], [0], marker='o', color='w',
                            markerfacecolor='k', markersize=5)]
        ax.legend(customs, [pm_r.get_label(), pd_r.get_label(), pm_t.get_label(), pd_t.get_label(),
                        f'Conditional z={(param_cond_tes[i].numpy()[0]):.2f}'], fontsize=10)
        fig.savefig(f'figs/NoisyComp{i+1}.pdf')
        '''
        contour_param_r = dict(linewidths=0.9, alpha=0.9, colors='tab:blue')
        contour_param_t = dict(linewidths=0.9, alpha=0.9, colors='tab:orange')
        figure = corner.corner(data_r, color='tab:blue', labels=label, label_kwargs=dict(fontsize=16), show_titles=True, title_kwargs={"fontsize": 14}, contour_kwargs=contour_param_r, bins=bins)
        corner.corner(data_t, fig=figure, color='tab:orange', labels=label, label_kwargs=dict(fontsize=16), contour_kwargs=contour_param_t, bins=bins)
        axes = np.array(figure.axes).reshape(D, D)
        for ax in figure.get_axes():
            ax.tick_params(axis='both', direction='in', labelsize=12)
        axes[1, -3].text(0.6, 0.8, f'Samples on NN vs Samples on Real Model,\n cond={param_cond_tes[i].numpy()[0]:.2f}', fontsize=25, horizontalalignment='center', c='k', weight='bold')
        axes[1, -3].text(0.6, 0.5, f'Noise Convolved Real Model', fontsize=25, horizontalalignment='center', c='tab:blue', weight='bold')
        axes[1, -3].text(0.6, 0.2, f'Noise Convolved NN Predictions', fontsize=25, horizontalalignment='center', c='tab:orange', weight='bold')
        figure.savefig(f'figs/noisyComp{i+1}.pdf')


    # noiseless data vs deconv data
    for i in cond_array:
        '''
        noise  = np.zeros((Nr, D, D))
        data_r = np.zeros((Nr, D))
        data_t = np.zeros((Nr, D))
        for j in range(Nr):
            data_r[j], _ = sample_func(weight_tes[i], means_tes[i], covars_tes[i], 1)
            data_t[j]    = gmm.sample(param_cond_tes[i].unsqueeze(0), 1).squeeze().detach().numpy()
        '''
        data_r, _ = sample_func(weight_tes[i], means_tes[i], covars_tes[i], N=Nr)
        data_t    = gmm.sample(param_cond_tes[i].unsqueeze(0), Nr).squeeze().detach().numpy()
        weight_t, means_t, covars_t = gmm(param_cond_tes[i].unsqueeze(0))
        '''
        fig, ax = plt.subplots()

        pm_r = ax.scatter(*means_tes[i][:,0:2].transpose(), label='True Means')
        pd_r = ax.scatter(*data_r[::step][:,0:2].transpose(), marker='.', color='tab:blue', label='Samples (on true)', alpha=0.2)
        sns.kdeplot(x=data_r[:,0], y=data_r[:,1], color='tab:blue', alpha=0.5)

        pm_t = ax.scatter(*means_t.detach().numpy()[0][:,0:2].transpose(), label='Predicted Means')
        pd_t = ax.scatter(*data_t[::step][:,0:2].transpose(), marker='.', color='tab:orange', label='Predictions', alpha=0.2)
        sns.kdeplot(x=data_t[:,0], y=data_t[:,1], color='tab:orange', alpha=0.5)

        ax.set_title('True Model vs Deconvolved Prediction')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        customs = [pm_r, pd_r, pm_t, pd_t,
                    Line2D([0], [0], marker='o', color='w',
                            markerfacecolor='k', markersize=5)]
        ax.legend(customs, [pm_r.get_label(), pd_r.get_label(), pm_t.get_label(), pd_t.get_label(),
                        f'Conditional z={(param_cond_tes[i].numpy()[0]):.2f}'], fontsize=10)
        fig.savefig(f'figs/cleanComp{i+1}.pdf')
        '''
        contour_param_r = dict(linewidths=0.9, alpha=0.9, colors='tab:blue')
        contour_param_t = dict(linewidths=0.9, alpha=0.9, colors='tab:orange')
        figure = corner.corner(data_r, color='tab:blue', labels=label, label_kwargs=dict(fontsize=16), show_titles=True, title_kwargs={"fontsize": 14}, contour_kwargs=contour_param_r, bins=bins)
        corner.corner(data_t, fig=figure, color='tab:orange', labels=label, label_kwargs=dict(fontsize=16), contour_kwargs=contour_param_t, bins=bins)
        axes = np.array(figure.axes).reshape(D, D)
        for ax in figure.get_axes():
            ax.tick_params(axis='both', direction='in', labelsize=12)
        axes[1, -3].text(0.6, 0.8, f'Samples on NN vs Samples on Real Model,\n cond={param_cond_tes[i].numpy()[0]:.2f}', fontsize=25, horizontalalignment='center', c='k', weight='bold')
        axes[1, -3].text(0.6, 0.5, f'Noise Convovled Real Model', fontsize=25, horizontalalignment='center', c='tab:blue', weight='bold')
        axes[1, -3].text(0.6, 0.2, f'Noise Convovled NN Predictions', fontsize=25, horizontalalignment='center', c='tab:orange', weight='bold')
        figure.savefig(f'figs/cleanComp{i+1}.pdf')


def KLdiv_figure(param_cond_t, weights_t, means_t, covars_t, data_t, noise_t, gmm):

    # demension of the data
    D = (data_t.shape)[-1]
    # reshape the data
    data_t = data_t.reshape(-1,D)
    # derive the trained parameters
    param_cond_t = param_cond_t.reshape(-1, 1)
    weight_tes, means_tes, covars_tes = gmm(param_cond_t)
    weight_tes = weight_tes.detach()
    means_tes  = means_tes.detach()
    covars_tes = covars_tes.detach()

    # log-likelihood on NN and real model
    noisy_covar = covars_tes + noise_t.reshape(-1, D, D)[:, None, ...]
    llkhs_tes = mvn(loc=means_tes,
                        covariance_matrix=noisy_covar
                        ).log_prob(data_t[:,None,:])
    llkh_tes = torch.logsumexp(llkhs_tes + np.log(weight_tes), -1).numpy()

    noisy_covar = torch.from_numpy(covars_t) + noise_t.reshape(-1, D, D)[:, None, ...]
    llkhs_r   = mvn(loc=torch.from_numpy(means_t),
                        covariance_matrix=noisy_covar
                        ).log_prob(data_t[:,None,:])
    llkh_r   = torch.logsumexp((llkhs_r  + np.log(weights_t)), -1).numpy()

    # KL divergence
    KL_Div  = llkh_r - llkh_tes

    # binned KL
    param_conds = np.arange(20)/20 + 1/40
    KL_Divs = np.zeros(20)
    for i in range(20):
        bln = (param_cond_t > (param_conds[i]-1/40)) * (param_cond_t <= param_conds[i]+1/40)
        KL_Divs[i] = KL_Div[bln.flatten().numpy()].mean()

    # figure. KL divergence vs conditional
    fig, ax = plt.subplots()
    ax.scatter(param_cond_t.flatten(), KL_Div, label='KL Divs on Training Set', marker='.', alpha=0.2)
    ax.plot(param_conds, KL_Divs, label='Binned KL Divergence', color='tab:orange')
    ax.set_xlabel('Conditional Parameter z')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence on Different Conditionals')
    ax.legend(fontsize=14)
    fig.savefig('figs/KLDiv.pdf')

    plt.show()