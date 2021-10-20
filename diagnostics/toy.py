import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import numpy as np
import torch
import corner

from models.model import mvn

def cornerplots(data_tes, output_tes, labels, bins, ranges, K, noisy=False):
    """Making comparison corner plots of the test set.

    Args:
        data_tes (narray): the test data set.
        output_tes (narray): the NN's output set.
        labels (list): labels of each band (dimension).
        bins (int): the bins of the 2Dhist.
        ranges (list, shape nx2): the ranges of each band for plotting.
        K (int): the number of Gaussian components
        noisy (bool, optional): if the output data are samples from NN with noise. Defaults to False.
    """
    import corner
    fig = corner.corner(data_tes, labels=labels, label_kwargs={"fontsize": 20}, bins=bins, range=ranges)
    corner.corner(output_tes, fig=fig, color='tab:blue', labels=labels, label_kwargs={"fontsize": 20}, bins=bins, range=ranges)
    axes = np.array(fig.axes).reshape((data_tes.shape[-1], data_tes.shape[-1]))
    for ax in fig.get_axes():
        ax.tick_params(axis='both', which='major', direction='in', length=5, labelsize=15, width=2)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
    axes[0, -3].text(0.6, 0.4, f'Relative Flux', c='k',
                fontsize=20, horizontalalignment='center', weight='bold')
    axes[1, -3].text(0.6, 1.0, f'Test Set', c='k',
                fontsize=20, horizontalalignment='center', weight='bold')
    if noisy is False:
        tag = 'Deconvolved'
    if noisy is True:
        tag = 'Reconvolved'
    axes[1, -3].text(0.6, 0.8, f'Samples on NN ({tag})', c='tab:blue',
                fontsize=20, horizontalalignment='center', weight='bold')
    if noisy is False:
        tag = 'clean'
    if noisy is True:
        tag = 'noisy'
    fig.savefig(f'figs/{tag}_relative_flux_K{K}.pdf')
    plt.show()



def all_figures(K,
                D,
                train_loader_tes,
                gmm
                ):
    """All figures to show the performances of our network.

    Args:
        K (int): Number of Gaussian components (how many Gaussians).
        D (int): Dimension of data.
        train_loader_tes (pytorch dataloader): The test set.
        gmm (class): Neural network for Gaussian Mixture Model. See model/model.py
    """
    
    #All figures to show the performances of our network.
    data_tes = train_loader_tes.dataset.tensors[1].numpy()


    # sampling from trained model with noise
    output_tes = torch.Tensor([])
    for i, (f_J_i, _, err_r_i) in enumerate(train_loader_tes):
        f_J_i = torch.log(f_J_i)
        output_tes = torch.cat((output_tes, gmm.sample(f_J_i, 1, err_r_i).squeeze()))
    output_tes = output_tes.reshape(-1, D).numpy()
        
    # the corner plot of the relative fluxes
    bins=50
    labels = ['$f_z$','$f_Y$', '$f_H$','$f_{Ks}$','$f_{W1}$','$f_{W2}$']
    ranges = [(-0.5,1.),(-0.7,1.4),(-1.2,3.1),(-1.6,5.),(-2.5,8.3),(-4,12)]
    cornerplots(data_tes, output_tes, labels, bins, ranges, K, noisy=False)



    # sampling from trained model without noise
    output_tes = torch.Tensor([])
    for i, (f_J_i, _, _) in enumerate(train_loader_tes):
        f_J_i = torch.log(f_J_i)
        output_tes = torch.cat((output_tes, gmm.sample(f_J_i, 1).squeeze()))
    output_tes = output_tes.reshape(-1, D).numpy()
    
    # the corner plot of the relative fluxes
    bins=50
    labels = ['$f_z$','$f_Y$', '$f_H$','$f_{Ks}$','$f_{W1}$','$f_{W2}$']
    ranges = [(-0.5,1.),(-0.7,1.4),(-1.2,3.1),(-1.6,5.),(-2.5,8.3),(-4,12)]
    cornerplots(data_tes, output_tes, labels, bins, ranges, K, noisy=True)



    '''
    fig, ax = plt.subplots()
    pd_t = ax.scatter(*data_tes_plot[:,0:2].transpose(), marker='.', color='grey', alpha=0.5, label='Training Set')
    pd_k = ax.scatter(*means0_t[:,0:2].transpose(), s=80, marker='.', color='tab:orange', label='kmeans centroids')
    ax.set_title('Training Set', fontsize=14)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.legend(fontsize=14)
    fig.savefig('figs/trainingset.pdf')
    '''