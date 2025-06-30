import torch
import numpy as np

import matplotlib.pyplot as plt

from paper.model import GMMNet
from diagnostics.plots_exp import density_comp, \
    deconv_comp, get_ranges

def density_comp_3(data_deconvolved, data_clean, data_noisy, label, nbins, conditional, path, ranges=None):
    """_summary_

    Args:
        data_deconvolved (_type_): _description_
        data_clean (_type_): _description_
        data_noisy (_type_): _description_
        label (_type_): _description_
        nbins (_type_): _description_
        conditional (_type_): _description_
        path (_type_): _description_
        ranges (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    import corner
    
    label_deconvolved = 'Deconvolution'
    label_clean = 'Noiseless Underlying'
    label_noisy = 'Noisy'
    D = data_deconvolved.shape[-1]

    figure = corner.corner(data_noisy, bins=nbins, range=ranges,
                        color='darkorange', labels=label, label_kwargs=dict(fontsize=16))
    if ranges is None:
        ranges = get_ranges(figure) # keeping the same range and bins
    corner.corner(data_clean, bins=nbins, range=ranges, fig=figure, 
                        color='black', labels=label, label_kwargs=dict(fontsize=16))
    corner.corner(data_deconvolved, bins=nbins, range=ranges, fig=figure, 
                        color='tab:red', hist_kwargs=dict(alpha=0.85),
                        contour_kwargs=dict(alpha=0.75),
                        labels=label, label_kwargs=dict(fontsize=16))
    
    
    axes = np.array(figure.axes).reshape(D, D)
    for ax in figure.get_axes():
        ax.tick_params(axis='both', direction='in', labelsize=12)
    
    # setting annotates
    axes[1, -3].text(0.6, 1.3, 'cond $\mathbf{c}$'+f'={conditional:.2f}',
                    fontsize=30, horizontalalignment='center', c='black', weight='bold')
    axes[1, -3].text(0.6, 0.8, label_noisy,
                    fontsize=25, horizontalalignment='center', c='darkorange', weight='bold')
    axes[1, -3].text(0.6, 0.5, label_clean,
                    fontsize=25, horizontalalignment='center', c='k', weight='bold')
    axes[1, -3].text(0.6, 0.2, label_deconvolved,
                    fontsize=25, horizontalalignment='center', c='tab:red', weight='bold')
    figure.savefig(path+f'Comp_{conditional:.2f}.pdf')

    plt.close(figure)


# components, dimension of data, and dimension of conditional
K, D, D_cond = 10, 7, 1

# load training and validation data and parameters
seed_list = [9, 11, 13, 15, 17, 19, 21, 23, 25, 27]
for seed in seed_list:

    param_path = f'params/experiment/seed{seed}/'
    fig_path = f'figs/experiment/seed{seed}/'

    gmm = GMMNet(K, D, conditional_dim=1)

    gmm.load_state_dict(torch.load(param_path+f'params.pkl'))

    # corner plots parameters
    c_list = [0.1, 0.5, 0.9] # cond = 0.1, 0.5, 0.9
    for c in c_list:

        label = [f'Dim. {i+1}' for i in range(D)]
        bins = 25

        data_r_noisy = np.load(fig_path+f'data_r_noisy_{c:.2f}.npy')
        data_r_clean = np.load(fig_path+f'data_r_clean_{c:.2f}.npy')
        data_p_clean = np.load(fig_path+f'data_p_clean_{c:.2f}.npy')

        # cornerplots noisy
        conditional = c
        # conerplots noisy distribution vs underlying distribution vs deconvolution
        density_comp_3(data_p_clean, data_r_clean, data_r_noisy, 
                        label, bins, conditional, path=fig_path)
        