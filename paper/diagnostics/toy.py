import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from astropy.io import fits
import numpy as np
import torch
import corner
import imageio
import copy


from deprecate.model import mvn

def cornerplots(data_tes, output_tes, labels, bins, ranges, anno, name, noisy=False):
    """Making comparison corner plots of the test set.

    Args:
        data_tes (narray): the test data set.
        output_tes (narray): the NN's output set.
        labels (list): labels of each band (dimension).
        bins (int): the bins of the 2Dhist.
        ranges (list, shape nx2): the ranges of each band for plotting.
        anno (string): annotation of the plot.
        name (string): the specific name of the plot of relative fluxes.
        noisy (bool, optional): if the output data are samples from NN with noise. Defaults to False.
    """
    fig = corner.corner(data_tes, labels=labels, label_kwargs={"fontsize": 25}, bins=bins, range=ranges)
    corner.corner(output_tes, fig=fig, color='r', labels=labels, label_kwargs={"fontsize": 25}, bins=bins, range=ranges, alpha=0.7)
    axes = np.array(fig.axes).reshape((data_tes.shape[-1], data_tes.shape[-1]))
    for ax in fig.get_axes():
        ax.tick_params(axis='both', which='major', direction='in', length=5, labelsize=15, width=2)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
    axes[0, -3].text(0.6, 0.4, anno, c='k',
                fontsize=20, horizontalalignment='center', weight='bold')
    axes[1, -3].text(0.6, 1.0, f'Test Set', c='k',
                fontsize=20, horizontalalignment='center', weight='bold')
    if noisy is False:
        tag = 'Deconvolved'
    if noisy is True:
        tag = 'Reconvolved'
    axes[1, -3].text(0.6, 0.8, f'Samples on NN ({tag})', c='r',
                fontsize=20, horizontalalignment='center', weight='bold')
    if noisy is False:
        tag = 'clean'
    if noisy is True:
        tag = 'noisy'
    fig.savefig(f'figs/{tag}_relative_flux_'+name+'.png')
    


def make_gif(Jbin_len,K,tag,save_name):
    # Create a gif of the relative flux contours
    import imageio
    folder='figs/'
    files = [folder+f'{tag}_relative_flux_d_J{i:d}_K{K:d}.png'
                 for i in range(Jbin_len)]
    images = [imageio.imread(file) for file in files]
    imageio.mimwrite(folder+save_name+'.gif', images, fps=1.5)

def f2mag(f):
    return 22.5 - 2.5 * np.log10(f)

def mag2f(mag):
    return 10**((22.5 - mag) / 2.5)



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

    # parameters for plotting
    bins=25
    labels = ['$f_z$','$f_Y$', '$f_H$','$f_{Ks}$','$f_{W1}$','$f_{W2}$']
    ranges = [(-0.5,1.),(-0.7,1.4),(-1.2,3.1),(-1.6,5.),(-2.5,8.3),(-4,12)]
    # parameters of plots in J band flux bins
    '''
    Jbin_len = 5
    Jbin_l = torch.linspace(1, 5, Jbin_len)
    Jbin_r = Jbin_l + 4/(Jbin_len-1)
    Jbin_r[-1] = 999
    '''
    path = 'data/'
    file_name = 'bin_edges_stars.fits'
    file = fits.open(path+file_name)
    edges = copy.deepcopy(file[1].data)
    file.close()
    Jbin_l = edges['ed_l']
    Jbin_r = edges['ed_h']
    Jbin_len = len(Jbin_l)
    
    
    #All figures to show the performances of our network.

    # sampling from trained model with noise
    f_J_tes = torch.Tensor([])
    data_tes = torch.Tensor([]) 
    output_tes = torch.Tensor([])
    for i, (f_J_i, data_i, err_r_i) in enumerate(train_loader_tes):
        f_J_tes = torch.cat((f_J_tes, f_J_i))
        data_tes = torch.cat((data_tes, data_i))
        output_tes = torch.cat((output_tes, gmm.sample(f_J_i, 1, err_r_i)))
    data_tes = data_tes.numpy()
    output_tes = output_tes.reshape(-1, D).numpy()
        
    # the corner plot of the relative fluxes
    name = f'd_K{K:d}'
    cornerplots(data_tes, output_tes, labels, bins, ranges, '', name, noisy=True)
    
    # the corner plot of the relative fluxes in each J band flux bin
    for i in range(Jbin_len):
        bln = (torch.exp(f_J_tes)<mag2f(Jbin_l[i])) & (torch.exp(f_J_tes)>=mag2f(Jbin_r[i]))
        bln = bln.numpy().flatten()
        data_tes_i = data_tes[bln]
        output_tes_i = output_tes[bln]
        anno = f'{Jbin_l[i]:.1f}<$J$<{Jbin_r[i]:.1f}'
        name = f'd_J{i:d}_K{K:d}'
        cornerplots(data_tes_i, output_tes_i, labels, bins, ranges, anno, name, noisy=True)

    tag='noisy'
    save_name = f'noisy_relative_f_d_K{K:d}'
    make_gif(Jbin_len, K, tag, save_name)
    plt.close()
    


    # shuffle the test set
    f_J_tes = torch.Tensor([])
    data_tes = torch.Tensor([])
    # sampling from trained model without noise
    output_tes = torch.Tensor([])
    for i, (f_J_i, data_i, _) in enumerate(train_loader_tes):
        f_J_tes = torch.cat((f_J_tes, f_J_i))
        data_tes = torch.cat((data_tes, data_i))
        output_tes = torch.cat((output_tes, gmm.sample(f_J_i, 1)))
    data_tes = data_tes.numpy()
    output_tes = output_tes.reshape(-1, D).numpy()

    # the corner plot of the relative fluxes
    name = f'd_K{K:d}'
    cornerplots(data_tes, output_tes, labels, bins, ranges, '', name, noisy=False)
    
    # the corner plot of the relative fluxes
    name = f'd_K{K:d}'
    cornerplots(data_tes, output_tes, labels, bins, ranges, '', name, noisy=False)
    # the corner plot of the relative fluxes in each J band flux bin
    for i in range(Jbin_len):
        bln = (torch.exp(f_J_tes)<mag2f(Jbin_l[i])) & (torch.exp(f_J_tes)>=mag2f(Jbin_r[i]))
        bln = bln.numpy().flatten()
        data_tes_i = data_tes[bln]
        output_tes_i = output_tes[bln]
        anno = f'{Jbin_l[i]:.1f}<$J$<{Jbin_r[i]:.1f}'
        name = f'd_J{i:d}_K{K:d}'
        cornerplots(data_tes_i, output_tes_i, labels, bins, ranges, anno, name, noisy=False)

    tag='clean'
    save_name = f'clean_relative_f_d_K{K:d}'
    make_gif(Jbin_len, K, tag, save_name)
    plt.close()
    



