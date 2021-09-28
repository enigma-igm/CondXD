import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import corner


import sys
# TODO: remove this once we make a package
sys.path.append("..") # Adds higher directory to python modules path.
from models.model import GMMNet




def plot_digits(sample_digit=0):
    # sample 44 new points from the data
    with torch.no_grad():
        context = sample_digit * torch.ones(44, 1)
        context = (context - labels_mean)/labels_std  # normalize context
        xd_data = xd_gmm.sample(context)
        basic_data = basic_gmm.sample(context)
        
    xd_data = pca.inverse_transform((xd_data*data_std + data_mean).numpy())
    basic_data = pca.inverse_transform((basic_data*data_std + data_mean).numpy())

    # turn data into a 4x11 grid
    xd_data = xd_data.reshape((4, 11, -1))
    basic_data = basic_data.reshape((4, 11, -1))
    noisy_data = pca.inverse_transform((data*data_std + data_mean).numpy())
    noisy_data = noisy_data[:44].reshape((4, 11, -1))
    real_data = digits[:44]
    real_data = pca.inverse_transform(pca.transform(real_data.numpy())).reshape((4, 11, -1))

    # plot real digits and resampled digits
    fig, ax = plt.subplots(19, 11, subplot_kw=dict(xticks=[], yticks=[]))
    for j in range(11):
        ax[4, j].set_visible(False)
        ax[9, j].set_visible(False)
        ax[14, j].set_visible(False)
        for i in range(4):
            im = ax[i, j].imshow(real_data[i, j].reshape((28, 28)),
                                cmap=plt.cm.binary, interpolation='nearest')
            im.set_clim(0, 16)
            im = ax[i + 5, j].imshow(noisy_data[i, j].reshape((28, 28)),
                                cmap=plt.cm.binary, interpolation='nearest')
            im.set_clim(0, 16)
            im = ax[i + 10, j].imshow(basic_data[i, j].reshape((28, 28)),
                                    cmap=plt.cm.binary, interpolation='nearest')
            im.set_clim(0, 16)
            im = ax[i + 15, j].imshow(xd_data[i, j].reshape((28, 28)),
                                    cmap=plt.cm.binary, interpolation='nearest')
            im.set_clim(0, 16)

    ax[0, 5].set_title('Selection from the input data')
    ax[5, 5].set_title(f'Noisy data (used for training)')
    ax[10, 5].set_title(f'Conditional samples drawn from basic model (context={int(sample_digit)})')
    ax[15, 5].set_title(f'Conditional samples drawn from xd model (context={int(sample_digit)})')


def corner_plots(sample_digit=0, n_samples=5000):
    with torch.no_grad():
        context = sample_digit * torch.ones(n_samples, 1)
        context = (context - labels_mean)/labels_std  # normalize context
        xd_data = xd_gmm.sample(context)*data_std + data_mean
        basic_data = basic_gmm.sample(context)*data_std + data_mean
    
    noisy_data = (data*data_std + data_mean)

    real_data = digits[:n_samples]
    real_data = pca.transform(real_data.numpy())
    
    corner.corner(real_data, titles=['Real', ''], range=((-4, 4), (-4, 4)))
    plt.title("true model")
    corner.corner(noisy_data.squeeze(1).numpy(), titles=['Noisy', ''], range=((-4, 4), (-4, 4)))
    plt.title("noisy model")
    corner.corner(basic_data.squeeze(1).numpy(), titles=['Basic', ''], range=((-4, 4), (-4, 4)))
    plt.title("noisy gmm")
    corner.corner(xd_data.squeeze(1).numpy(), titles=['xd', ''], range=((-4, 4), (-4, 4)))
    plt.title("xd gmm")
    
    

# settings
digit_of_interest = 3
n_clusters = 5    # number of gaussian components 
n_pca_components = 2  # dimension of the PCA subspace
sigma = 1e1

# load the data
mnist = MNIST("data/", train=True, download=True)

digits = mnist.data.flatten(-2, -1)
labels = mnist.targets

digits = digits[labels == digit_of_interest]
labels = labels[labels == digit_of_interest]


# project the 28*28-dimensional data to a lower dimension
pca = PCA(n_components=n_pca_components, whiten=True)
data = pca.fit_transform(digits.numpy())
# add homoscedastic noise to the data

data = torch.tensor(data).float()
labels = labels.unsqueeze(1).float()
# normalize labels and data
labels_mean = 0
labels_std = 1
labels = (labels - labels_mean)/labels_std

data = data + torch.randn_like(data)*sigma
data_mean = data.mean(0)
data_std = data.std(0)
data = (data - data_mean)/data_std
normalized_sigma = sigma/data_std
# synthesize and add homoscedastic noise
noise_covar = torch.diag(torch.ones(n_pca_components))*(normalized_sigma**2)
noise_covars = torch.stack([noise_covar for _ in range(data.shape[0])])

# kmeans to classify each data point
kmeans = KMeans(n_clusters=n_clusters, random_state=12321).fit(data)

# initialization
xd_gmm = GMMNet(n_components=n_clusters, 
             data_dim=n_pca_components, 
             conditional_dim=1, 
             cluster_centroids=torch.tensor(kmeans.cluster_centers_).float(), 
             vec_dim=128,
             num_embedding_layers=3,
             num_weights_layers=1,
             num_means_layers=1,
             num_covar_layers=1)
# initialization
basic_gmm = GMMNet(n_components=n_clusters, 
             data_dim=n_pca_components, 
             conditional_dim=1, 
             cluster_centroids=torch.tensor(kmeans.cluster_centers_).float(), 
             vec_dim=128,
             num_embedding_layers=3,
             num_weights_layers=1,
             num_means_layers=1,
             num_covar_layers=1)


learning_rate = 3e-4
xd_optimizer = torch.optim.Adam(xd_gmm.parameters(), lr=learning_rate, weight_decay=0.001)
xd_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(xd_optimizer, factor=0.4, patience=2)
basic_optimizer = torch.optim.Adam(basic_gmm.parameters(), lr=learning_rate, weight_decay=0.001)
basic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(basic_optimizer, factor=0.4, patience=2)

batch_size = 128
train_loader = DataLoader(TensorDataset(data, labels, noise_covars), batch_size=batch_size, shuffle=True)

epoch = 50
try:
    for n in range(epoch):
        xd_train_loss = 0
        basic_train_loss = 0
        for i, (dig, lab, noise) in enumerate(train_loader):
            xd_optimizer.zero_grad()
            basic_optimizer.zero_grad()
            
            xd_loss = -xd_gmm.log_prob_b(dig, lab, noise=noise).mean()
            xd_train_loss += xd_loss.item()
            
            basic_loss = -basic_gmm.log_prob_b(dig, lab).mean()
            basic_train_loss += basic_loss.item()

            # backward and update parameters
            xd_loss.backward()
            xd_optimizer.step()
            
            basic_loss.backward()
            basic_optimizer.step()
        
        xd_train_loss = xd_train_loss / dig.shape[0]
        basic_train_loss = basic_train_loss / dig.shape[0]
        print(f'Epoch {n+1}: Basic loss: {basic_train_loss:.3f}   XD loss: {xd_train_loss:.3f}')
        xd_scheduler.step(xd_train_loss)
        basic_scheduler.step(basic_train_loss)

except KeyboardInterrupt:
    pass

plot_digits(sample_digit=digit_of_interest)
corner_plots(sample_digit=digit_of_interest)
plt.show()
