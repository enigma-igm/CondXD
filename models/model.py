import torch
import torch.nn as nn
import torch.distributions as dist
from torch.utils.data import WeightedRandomSampler
mvn = dist.multivariate_normal.MultivariateNormal

from IPython import embed


class GMMNet(nn.Module):
    """Neural network for Gaussian Mixture Model"""
    def __init__(self, 
                 n_components, 
                 data_dim, 
                 conditional_dim, 
                 vec_dim=128,
                 num_embedding_layers=3,
                 num_weights_layers=1,
                 num_means_layers=1,
                 num_covar_layers=1,
                 activation='PReLU',
                 w = 1e-6
                 ):
        super().__init__()
        self.n_components = n_components
        self.data_dim = data_dim
        self.conditional_dim = conditional_dim
        self.vec_dim = vec_dim
        self.num_embedding_layers = num_embedding_layers
        self.num_weights_layers = num_weights_layers
        self.num_means_layers = num_means_layers
        self.num_covar_layers = num_covar_layers
        self.activation = getattr(nn, activation)
        self.w = w
        
        embedding_layers = [nn.Linear(self.conditional_dim, self.vec_dim), self.activation()]
        for _ in range(num_embedding_layers - 1):
            embedding_layers += [nn.Linear(self.vec_dim, self.vec_dim), self.activation()]
        self.embedding_network = nn.Sequential(*embedding_layers)
       
        weights_layers = []
        for _ in range(num_weights_layers - 1):
            weights_layers += [nn.Linear(self.vec_dim, self.vec_dim), self.activation()]
        weights_layers += [nn.Linear(self.vec_dim, self.n_components), nn.Softmax(-1)]
        self.weights_network = nn.Sequential(*weights_layers)
        
        means_layers = []
        for _ in range(num_means_layers - 1):
            means_layers += [nn.Linear(self.vec_dim, self.vec_dim), self.activation()]
        means_layers += [nn.Linear(self.vec_dim, self.n_components*self.data_dim)]
        self.means_network = nn.Sequential(*means_layers)
        
        covar_layers = []
        for _ in range(num_covar_layers - 1):
            covar_layers += [nn.Linear(self.vec_dim, self.vec_dim), self.activation()]
        covar_layers += [nn.Linear(self.vec_dim, self.n_components*self.data_dim*(self.data_dim+1)//2)]
        self.covar_network = nn.Sequential(*covar_layers)


    def forward(self, conditional):
        '''
        data: shape(batch_size, D)
        '''
        
        B = conditional.shape[0] # batch size
        
        # embed conditional info
        embedding = self.embedding_network(conditional)
        
        # calculate weights
        weights = self.weights_network(embedding)
        
        # calculate means
        means = self.means_network(embedding)
        means = means.reshape(-1, self.n_components, self.data_dim)
        
        # calculate cholesky matrix
        covars_ele = self.covar_network(embedding)
        d_idx = torch.eye(self.data_dim).to(torch.bool) # diagonal index
        l_idx = torch.tril_indices(self.data_dim, self.data_dim, -1) # lower triangle index
        
        scale_tril = torch.zeros((B, self.n_components, self.data_dim, self.data_dim))
        log_diagonal = covars_ele[:, :self.n_components*self.data_dim].reshape(B, self.n_components, self.data_dim)
        scale_tril[:, :, d_idx] = torch.exp(log_diagonal)
        lower_tri = covars_ele[:, self.n_components*self.data_dim:].reshape(B, self.n_components, self.data_dim*(self.data_dim-1)//2)
        scale_tril[:, :, l_idx[0], l_idx[1]] = lower_tri
        
        # calculate covariance matrix
        covars = torch.matmul(scale_tril, scale_tril.transpose(-2, -1)) + torch.eye(self.data_dim)*1e-4
        
        return weights, means, covars


    def reg_loss(self, covars):
        '''
        regression loss.
        '''
        l = self.w * 1/torch.diagonal(covars, dim1=-1, dim2=-2)
        return l.sum(axis=(-1,-2))
    

    def log_prob_GMM(self, data, weights, means, covars, noise=None):

        if noise is None:
            noise = torch.zeros_like(covars)
        else:
            noise = noise[:, None, ...]  # add noise to all components

        noisy_covars = covars + noise
        
        log_resp = mvn(loc=means, covariance_matrix=noisy_covars).log_prob(data[:, None, :])
        
        log_resp += torch.log(weights)

        log_prob_GMM = torch.logsumexp(log_resp, dim=1)

        return log_prob_GMM


    def log_prob_conditional(self, data, conditional, noise=None):

        weights, means, covars = self.forward(conditional)

        log_prob = self.log_prob_GMM(data, weights, means, covars, noise=noise)

        
        return log_prob
        

    def loss(self, data, conditional, noise=None, regression=False):

        weights, means, covars = self.forward(conditional)

        log_prob_b = self.log_prob_GMM(data, weights, means, covars, noise=noise)
        
        if regression is False:
            train_loss = (-log_prob_b).sum()
        else:
            train_loss = (-log_prob_b + self.reg_loss(covars)).sum()

        return train_loss


    def sample(self, conditional, n_per_conditional=1, noise=None):
        
        weights, means, covars = self.forward(conditional)

        batchsize = conditional.shape[0]
        draw = list(WeightedRandomSampler(weights, n_per_conditional))
        means = means[:, draw][torch.eye(batchsize).to(torch.bool)]
        covars = covars[:, draw][torch.eye(batchsize).to(torch.bool)]

        if noise is None:
            noise = torch.zeros_like(covars)
        elif noise.dim() != covars.dim():  
            noise = noise[:, None, ...]  # add noise to all components
        
        noisy_covars = covars + noise
        
        data = mvn(loc=means, covariance_matrix=noisy_covars).sample()

        return data
        