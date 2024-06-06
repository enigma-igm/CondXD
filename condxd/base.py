import torch
import torch.nn as nn
import torch.distributions as dist

from IPython import embed

mvn = dist.multivariate_normal.MultivariateNormal

class CondXDBase(nn.Module):
    def __init__(self,
                 n_Gaussians,
                 sample_dim,
                 conditional_dim,
                 vec_dim=128,
                 num_embedding_layers=3,
                 num_mixcoef_layers=1,
                 num_means_layers=1,
                 num_covar_layers=1,
                 activation='PReLU',
                 w=1e-6
                 ):
        
        """Implementation of a conditional extreme deconvolution using GMM and
        Neural network (NN) with Pytorch. For the stem-branch structure of the
        NN architecture see paper.
        
        Parameters
        ----------
        n_Gaussians : int 
            Number of Gaussian components.

        sample_dim : int
            Number of dimensions for samples.

        conditional_dim : int
            Number of dimensions for conditional.

        vec_dim : int, (optional, default=128)
            Number of dimensions for outputs of linear layers.

        num_embedding_layers : int (optional, default=3)
            Number of embedding linear (FC) layers. The NN architecture has a
            stem-branch structure, and this is for the stem part.

        num_mixcof_layers : int, (optional, default=1)
            Number of mixing coefficient branch linear (FC) layers.

        num_means_layers : int (optional, default=1) 
            Number of means branch layers.

        num_covar_layers : int (optional, default=1): 
            Number of covariance matrix branch linear (FC) layers.

        activation : str (optional='PReLU'): 
            Type of activation function after the linear (FC) layers, except
            for the last linear layer. Note that softmax and exponential
            functions can follow the last linear layer to produce mixing 
            coefficients and Cholesky decomposition factor of covariance matrix
            respectively.

        w : float (optional, default=1e-6) 
            Coefficient for regression loss. The regression loss is added to
            the NN loss (negative likelihood), computed as the sum of the 
            inverse of the diagonal elements for all the covariance matrice
            (all Gaussian components).
        """
        
        super(CondXDBase, self).__init__()

        self.n_Gaussians = n_Gaussians
        self.sample_dim = sample_dim
        self.conditional_dim = conditional_dim
        self.vec_dim = vec_dim
        self.num_embedding_layers = num_embedding_layers
        self.num_mixcoef_layers = num_mixcoef_layers
        self.num_means_layers = num_means_layers
        self.num_covar_layers = num_covar_layers
        self.activation = getattr(nn, activation)
        self.w = w

        # The NN layers
        # stem part
        embedding_layers = [
            nn.Linear(self.conditional_dim, self.vec_dim), 
            self.activation()
        ]

        for _ in range(num_embedding_layers - 1):
            embedding_layers += [
                nn.Linear(self.vec_dim, self.vec_dim), 
                self.activation()
            ]

        self.embedding_network = nn.Sequential(*embedding_layers)

        # mixing coefficients branch part
        mixcoef_layers = []
        for _ in range(num_mixcoef_layers - 1):
            mixcoef_layers += [
                nn.Linear(self.vec_dim, self.vec_dim), 
                self.activation()
            ]

        mixcoef_layers += [
            nn.Linear(self.vec_dim, self.n_Gaussians), 
            nn.Softmax(-1)
        ]

        self.mixcoef_network = nn.Sequential(*mixcoef_layers)

        # means branch part
        means_layers = []
        for _ in range(num_means_layers - 1):
            means_layers += [
                nn.Linear(self.vec_dim, self.vec_dim), 
                self.activation()
            ]

        means_layers += [
            nn.Linear(self.vec_dim, self.n_Gaussians * self.sample_dim)
        ]

        self.means_network = nn.Sequential(*means_layers)

        # covariance Cholesky decomposition branch part
        covar_layers = []
        for _ in range(num_covar_layers - 1):
            covar_layers += [
                nn.Linear(self.vec_dim, self.vec_dim), 
                self.activation()
            ]

        covar_layers += [
            nn.Linear(self.vec_dim, 
                    self.n_Gaussians * self.sample_dim * (self.sample_dim + 1) // 2)
        ]

        self.covar_network = nn.Sequential(*covar_layers)

    def forward(self, conditional):
        """Defines the forward pass of CondXD.

        Parameters
        ----------
        conditional : torch.Tensor
            The conditional (input) of CondXD, with shape (batch_size, 
            conditional_dim)

        Returns
        -------
        mixcoef : torch.Tensor
            The mixing coefficients for each component of the mixture model,
            with shape (batch_size, n_Gaussians).

        means : torch.Tensor
            The means of each Gaussian component in the mixture model,
            with shape (batch_size, n_Gaussians, sample_dim).

        covars : torch.Tensor
            The covariances of each Gaussian component in the mixture model,
            with shape (batch_size, n_Gaussians, sample_dim, sample_dim). 
            Built from Cholesky decomposition to ensure positive definiteness.
        """

        batch_size = conditional.shape[0]  # batch size

        # embed conditional info
        embedding = self.embedding_network(conditional)

        # calculate mixcoef
        mixcoef = self.mixcoef_network(embedding)

        # calculate means
        means = self.means_network(embedding).reshape(
            -1, self.n_Gaussians, self.sample_dim)

        # calculate cholesky decomposition of covariance matrix
        covars_elements = self.covar_network(embedding)

        # diagonal index
        d_idx = torch.eye(self.sample_dim).to(torch.bool)
        # lower triangle index
        l_idx = torch.tril_indices(self.sample_dim, self.sample_dim, -1)

        Cholesky = torch.zeros(
            (batch_size, self.n_Gaussians, self.sample_dim, self.sample_dim))
        # first self.n_Gaussians * self.sample_dim elements are on diagonal
        log_diagonal = covars_elements[:, :self.n_Gaussians * self.sample_dim
                                    ].reshape(batch_size, self.n_Gaussians, 
                                              self.sample_dim)
        Cholesky[:, :, d_idx] = torch.exp(log_diagonal)  # ensure positive
        # later elements are on lower triangle
        lower_tri = covars_elements[:, self.n_Gaussians * self.sample_dim:
                                    ].reshape(batch_size, self.n_Gaussians,
                                            self.sample_dim * (self.sample_dim - 1) // 2)
        Cholesky[:, :, l_idx[0], l_idx[1]] = lower_tri

        # calculate covariance matrix
        covars = torch.matmul(Cholesky, Cholesky.transpose(-2, -1))
        covars += torch.eye(self.sample_dim) * 1e-4  # ensure positive definiteness

        return mixcoef, means, covars

    def reg_loss(self, covars):
        """Computes a regularization loss term to be added to the NN loss
        (negative likelihood) to prevent singularity problem, computed as the
        sum of the inverse of the diagonal elements of all the covariance 
        matrice. The sum is also multiplied with a coefficient, self.w, an
        input to the CondXD or CondXDBase class.

        Parameters
        ----------
        covars : torch.Tensor
            The covariances of each Gaussian component in the mixture model,
            built from Cholesky decomposition to ensure positive definiteness.
            Expected shape is with shape (batch_size, n_Gaussians, sample_dim, 
            sample_dim).

        Returns
        -------
        reg_loss : torch.Tensor:
            The regularization loss to be added to the NN loss, computed as the
            sum of the inverse of the diagonal elements of all the covariance 
            matrice. 1D with length (bathszie). The result is a 1D tensor with 
            length equal to the batch size (`batch_size`), where each element 
            represents the regularization loss for a corresponding sample in 
            the batch.
        """
        
        l = self.w * 1 / torch.diagonal(covars, dim1=-1, dim2=-2)

        return l.sum(axis=(-1, -2))

    def log_prob_GMM(self, sample, mixcoef, means, covars, noise=None):
        """Compute the log likelihood of samples given GMM parameters. A 
        general algorithm that suits all samples and GMM. Also base for 
        computing log likelihood of sample given conditional.

        Parameters
        ----------
        sample : torch.Tensor
            Samples that needs to compute log likelihood, with shape 
            (batch_size, sample_dim).

        mixcoef : torch.Tensor
            The mixing coefficients for each component of the GMM, with shape
            (batch_size, n_Gaussians).

        means : torch.Tensor
            The means of each Gaussian component in the mixture model,
            with shape (batch_size, n_Gaussians, sample_dim).

        covars : torch.Tensor
            The covariances of each Gaussian component in the mixture model,
            with shape (batch_size, n_Gaussians, sample_dim, sample_dim). 
            Built from Cholesky decomposition to ensure positive definiteness.

        noise : torch.Tensor (optional, default=None)
            Gaussian noise covariance matrix of every sample, with shape 
            (batch_size, sample_dim, sample_dim). If None, no noise is added. 

        Returns
        -------
        log_prob_GMM : torch.Tensor
            The log likelihood of samples given GMM parameters. The result is 
            a 1D tensor with length equal to the batch size (`batch_size`), 
            where each element represents the log likelihood loss for a 
            corresponding sample in the batch.
        """

        if noise is None:
            noise = torch.zeros_like(covars)
        else:
            noise = noise[:, None, ...]  # Add noise to all Gaussian components

        noisy_covars = covars + noise

        # Strengthen symmetry
        noisy_covars = 0.5 * (noisy_covars + noisy_covars.transpose(-1, -2))

        # Log likelihood of every Gaussian
        log_resp = mvn(loc=means, covariance_matrix=noisy_covars).log_prob(
            sample[:, None, :]
        )

        # Multiply with mixing coefficients
        log_resp += torch.log(mixcoef)

        # Add up the weighted likelihood
        log_prob_GMM = torch.logsumexp(log_resp, dim=1)

        return log_prob_GMM

    def log_prob_conditional(self, sample, conditional, noise=None):
        """
        Compute the log likelihood of samples using a GMM predicted by CondXD, 
        which takes the conditionals of the samples as input.

        log likelihood = log ( p (noisy sample | conditional) )

        Parameters
        ----------
        sample : torch.Tensor
            The sample tensor for which log likelihood is to be computed, with
            shape (batch_size, sample_dim).
        
        conditional : torch.Tensor
            The conditional input tensor used to generate GMM parameters via 
            CondXD, with shape (batch_size, conditional_dim).
        
        noise : torch.Tensor (optional, default=None)
            Gaussian noise covariance matrix of every sample, with shape 
            (batch_size, sample_dim, sample_dim). If None, no noise is added.

        Returns
        -------
        log_prob : torch.Tensor
            The computed log likelihood of samples `sample' given `conditional',
            with shape (batch_size, 1). In the output, each element represents 
            the log likelihood loss for a corresponding sample in the batch.
        """
        sample = torch.Tensor(sample)
        conditional = torch.Tensor(conditional)
        noise = torch.Tensor(noise)

        mixcoef, means, covars = self.forward(conditional)

        log_prob = self.log_prob_GMM(
            sample, mixcoef, means, covars, noise=noise
        )

        return log_prob

    def loss(self, conditional, sample, noise=None, regularization=False):
        """
        Computes the training loss, which can be either the negative log 
        likelihood of the samples under the GMM parameters predicted by CondXD
        or an augmented loss including a regularization term.

        Parameters
        ----------
        conditional : torch.Tensor
            The input conditional tensor used to generate GMM parameters via
            CondXD, with shape (batch_size, conditional_dim). 

        sample : torch.Tensor
            The sample tensor for which the loss is computed, with shape 
            (batch_size, sample_dim). 

        noise : torch.Tensor (optional, default=None)
            Gaussian noise covariance matrix of every sample, with shape 
            (batch_size, sample_dim, sample_dim). If None, no noise is added.

        regularization : bool (optional, default=False)
            A flag indicating whether to compute and add the regularization 
            loss instead of using soly the default negative log likelihood 
            loss. If True, the loss includes a regularization term computed
            as the sum of inverse of the diagonal elements of all the
            covariance matrice.

        Returns
        -------
        train_loss : torch.Tensor
            The computed training loss. If `regularization` is False, this is
            the negative log likelihood loss averaged over all samples in the 
            batch. If `regression` is True, the loss includes an additional 
            regularization term and is also averaged over all samples.
        """

        mixcoef, means, covars = self.forward(conditional)

        log_prob_b = self.log_prob_GMM(
            sample, mixcoef, means, covars, noise=noise
        )

        if regularization is False:
            train_loss = (-log_prob_b).mean()
        else:
            train_loss = (-log_prob_b + self.reg_loss(covars)).mean()

        return train_loss