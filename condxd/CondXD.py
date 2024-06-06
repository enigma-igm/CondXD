import os
import copy

import numpy as np

import torch
import torch.distributions as dist
from torch import multinomial
from torch.utils.data import DataLoader, TensorDataset
from .base import CondXDBase

from IPython import embed

__all__ = ['CondXD']

mvn = dist.multivariate_normal.MultivariateNormal

class CondXD(CondXDBase):

    """
    A derived class from CondXDBase for specific conditional density estimation
    tasks. It incorporates advanced functionalities such as data loading, custom 
    optimizer and scheduler configurations, training, validation, and testing
    routines, as well as sampling. 

    This class is designed to be flexible, allowing easy modifications to the
    optimization process and providing utilities for efficiently managing
    and evaluating generative models.

    Parameters
    ----------
    n_Gaussians : int
        The number of Gaussian components to be used in the mixture model.

    sample_dim : int
        The dimensionality of the sample data.

    conditional_dim : int
        The dimensionality of the conditional data.

    output_path : str (optional, default=None)
        The path where the CondXD model is to be saved.
        If not provided, the model is not saved by default.

    Attributes
    ----------
    model : CondXDBase
        An instance of the CondXDBase model configured with the specified
        number of Gaussians, sample dimensionality, and conditional
        dimensionality. 

    optimizer_params : dict
        The parameters for optimizer used to train the model. Defaults to 
        learning rate lr=1e-3 and weight decay weight_decay=0.001.

    optimizer : torch.optim.Optimizer
        The optimizer used for training the model. Defaults to Adam optimizer
        with pre-configured learning rate = 1e-3 and weight decay = 0.001.

    scheduler_params : dict
        The parameters for a learning rate scheduler for adjusting the 
        optimizer's learning rate based on validation loss. Defaults to 
        factor=0.4 and patience=2.

    scheduler : torch.optim.lr_scheduler
        A learning rate scheduler for adjusting the optimizer's learning rate
        based on validation loss. Defaults to ReduceLROnPlateau with factor=0.4
        and patience=2.

    num_epoch : int
        The number of epochs to train the model. Defaults to 100.

    train_loss_list, valid_loss_list : ndarray
        NumPy arrays storing the training and validation loss for each epoch.

    dataloader_tra, dataloader_val, dataloader_tes : DataLoader
        DataLoader instances for loading the training, validation, and test data.
        Only available after running the method load_data(cond, sample, noise...).

    size_tra, size_val, size_tes : int
        The actual sizes of the training, validation, and test datasets,
        calculated from the `tra_val_tes_size` input and the total number of samples.
        Only available after running the method load_data(cond, sample, noise...).
    
    batch_size : int
        The number of samples per batch to load during the training, validation,
        and testing phases. This size affects the gradient estimation and can
        have significant impacts on the model's training performance and efficiency.
        Only available after running the method load_data(cond, sample, noise...).

    Methods
    -------
    load_data(cond, sample, noise, tra_val_tes_size=(70, 15, 15), batch_size=500)
        Loads and preprocesses the data, splits it into training, validation,
        and testing sets, and prepares DataLoaders for training and evaluation.
        See details in this method.

    update_optimizer(optimizer_class=None, **optimizer_params)
        Updates the model's optimizer, optionally changing its class and
        parameters. See details in the method. See details in this method.

    update_scheduler(scheduler_class=None, **scheduler_params)
        Updates the learning rate scheduler, optionally changing its class and
        configuration. See details in the method. See details in this method.

    deconvolve(num_epoch=None)
        Trains the model for a specified number of epochs, evaluates it on
        the validation set, and saves the best performing model.

    sample(conditional, n_per_conditional=1, noise=None):
        Draws samples from GMM predicted by CondXD inputing conditional.
        See details in this method.
        
    _real_size(tra_val_tes_size, n_sample)
        Calculates the real sizes of training, validation, and test datasets
        based on specified proportions. 

    _split_data(cond, sample, noise, random_seed=1234)
        Splits the conditional data, samples, and noise into training,
        validation, and test sets.

    _split_data(cond, sample, noise, random_seed=1234)
        Splits the conditional data, samples, and noise into training,
        validation, and test sets.

    Examples
    --------
    >>> condxd = CondXD(n_Gaussians=3, sample_dim=2, conditional_dim=2)
    >>> condxd.load_data(cond_data, sample_data, noise_data)
    >>> condxd.deconvolve()
    >>> samples = condxd.sample(cond_data, n_per_conditional=50)
    >>> condxd.save(filename='condxd_model.pkl')

    If you want to load a trained model and sample from it:
    >>> condxd = CondXD(n_Gaussians=3, sample_dim=2, conditional_dim=2)
    >>> condxd.load('condxd_model.pkl')
    >>> samples = condxd.sample(cond_data, n_per_conditional=50)
    """


    def __init__(self,
                 n_Gaussians,
                 sample_dim,
                 conditional_dim):

        super(CondXD, self).__init__(n_Gaussians, sample_dim, conditional_dim)
            
        self.n_Gaussians = n_Gaussians
        self.sample_dim = sample_dim
        self.conditional_dim = conditional_dim
        
        # Setup the optimizer with formatted arguments for readability
        self.optimizer_params = {
            'lr' : 1e-3,
            'weight_decay' : 0.001
        }
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            **self.optimizer_params
        )
        
        # Setup the learning rate scheduler with formatted arguments
        self.scheduler_params = {
            'factor' : 0.4,
            'patience' : 2
        }
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            **self.scheduler_params
        )

    def load_data(self, cond, sample, noise=None, tra_val_tes_size=(70, 15, 15),
                batch_size=500, seed=1234):
        """
        Loads preprocessed data, then splits it into training, validation,
        and testing sets. Finally, it prepares DataLoader instances for each set
        to facilitate batch processing during training, evaluation and testing 
        phases.

        This method converts input data into PyTorch tensors, ensures the
        consistency of data dimensions, and organizes the data into batches,
        allowing for efficient model training, evaluation and testing.

        Parameters
        ----------
        cond : array-like
            The conditional data used as input for CondXD. Should be in a
            compatible format (e.g., list, NumPy array or torch.Tensor) that 
            can be converted to a torch.Tensor. Shape should be (n_samples, 
            conditional_dim), where n_samples is the total number of samples 
            and conditional_dim is the dimensionality of the conditional data.
            
        sample : array-like
            The sample data that CondXD aims to estimate probability density,
            given the conditional input. Format and shape requirements are 
            similar to `cond`, with shape (n_samples, sample_dim). If sample
            is noisy, the corresponding Gaussian noise covariance matrix should
            also be provided to `noise'.
            
        noise : array-like (optional, default=None)
            The noise covariance matrix  associated with each sample. Shape 
            should be (n_samples, sample_dim, sample_dim) to match the dimensions
            of the sample data.
            
        tra_val_tes_size : tuple of int (optional, default=(70, 15, 15))
            The proportions of the dataset to be allocated to the training,
            validation, and testing sets, respectively. Values are not required
            to sum to any specific number such as 100. The relative size is 
            when computing the real set size.
            
        batch_size : int (optional, default=500)
            The number of samples per batch to load during the training, validation,
            and testing phases. This size affects the gradient estimation and can
            have significant impacts on the model's training performance and efficiency.

        Raises
        ------
        ValueError
            If the lengths of `cond`, `sample`, and `noise` do not match, or if
            the dimensions of `sample` and `noise` do not match the expected
            sample_dim or each other.

        Notes
        -----
        The method internally calls `_split_data` to divide the dataset into
        training, validation, and testing sets based on `tra_val_tes_size`. It then
        uses these sets to create DataLoader instances, which are stored as
        attributes of the instance for use in training, evaluation and testing.
        """
        # Convert to torch.Tensor
        cond = torch.Tensor(cond)
        sample = torch.Tensor(sample)
        noise = torch.Tensor(noise)

        # Check dimension match
        n_cond, dim_cond = cond.shape
        n_sample, dim_sample = sample.shape
        n_noise, dim_noise, _ = noise.shape
        if not (n_cond == n_sample == n_noise):
            raise ValueError("The length of conditional {}, sample {} and noise {}"
                            " do not match.".format(n_cond, n_sample, n_noise))
        if not (dim_sample == dim_noise == self.sample_dim):
            raise ValueError("The dimensions of sample {}, noise {} and "
                            "initialization {} do not match."
                            .format(dim_sample, dim_noise, self.sample_dim))
        
        # Batch_size
        self.batch_size = batch_size

        # real size of training / validation / test set
        self.size_tra, self.size_val, self.size_tes = self._real_size(
            tra_val_tes_size, n_sample)

        # Define cond_tra, sample_tra, self.noise_tra
        splits = self._split_data(cond, sample, noise, random_seed=seed)
        cond_tra, sample_tra, noise_tra, \
        cond_val, sample_val, noise_val, \
        cond_tes, sample_tes, noise_tes = splits

        # Load data into batches
        self.dataloader_tra = DataLoader(
                TensorDataset(cond_tra, sample_tra, noise_tra),
                batch_size=self.batch_size, shuffle=True)

        self.dataloader_val = DataLoader(
                TensorDataset(cond_val, sample_val, noise_val),
                batch_size=self.batch_size, shuffle=False)

        self.dataloader_tes = DataLoader(
                TensorDataset(cond_tes, sample_tes, noise_tes),
                batch_size=self.batch_size, shuffle=False)




    def _real_size(self, tra_val_tes_size, n_sample):
        """Calculating the real size of the training, validation, and testing
        samples.
        """
        size_tra, size_val, size_tes = tra_val_tes_size
        sum_size = size_tra + size_val + size_tes

        real_size_tra = np.round(n_sample * size_tra / sum_size).astype('int')
        real_size_val = np.round(n_sample * size_val / sum_size).astype('int')
        real_size_tes = np.round(n_sample * size_tes / sum_size).astype('int')

        if real_size_tra + real_size_val + real_size_tes != n_sample:
            delta = (real_size_tra + real_size_val + real_size_tes - n_sample)
            real_size_tes -= delta.astype('int')
        
        return real_size_tra, real_size_val, real_size_tes


    def _split_data(self, cond, sample, noise, random_seed=1234):
        """Split cond, sample, and noise into train, val, and test sets.
        """
        sizes = [self.size_tra, self.size_val, self.size_tes]
        id_sep = torch.cat([
            torch.full((size,), i, dtype=torch.int) 
            for i, size in enumerate(sizes, start=1)
        ])
        
        torch.manual_seed(random_seed)
        id_sep = id_sep[torch.randperm(id_sep.size(0))]

        self.id_seperation = id_sep

        splits = {}
        for i, set_name in enumerate(['tra', 'val', 'tes'], start=1):
            idx = id_sep == i
            splits[f'cond_{set_name}'] = cond[idx]
            splits[f'sample_{set_name}'] = sample[idx]
            splits[f'noise_{set_name}'] = noise[idx]
        
        return [
            splits['cond_tra'], splits['sample_tra'], splits['noise_tra'],
            splits['cond_val'], splits['sample_val'], splits['noise_val'],
            splits['cond_tes'], splits['sample_tes'], splits['noise_tes']
        ]

        
    def update_optimizer(self, optimizer_class=None, **optimizer_params):
        """
        Updates the optimizer used for training the model. This method allows
        customizing optimizer class and its parameters, facilitating 
        experimentation with different optimization strategies and 
        configurations.

        Parameters
        ----------
        optimizer_class : torch.optim.Optimizer class (optional, default=None)
            The class of optimizer to use for training the model. If None, the
            current optimizer's class is used. 
        
        **optimizer_params : dict
            Arbitrary keyword arguments specific to the optimizer class. These
            are passed directly to the optimizer's constructor. Common parameters
            include `lr` for learning rate and `weight_decay` for regularization.
            For more paramesters see CondXD.optimizer.param_groups.

        Examples
        --------
        >>> model = CondXD(n_Gaussians=3, sample_dim=2, conditional_dim=2)
        >>> model.update_optimizer(optimizer_class=torch.optim.SGD, lr=0.01, momentum=0.9)
        This changes the model's optimizer to SGD with a learning rate of 0.01
        and momentum of 0.9.
        """

        # Update the optimizer with new parameters
        if optimizer_params:
            self.optimizer_params.update(optimizer_params)
        # Use the current optimizer class if none is provided
        if optimizer_class is None:
            optimizer_class = type(self._optimizer)
        # Update the optimizer
        self.optimizer = optimizer_class(
            self.parameters(),
            **self.optimizer_params
        )
        # Since the optimizer has changed, reinitialize the scheduler with the updated optimizer
        self.update_scheduler()

    def update_scheduler(self, scheduler_class=None, **scheduler_params):
        """
        Updates the learning rate scheduler associated with the optimizer. This
        method is useful for adjusting the learning rate scheduling strategy 
        and its parameters during training, supporting flexible optimization 
        dynamics. If any bug is found, please update the scheduler manually:
        CondXD.scheduler = torch.optim.lr_scheduler.schedulertype(...),

        Parameters
        ----------
        scheduler_class : torch.optim.lr_scheduler class (optional, default=None)
            The class of scheduler to use for adjusting the learning rate during
            training. If None, the current scheduler's class is used.

        **scheduler_params : dict
            Arbitrary keyword arguments specific to the scheduler class. These
            parameters are passed directly to the scheduler's constructor.
            Note that `optimizer` parameter is automatically handled and should
            not be passed.

        Examples
        --------
        >>> model = CondXD(n_Gaussians=3, sample_dim=2, conditional_dim=2)
        >>> model.update_scheduler(scheduler_class=torch.optim.lr_scheduler.StepLR, step_size=5, gamma=0.5)
        This changes the model's scheduler to StepLR, decreasing the learning rate
        by half every 5 epochs.
        """

        # Update only the provided parameters, keep others as is
        if scheduler_params:
            self.scheduler_params.update(scheduler_params)
        # Use the current scheduler class if none is provided
        if scheduler_class is None:
            scheduler_class = type(self._scheduler)
        # Update the scheduler
        self.scheduler = scheduler_class(
            self.optimizer,
            **self.scheduler_params
        )

    def deconvolve(self, num_epoch=100):
        """
        Trains the model for a specified number of epochs. During each epoch, the
        method performs training on the training dataset and evaluates the model on
        the validation dataset. It also tracks and prints the training and validation
        loss for each epoch. The best model, determined by the lowest validation loss,
        is saved if an output path is provided. After training, if a test set is 
        provided the method evaluates the model on the test dataset and prints the 
        final test loss.

        Parameters
        ----------
        num_epoch : int, optional
            The number of epochs to train the model. If not specified, the method
            uses the class's default number 100 of epochs set during initialization.

        Notes
        -----
        The training process involves:
        - Setting the model to training mode.
        - Iterating over the training DataLoader to compute the loss and update the model parameters.
        - Setting the model to evaluation mode.
        - Iterating over the validation DataLoader to compute the validation loss.
        - Updating the best model based on validation loss and optionally saving it.
        - Finally, the method sets the model to evaluation mode again and computes the test loss.

        This method updates the `train_loss_list` and `valid_loss_list` attributes
        with the training and validation loss values for each epoch, respectively.

        Examples
        --------
        >>> condxd = CondXD(n_Gaussians=3, sample_dim=2, conditional_dim=2)
        >>> condxd.load_data(cond_data, sample_data, noise_data)
        >>> condxd.deconvolve(num_epoch=50)
        This example initializes the CondXD model with specific parameters, loads
        the data, and then trains the model for 50 epochs.
        """

        self.num_epoch = num_epoch
        self.train_loss_list = np.ones(self.num_epoch) * np.nan
        self.valid_loss_list = np.ones(self.num_epoch) * np.nan

        lowest_loss = float('inf')  # Use inf for initial comparison
        best_model = copy.deepcopy(self.state_dict())

        for epoch in range(self.num_epoch):
            train_loss = self._train_epoch(epoch)
            val_loss = self._validate_epoch(epoch)
            print(f"Epoch {epoch}, training loss: {train_loss:.5f}, "
                  f"validation loss: {val_loss:.5f}.")

            # Update best model if validation loss is improved
            if val_loss < lowest_loss:
                lowest_loss = val_loss
                best_model = copy.deepcopy(self.state_dict())
        
        # Load the best model back into the model
        self.load_state_dict(best_model)

        # Final test
        if self.size_tes != 0:
            tes_loss = self.test_model()

    def _train_epoch(self, epoch):
        self.train()  # Set model to training mode
        total_loss = 0
        for cond_i, sample_i, noise_i in self.dataloader_tra:
            self.optimizer.zero_grad()
            loss = self.loss(cond_i, sample_i, noise=noise_i, 
                             regularization=True)
            total_loss += loss.item() * cond_i.size(0)
            loss.backward()
            self.optimizer.step()
        
        avg_loss = total_loss / self.size_tra
        self.train_loss_list[epoch] = avg_loss
        # print(f'Epoch {epoch}, training loss: {avg_loss:.5f}.')
        return avg_loss

    def _validate_epoch(self, epoch):
        self.eval()  # Set model to evaluation mode
        total_loss = 0
        with torch.no_grad():  # No gradients needed
            for cond_i, data_i, noise_i in self.dataloader_val:
                loss = self.loss(cond_i, data_i, noise=noise_i, 
                                       regularization=True)
                total_loss += loss.item() * cond_i.size(0)
        
        avg_loss = total_loss / self.size_val
        self.valid_loss_list[epoch] = avg_loss
        # print(f'Epoch {epoch}, validation loss: {avg_loss:.5f}.')
        return avg_loss

    def test_model(self, external_cond=None, external_samples=None):
        """Under construction. Wants to work for external conds and samples.

        Parameters
        ----------
        external_cond : _type_, optional
            _description_, by default None
        external_samples : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        self.eval()  # Set model to evaluation mode
        total_loss = 0
        
        if (external_cond is None) and (external_samples is None):
            testloader = self.dataloader_tes

        with torch.no_grad():  # No gradients needed
            for cond_i, data_i, noise_i in testloader:
                loss = self.loss(cond_i, data_i, noise=noise_i, 
                                       regularization=True)
                total_loss += loss.item() * cond_i.size(0)
        
        tes_loss = total_loss / self.size_tes
        print(f'Final test loss: {tes_loss:.5f}.')
        
        return tes_loss


    def sample(self, conditional, n_per_conditional=1, noise=None):
        """To draw samples from GMM predicted by CondXD taking conditional
        input.

        Parameters
        ----------
        conditional : torch.Tensor
            The input conditional tensor based on which the GMM to be sampled 
            is predicted. The expected shape should be (batch_size, 
            conditional_dim).

        n_per_conditional : int (optional, default=1)
            The number of samples to draw per input conditional, resulting in a
            total of `batch_size * conditional_dim` samples.

        noise : torch.Tensor (optional, default=None)
            Gaussian noise covariance matrix of every sample, with shape 
            (batch_size, n_per_conditional, sample_dim, sample_dim). If None, no 
            noise is added.

        Returns
        -------
        samples : torch.Tensor
            The drawn samples as a tensor. The shape of the output will depend 
            on `n_per_conditional` and the model's output sample dimension, 
            generally expected to be (batch_size, n_per_conditional, sample_dim).

        Notes
        -----
        This method utilizes the GMM parameters (mixing coefficients, means, 
        covars) obtained from the forward pass of the input conditionals 
        through the CondXD model, selects components based on the mixing 
        coefficients, and generates samples accordingly. Noise covariance can 
        be added to the covariance matrices if there is Gaussian noise.
        """
    
        conditional = torch.Tensor(conditional)
        
        mixcoef, means, covars = self.forward(conditional)
        
        batchsize = conditional.shape[0]
        draw = multinomial(mixcoef, n_per_conditional, replacement=True)
        means  = means[:, draw][torch.eye(batchsize).to(torch.bool)]
        covars = covars[:, draw][torch.eye(batchsize).to(torch.bool)]

        if noise is None:
            noise = torch.zeros_like(covars)
        else:
            noise = torch.Tensor(noise)
        if noise.dim() != covars.dim():
            noise = noise[:, None, ...]  # add noise to all components

        noisy_covars = covars + noise

        noisy_covars = 0.5 * (noisy_covars + noisy_covars.transpose(-1, -2))

        sample = mvn(loc=means, covariance_matrix=noisy_covars).sample()

        return sample
    
    def save(self, filename=None):
        """Save the model to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save the model as (including path).
        """
        if filename is None:
            filename = 'CondXD_params.pkl'

        torch.save(self.state_dict(), filename)
    
    def load(self, filename):
        """Load the model from a file.

        Parameters
        ----------
        filename : str
            The name of the file to load the model from.
        """
        self.load_state_dict(torch.load(filename))
        self.eval()