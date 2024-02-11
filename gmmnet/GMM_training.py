#!/usr/bin/env python

import os
import copy

import numpy as np

from astropy.io import fits
from astropy.table import Table

import torch
from torch.utils.data import DataLoader, TensorDataset

from models import model

from IPython import embed

class GMM:
    """
    Class for deconvolving and fitting data with Gaussian mixture models, and classifying them
    """
    def __init__(self, GMM_params, hyper_params, low_SN_mag=21):
        """ This is the initialization function

            Args:
                GMM_params (dictionary): physical parameters for contaminants and QSOs
                hyper_params (dictionary): hyperparameters used to train the NN
                low_SN_mag (float): reference magnitude above which the covariance matrix is simply diagonal
        """

        # Read the hyperparmeters
        self.lr = hyper_params.get("learning_rate", 1e-3)
        self.batch_size = hyper_params.get("batch_size", 500)
        self.schedule_factor = hyper_params.get("schedule_factor", 0.4)
        self.patience = hyper_params.get("patience", 2)
        self.num_epoch = hyper_params.get("num_epoch", 100)
        self.weight_decay = hyper_params.get("weight_decay", 1e-3)
        self.size_train = hyper_params.get("size_training", 70)
        self.size_val = hyper_params.get("size_validation", 15)
        self.size_test = hyper_params.get("size_testing", 15)
        self.n_gauss = hyper_params.get("n_gauss", 20)
        self.conditional_dim = hyper_params.get("conditional_dim", 1)

        # Check that the fraction training, validation, and testing samples sum up to 100
        if (self.size_train+self.size_val+self.size_test) != 100:
            print('ERROR: size of training+validation+test sample is {}'.format(int(self.size_train + self.size_val
                                                                                    + self.size_test)))
            print('Please check it before running again the script')
            exit()

        # Read the data
        hdu_list = fits.open(GMM_params['path']+'/'+GMM_params['table_name'], memmap=True)
        output = Table(hdu_list[1].data)
        flux = np.array([output[GMM_params['fluxes'][i]] for i in range(len(GMM_params['fluxes']))])
        flux_err = np.array([output[GMM_params['fluxes_err'][i]] for i in range(len(GMM_params['fluxes_err']))])
        ref_f = output[GMM_params['ref_flux']].astype('float').reshape(-1, 1)
        self.classif = GMM_params.get("class", "contaminants")
        ref_f_err = output[GMM_params['ref_flux_err']].astype('float').reshape(-1, 1)

        hdu_list.close()

        self.rel_flux = torch.Tensor(flux/ref_f.T).transpose(1, 0)
        flux_err = torch.Tensor(flux_err).transpose(1, 0)
        ref_f = torch.Tensor(ref_f)
        self.ref_f = torch.log(ref_f)
        self.ref_f_err = torch.Tensor(ref_f_err)
        self.rel_flux_err = self._get_noise_covar(flux_err, ref_f, low_SN_mag=low_SN_mag)

        # If training for QSOs, adds a tiny error to the data
        if self.classif in "qso":
            self.rel_flux = torch.Tensor(np.random.normal(self.rel_flux, 0.01))
            diag = np.arange(self.rel_flux.shape[1])
            self.rel_flux_err = torch.Tensor(np.zeros_like(self.rel_flux_err))
            self.rel_flux_err[:, diag, diag] = 0.01

        self._real_size()

    def _get_noise_covar(self, flux_err, ref_f, low_SN_mag=21):
        """Generating the covariance matrix of the relative fluxes. If the reference mag is fainter than
        the low_SN_mag, then the covariance is simply diagonal.

            Args:
                flux_err (tensor): errors matrix of the different fluxes
                ref_f (tensor): detection band fluxes
                low_SN_mag (float): reference magnitude above which the covariance matrix is simply diagonal

        Returns:
            err_r_set (tensor): covariant noisy matrix of the relative fluxes.
        """
        # new covariance matrix
        len_data, dimension = self.rel_flux.shape
        err_r_set = torch.zeros((len_data, dimension, dimension))

        # compute off-diagonal elements
        high_SN_bln = ((22.5 - 2.5 * torch.log10(ref_f)) <= low_SN_mag).flatten()
        for i in range(1, dimension):
            for j in range(i):
                err_r_set[high_SN_bln, i, j] = (self.rel_flux[:, i] * self.rel_flux[:, j] / ref_f[:, 0] ** 2
                                                * self.ref_f_err[:, 0] ** 2)[high_SN_bln]
        err_r_set = err_r_set + err_r_set.transpose(2, 1)

        # compute diagonal elements
        for i in range(dimension):
            err_r_set[:, i, i] = 1 / ref_f[:, 0] ** 2 * flux_err[:, i] ** 2 +\
                                 self.rel_flux[:, i] ** 2 / ref_f[:, 0] ** 2 * self.ref_f_err[:, 0] ** 2

        return err_r_set

    def _real_size(self,):
        """Define the real size of the training, validation, and testing samples
        """

        len_data = self.rel_flux.shape[0]
        real_size_tra = np.round(len_data * self.size_train / 100).astype('int')
        real_size_val = np.round(len_data * self.size_val / 100).astype('int')
        real_size_tes = np.round(len_data * self.size_test / 100).astype('int')
        if real_size_tra + real_size_val + real_size_tes != len_data:
            delta = (real_size_tra + real_size_val + real_size_tes - len_data).astype('int')
            real_size_tes -= delta

        self.size_train = real_size_tra
        self.size_val = real_size_val
        self.size_test = real_size_tes

        id_sep = np.append(np.ones(real_size_tra), np.append(np.ones(real_size_val) * 2,
                                                             np.ones(real_size_tes) * 3)).astype('int')
        np.random.seed(1234)
        np.random.shuffle(id_sep)
        self.id_sep = id_sep

    def sample_splitting(self, sample='training'):
        """ Create the sample corresponding to the defined parameter: training, validation, testing.

            Args:
                sample (string): keyword specifying which sample need to be sampled.

        :param sample:

        Returns:
            data: sampled relative fluxes
            data_err: sampled covariance matrix
            ref_f: sampled reference flux for the specified sample
        """

        asign = 0
        if sample in 'training':
            asign = 1
        elif sample in 'validation':
            asign = 2
        elif sample in 'testing':
            asign = 3
        else:
            print('ERROR: sample {} is not defined'.format(sample))
            print('Please check it before running again the script')
            exit()

        data = self.rel_flux[self.id_sep == asign]
        data_err = self.rel_flux_err[self.id_sep == asign]
        ref_f = self.ref_f[self.id_sep == asign]

        return data, data_err, ref_f

    def gmm_fit(self, ref_f_train, data_train, rel_err_train, ref_f_val, data_val, rel_err_val, model_name):
        """ Deconvolve and fit GMM to the training data.

            Args:
                ref_f_train (tensor): reference flux for the training sample.
                data_train (tensor): relative fluxes for the training sample.
                rel_err_train (tensor): covariance matrix for the training sample.
                ref_f_val (tensor): reference flux for the validation sample.
                data_val (tensor): relative fluxes for the validation sample.
                rel_err_val (tensor): covariance matrix for the validation sample.
                model_name (string): name of the output trained model.
        """

        # Create the directory to save the GMMs
        path = os.getcwd()
        if os.path.isdir('XD_fit_models'):
            print("Directory XD_fit_models already exists")
        else:
            print("Creating the directory: XD_fit_models")
            os.mkdir(path + '/XD_fit_models')

        # put data into batches
        batch_size = self.batch_size
        train_loader = DataLoader(TensorDataset(ref_f_train, data_train, rel_err_train),
                                  batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(TensorDataset(ref_f_val, data_val, rel_err_val),
                                  batch_size=batch_size, shuffle=False)

        # NN initialization
        gmm = model.GMMNet(self.n_gauss, self.rel_flux.shape[1], conditional_dim=self.conditional_dim)
        optimizer = torch.optim.Adam(gmm.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.schedule_factor,
                                                               patience=self.patience)

        # start training process loop
        epoch = self.num_epoch
        lowest_loss = 9999
        size_train = len(ref_f_train)
        size_val = len(ref_f_val)
        for n in range(epoch):
            try:
                # training
                gmm.train()
                train_loss = 0
                for i, (ref_f_i, data_i, data_err_i) in enumerate(train_loader):
                    optimizer.zero_grad()
                    loss = gmm.loss(data_i, ref_f_i, noise=data_err_i, regression=True)
                    train_loss += loss
                    # backward and update parameters
                    loss.backward()
                    optimizer.step()

                train_loss = train_loss / size_train
                print('\nEpoch', (n + 1), 'Training loss: ', train_loss.item())
                scheduler.step(train_loss)

                # validating
                gmm.eval()
                val_loss = 0
                for i, (ref_f_i, data_i, data_err_i) in enumerate(valid_loader):
                    optimizer.zero_grad()
                    loss = gmm.loss(data_i, ref_f_i, noise=data_err_i, regression=True)
                    val_loss += loss

                val_loss = val_loss / size_val
                print('Epoch', (n + 1), 'Validation loss: ', val_loss.item())
                if val_loss < lowest_loss:
                    lowest_loss = val_loss
                    best_model = copy.deepcopy(gmm)
                    torch.save(best_model.state_dict(), f'XD_fit_models/{self.n_gauss}G_{model_name}.pkl')

            except KeyboardInterrupt:
                break

    def test_sample(self, ref_f_test, data_test, rel_err_test, model_name):
        """ Deconvolve and fit GMM to the test data

            Args:
                ref_f_test (tensor): reference flux for the test sample.
                data_test (tensor): relative fluxes for the test sample.
                rel_err_test (tensor): covariance matrix for the test sample.
                model_name (string): name of the output trained model.
        """

        test_loader = DataLoader(TensorDataset(ref_f_test, data_test, rel_err_test), batch_size=self.batch_size,
                                 shuffle=False)

        # NN initialization
        best_model = model.GMMNet(self.n_gauss, self.rel_flux.shape[1], conditional_dim=self.conditional_dim)
        best_model.load_state_dict(torch.load(f'XD_fit_models/{self.n_gauss}G_{model_name}.pkl'))

        best_model.eval()
        tes_loss = 0
        for i, (ref_f_i, data_i, rel_err_i) in enumerate(test_loader):
            loss = best_model.loss(data_i, ref_f_i, noise=rel_err_i, regression=True)
            tes_loss += loss

        tes_loss = tes_loss / self.size_test
        print('\nTest loss:', tes_loss.item())

    def sample_data(self, model_name, N=None):
        """ Sample data from a specific model.

            Args:
                model_name (string): name of the model from which sample model.

            Returns:
                real_data (tensor): the real relative fluxes to compare with the sample from the model.
                output (tensor): the sampled noiseless relative fluxes.
                output_noisy (tensor): the sampled noisy relative fluxes.

        """

        # NN initialization 
        best_model = model.GMMNet(self.n_gauss, self.rel_flux.shape[1], conditional_dim=self.conditional_dim)
        best_model.load_state_dict(torch.load(f'XD_fit_models/{self.n_gauss}G_{model_name}.pkl'))
        best_model.eval()

        if N is None:
            ref_f = self.ref_f
            rel_flux = self.rel_flux
            rel_flux_err = self.rel_flux_err
        else:
            # TODO DMY: maybe we need to first fit a disribution and sample from it
            indices = np.random.choice(len(self.ref_f), int(N), replace=True)
            ref_f = self.ref_f[indices]

            # TODO DMY: check if this is correct!
            rel_flux = self.rel_flux[indices]
            rel_flux_err = self.rel_flux_err[indices]

        output = torch.zeros_like(rel_flux)
        output_noisy = torch.zeros_like(rel_flux)

        for i in range(len(ref_f)):

            output[i] = best_model.sample(ref_f[i].unsqueeze(0), 1)
            output_noisy[i] = best_model.sample(ref_f[i].unsqueeze(0), 1,
                                                rel_flux_err[i].unsqueeze(0))

        output = output.numpy()
        output_noisy = output_noisy.numpy()

        return output, output_noisy, ref_f.numpy()
    
    def get_real_data(self):
        return self.rel_flux.numpy()