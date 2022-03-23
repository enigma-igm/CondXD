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
        """

        :param GMM_params:
        :param hyper_params:
        :param low_SN_mag:
        """

        # Read the hyperparmeters
        self.lr = hyper_params.get("learning_rate", 1e-3)
        self.bathc_size = hyper_params.get("batch_size", 500)
        self.schedule_factor = hyper_params.get("schedule_factor", 0.4)
        self.patience = hyper_params.get("patience", 2)
        self.num_epoch = hyper_params.get("num_epoch", 100)
        self.weight_decay = hyper_params.get("weight_decay", 1e-3)
        self.size_train = hyper_params.get("size_training", 70)
        self.size_val = hyper_params.get("size_validation", 15)
        self.size_test = hyper_params.get("size_testing", 15)
        self.n_gauss = hyper_params.get("n_gauss", 20)

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
        ref_f_err = output[GMM_params['ref_flux_err']].astype('float').reshape(-1, 1)
        hdu_list.close()

        self.rel_flux = torch.Tensor(flux/ref_f.T).transpose(1, 0)
        flux_err = torch.Tensor(flux_err).transpose(1, 0)
        ref_f = torch.Tensor(ref_f)
        self.ref_f = torch.log(ref_f)
        self.ref_f_err = torch.Tensor(ref_f_err)
        self.rel_flux_err = self._get_noise_covar(flux_err, ref_f, low_SN_mag=low_SN_mag)

        self._real_size()

    def _get_noise_covar(self, flux_err, ref_f, low_SN_mag=21):
        """Generating the covariance matrix of the relative fluxes. If the reference mag is fainter than
        the defined SNR (low_SN_mag), then the covariance is simply diagonal.

        :param flux_err:
        :param low_SN_mag:

        Returns:
            tensor: covariant noisy matrix of the relative fluxes.
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
        """ Create the sample corresponding to the define parameter

        :param sample:

        Returns:
            the relative fluxes, covariance matrix, and reference flux for the specified sample
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
        """ Deconvolve and fir GMM to the training data

        :param ref_f_train:
        :param data_train:
        :param rel_err_train:
        :param ref_f_val:
        :param data_val:
        :param rel_err_val:
        :param model_name:
        """

        # Create the directory to save the GMMs
        path = os.getcwd()
        if os.path.isdir('XD_fit_models'):
            print("Directory XD_fit_models already exists")
        else:
            print("Creating the directory: XD_fit_models")
            os.mkdir(path + '/XD_fit_models')

        # put data into batches
        batch_size = self.bathc_size
        train_loader = DataLoader(TensorDataset(ref_f_train, data_train, rel_err_train),
                                  batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(TensorDataset(ref_f_val, data_val, rel_err_val),
                                  batch_size=batch_size, shuffle=False)

        # NN initialization
        gmm = model.GMMNet(self.n_gauss, self.rel_flux.shape[1], conditional_dim=1)
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
                    log_prob_b, loss = gmm.score(data_i, ref_f_i, noise=data_err_i, regression=True)
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
                    log_prob_b, loss = gmm.score(data_i, ref_f_i, noise=data_err_i)
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
        """ Deconvolve and fir GMM to the training data

        :param ref_f_test:
        :param data_test:
        :param rel_err_test:
        :param model_name:
        """
        test_loader = DataLoader(TensorDataset(ref_f_test, data_test, rel_err_test), batch_size=self.bathc_size,
                                 shuffle=False)

        # NN initialization
        best_model = model.GMMNet(self.n_gauss, self.rel_flux.shape[1], conditional_dim=1)
        best_model.load_state_dict(torch.load(f'XD_fit_models/{self.n_gauss}G_{model_name}.pkl'))

        best_model.eval()
        tes_loss = 0
        for i, (ref_f_i, data_i, rel_err_i) in enumerate(test_loader):
            log_prob_b, loss = best_model.score(data_i, ref_f_i, noise=rel_err_i)
            tes_loss += loss

        tes_loss = tes_loss / self.size_test
        print('\nTest loss:', tes_loss.item())

    def sample_data(self, model_name):
        """ Deconvolve and fir GMM to the training data

        :param model_name:
        Returns:
            the relative fluxes, and the sampled deconvolved and reconvolved relative fluxes
        """

        # NN initialization
        best_model = model.GMMNet(self.n_gauss, self.rel_flux.shape[1], conditional_dim=1)
        best_model.load_state_dict(torch.load(f'XD_fit_models/{self.n_gauss}G_{model_name}.pkl'))
        best_model.eval()

        output = torch.zeros_like(self.rel_flux)
        output_noisy = torch.zeros_like(self.rel_flux)

        for i in range(len(self.ref_f)):
            output[i] = best_model.sample(self.ref_f[i].unsqueeze(0), 1)
            output_noisy[i] = best_model.sample(self.ref_f[i].unsqueeze(0), 1,
                                                self.rel_flux_err[i].unsqueeze(0))

        output = output.numpy()
        output_noisy = output_noisy.numpy()
        real_data = self.rel_flux.numpy()

        return real_data, output, output_noisy
