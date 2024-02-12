import os
import copy

import numpy as np

from astropy.table import Table
from astropy.io import fits, ascii

import torch
from torch.utils.data import DataLoader, TensorDataset

# TODO: DMY - we may want to rename this to something more descriptive
from .base import CondXDBase

from IPython import embed

__all__ = ['CondXD']

class CondXD(CondXDBase):
    """
    TODO

    load processed data (not file, directly for training)
	data split
	fit
	val/test (output loss)
	sample
    """
    pass