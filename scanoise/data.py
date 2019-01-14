import pickle as pkl

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .utils import to_tensor


MAX_TRAIN_RATIO = 0.8
RANDOM_SEED = 1234


def _load_data(fn, dtype=np.float32):
    return np.load(fn).astype(dtype)


def load_dataset(*tensors):
    return TensorDataset(*tensors)


def get_data_loader(dataset, batch_size=64, shuffle=True, drop_last=True, n_jobs=0):
    return DataLoader(
        dataset, batch_size=batch_size, drop_last=drop_last,
        shuffle=shuffle, num_workers=n_jobs
    )


def get_splits(X, Y, n_trains, n_valids, random_state=RANDOM_SEED):
    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=n_trains, test_size=n_valids
    )
    for train_ix, valid_ix in sss.split(X, Y):
        Xtr, Xvl = X[train_ix], X[valid_ix]
        Ytr, Yvl = Y[train_ix], Y[valid_ix]

    return ((Xtr, Ytr), (Xvl, Yvl)), train_ix, valid_ix


def prepare_data(trace_fn, label_fn, n_trains='full', n_tests=25000):
    # load data
    X = _load_data(trace_fn)
    Y = _load_data(label_fn).astype(int).ravel()

    # pre-split test set
    Xts, Yts = to_tensor(X[-n_tests:]), to_tensor(Y[-n_tests:])

    # if using full or the number exceeds the 90% of the remaining traces
    # after split the test, use that 90%.
    n_train_total = X.shape[0] - n_tests
    max_train = int(n_train_total * MAX_TRAIN_RATIO)
    n_valids = n_train_total - max_train
    if (n_trains == 'full') or (n_trains >= max_train):
        n_trains = max_train

    ((Xtr, Ytr), (Xvl, Yvl)), train_ix, valid_ix = get_splits(
        X[:-n_tests], Y[:-n_tests], n_trains, n_valids
    )

    # to tensor
    Xtr, Xvl = to_tensor(Xtr), to_tensor(Xvl)
    Ytr, Yvl = to_tensor(Ytr), to_tensor(Yvl)

    return {
        'train': {'data': load_dataset(Xtr, Ytr), 'idx': train_ix},
        'valid': {'data': load_dataset(Xvl, Yvl), 'idx': valid_ix},
        'test': {'data': load_dataset(Xts, Yts),
                 'idx': np.arange(X.shape[0] - n_tests, X.shape[0])}
    }
