from os.path import join, basename, dirname
from pathlib import Path
import time
import copy

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .data import prepare_data, get_data_loader
from .models import (CNN1DDPAv4,
                     CNN1DRand,
                     CNN1DShivam,
                     CNN1DASCAD,
                     CNN1DASCAD700)
from .utils import to_numpy, replace_dots


MODEL_MAP = {
    'dpav4': CNN1DDPAv4,
    'random': CNN1DRand,
    'shivam': CNN1DShivam,
    'ascad': CNN1DASCAD700,
    'Benadjila': CNN1DASCAD
}
N_CLASSES = 256


def run(epoch, type, model, dataloader, loss_function,  optimizer=None):
    """"""
    start = time.time()

    if type == 'train':
        model.train()
        assert optimizer is not None
    else:
        model.eval()
        optimizer = 'null'

    error, metric = 0, 0
    for j, batch in enumerate(dataloader):
        X, y_true = batch
        if next(model.parameters()).is_cuda:
            X, y_true = X.cuda(), y_true.cuda()

        # flush optimizer
        if type == 'train':
            optimizer.zero_grad()

        y_pred = model(X[:, None])
        l = loss_function(y_pred, y_true)

        # update
        if type == 'train':
            l.backward()
            optimizer.step()

        error += to_numpy(l)
        metric += accuracy_score(
            to_numpy(y_pred).argmax(axis=1), to_numpy(y_true)
        )
    error /= (j + 1)
    metric /= (j + 1)

    return {
        'loss': error,
        'accuracy': metric,
        'time': time.time() - start,
        'type': type,
        'epoch': epoch
    }


def save_test_prob(n, model, test_dataloader, out_path, name):
    out_fn = join(out_path, name + '_{:03d}.csv.tar.gz'.format(n))
    model.eval()  # just to make sure
    prob = []
    for X, _ in test_dataloader:
        if next(model.parameters()).is_cuda:
            X = X.cuda()
        o = model(X[:, None])
        p = F.softmax(o, dim=1)
        prob.append(to_numpy(p))
    prob = np.concatenate(prob, axis=0)
    pd.DataFrame(prob).to_csv(
        out_fn, header=None, index=None,
        float_format='%.5f', compression='gzip'
    )


def report_progress(n, reports, is_test):
    # report
    tmp_report = pd.DataFrame(reports)
    cur =  tmp_report[tmp_report.epoch == n]
    if is_test:
        print(
            "{:03d}th iter --- [tloss: {:.4f}] [vloss: {:4f}] [vacc: {:.4f}] [tacc: {:.4f}]"
            .format(
                n, cur[cur.type == 'train'].loss.item(),
                cur[cur.type == 'eval'].loss.item(),
                cur[cur.type == 'eval'].accuracy.item(),
                cur[cur.type == 'test'].accuracy.item()
            )
        )
    else:
        print(
            "{:03d}th iter --- [tloss: {:.4f}] [vloss: {:4f}] [vacc: {:.4f}]"
            .format(
                n, cur[cur.type == 'train'].loss.item(),
                cur[cur.type == 'eval'].loss.item(),
                cur[cur.type == 'eval'].accuracy.item()
            )
        )


def train(trace_fn, label_fn, n_trains='full', n_tests=25000,
          model='dpav4', noise=0.5, lr=0.0001, l2=1e-7, batch_size=100,
          n_epoch=100, record_every=10, is_gpu=True, out_root='./', name=None):
    """Main training function

    Args:
        trace_fn (str): path to the trace (.npy) file
        label_fn (str): path to the label (.npy) file
        n_trains (str or int): number of traces to be used in training.
                               if 'full', it uses 90% of training set as
                               training traces
        n_tests (int): number of traces used for the testing
        model (str): model id for the desired model type (defined in model.py)
        noise (float): desired amount of noise addtion on the input
        lr (float): learning rate
        l2 (float): weight decay coefficient
        batch_size (int): batch size
        n_epoch (int): number of epochs
        record_every (int): frequency of evaluation (unit: epoch)
        is_gpu (bool): indicates trainer using GPU
    """
    # prepare filename / path
    if name is None:
        if n_trains != 'full':
            name = '{}_n{:d}_noise{:.01f}'.format(model, n_trains, noise)
        else:
            name = '{}_nfull_noise{:.01f}'.format(model, noise)
    name = replace_dots(name)

    # make path if not existing
    path = Path(out_root)
    path.mkdir(parents=True, exist_ok=True)

    # load datasets
    [train, valid, test] = [
        get_data_loader(dataset, batch_size=batch_size)
        for dataset
        in prepare_data(trace_fn, label_fn, n_trains, n_tests)
    ]

    # init model
    if model == 'Benadjila':
        # signal stat is explicitly computed before the training
        mean = train.tensors[0].mean()
        std = train.tensors[0].std()
        model = MODEL_MAP[model](1, N_CLASSES, noise, mean, std)
    else:
        model = MODEL_MAP[model](1, N_CLASSES, noise)

    if is_gpu:
        model.cuda()

    # setup optimizer
    opt = optim.Adam(
        filter(lambda w: w.requires_grad, model.parameters()),
        lr=lr, weight_decay=l2
    )

    # setup loss
    loss = nn.CrossEntropyLoss()
    
    # start training
    report = []
    best = {
        'epoch': None,
        'model': None,
        'vacc': None
    }
    for n in range(n_epoch + 1):
        is_test = (n % record_every == 0)

        # evaluate
        report.append(
            run(n, 'eval', model, valid, loss)
        )
        if (not best['vacc']) or (report[-1]['accuracy'] > best['vacc']):
            best['model'] = copy.deepcopy(model.cpu())
            best['epoch'] = n
            best['vacc'] = report[-1]['accuracy']
            if is_gpu:
                model.cuda()

        # test
        if is_test:
            report.append(
                run(n, 'test', model, test, loss)
            )

        # train 
        report.append(
            run(n, 'train', model, train, loss, opt)
        )
    
        report_progress(n, report, is_test)
    
    # save best model's prob
    save_test_prob(best['epoch'], best['model'],
                   test, out_root, name + '_best')

    # save output report
    pd.DataFrame(report).to_csv(join(out_root, name + '_report.csv'))