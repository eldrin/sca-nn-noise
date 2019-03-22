import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class CNN1DDPAv4(nn.Module):
    """"""
    def __init__(self, in_ch, n_out, gaussian_noise=0.5):
        """"""
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 8, 3)
        self.conv2 = nn.Conv1d(8, 16, 3)
        self.conv3 = nn.Conv1d(16, 32, 3)
        self.conv4 = nn.Conv1d(32, 64, 3)
        self.conv5 = nn.Conv1d(64, 128, 3)
        self.conv6 = nn.Conv1d(128, 128, 3)
        self.conv7 = nn.Conv1d(128, 128, 3)
        self.conv8 = nn.Conv1d(128, 256, 3)
        self.conv9 = nn.Conv1d(256, 256, 3)
        self.conv10 = nn.Conv1d(256, 256, 3)

        self.bn0 = nn.BatchNorm1d(1)
        self.bn1 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn7 = nn.BatchNorm1d(128)
        self.bn9 = nn.BatchNorm1d(256)

        self.pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(256, 256)
        self.out = nn.Linear(256, n_out)

        self.gaus_noise = gaussian_noise

    def forward(self, X):
        """"""
        x = self.bn0(X)

        if self.training:
            e = torch.randn(X.shape)
            if X.is_cuda: e = e.cuda()
            x = x + e * self.gaus_noise

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.pool(F.relu(self.conv6(x)))
        x = self.pool(F.relu(self.bn7(self.conv7(x))))
        x = self.pool(F.relu(self.conv8(x)))
        x = self.pool(F.relu(self.bn9(self.conv9(x))))
        x = F.relu(self.conv10(x))

        x = F.dropout(x.view(X.size(0), -1), training=self.training, p=0.5)
        x = F.dropout(F.relu(self.fc1(x)), training=self.training, p=0.5)
        x = self.out(x)

        return x


class CNN1DShivam(nn.Module):
    """"""
    def __init__(self, in_ch, n_out, gaussian_noise=0.5):
        """"""
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 8, 3)
        self.conv2 = nn.Conv1d(8, 16, 3)
        self.conv3 = nn.Conv1d(16, 32, 3)
        self.conv4 = nn.Conv1d(32, 64, 3)
        self.conv5 = nn.Conv1d(64, 128, 3)
        self.conv6 = nn.Conv1d(128, 128, 3)
        self.conv7 = nn.Conv1d(128, 256, 3)
        self.conv8 = nn.Conv1d(256, 256, 3)

        self.bn0 = nn.BatchNorm1d(1)
        self.bn1 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn7 = nn.BatchNorm1d(256)

        self.pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(512, 512)
        self.out = nn.Linear(512, n_out)

        self.gaus_noise = gaussian_noise

    def forward(self, X):
        """"""
        x = self.bn0(X)

        if self.training:
            e = torch.randn(X.shape)
            if X.is_cuda: e = e.cuda()
            x = x + e * self.gaus_noise

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.pool(F.relu(self.conv6(x)))
        x = self.pool(F.relu(self.bn7(self.conv7(x))))
        x = self.pool(F.relu(self.conv8(x)))

        x = F.dropout(x.view(X.size(0), -1), training=self.training, p=0.5)
        x = F.dropout(F.relu(self.fc1(x)), training=self.training, p=0.5)
        x = self.out(x)

        return x


class CNN1DRand(nn.Module):
    """"""
    def __init__(self, in_ch, n_out, gaussian_noise=0.5):
        """"""
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 8, 3)
        self.conv2 = nn.Conv1d(8, 16, 3)
        self.conv3 = nn.Conv1d(16, 32, 3)
        self.conv4 = nn.Conv1d(32, 64, 3)
        self.conv5 = nn.Conv1d(64, 128, 3)
        self.conv6 = nn.Conv1d(128, 128, 3)
        self.conv7 = nn.Conv1d(128, 128, 3)
        self.conv8 = nn.Conv1d(128, 256, 3)
        self.conv9 = nn.Conv1d(256, 256, 3)
        self.conv10 = nn.Conv1d(256, 256, 3)

        self.bn0 = nn.BatchNorm1d(1)
        self.bn1 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn7 = nn.BatchNorm1d(128)
        self.bn9 = nn.BatchNorm1d(256)

        self.pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(256, 256)
        self.out = nn.Linear(256, n_out)

        self.gaus_noise = gaussian_noise

    def forward(self, X):
        """"""
        x = self.bn0(X)

        if self.training:
            e = torch.randn(X.shape)
            if X.is_cuda: e = e.cuda()
            x = x + e * self.gaus_noise

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.pool(F.relu(self.conv6(x)))
        x = self.pool(F.relu(self.bn7(self.conv7(x))))
        x = self.pool(F.relu(self.conv8(x)))
        x = self.pool(F.relu(self.bn9(self.conv9(x))))
        x = self.pool(F.relu(self.conv10(x)))

        x = F.dropout(x.view(X.size(0), -1), training=self.training, p=0.5)
        x = F.dropout(F.relu(self.fc1(x)), training=self.training, p=0.5)
        x = self.out(x)

        return x


class CNN1DASCAD700(nn.Module):
    """"""
    def __init__(self, in_ch, n_out, gaussian_noise=0.5):
        """"""
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 8, 3)
        self.conv2 = nn.Conv1d(8, 16, 3)
        self.conv3 = nn.Conv1d(16, 32, 3)
        self.conv4 = nn.Conv1d(32, 64, 3)
        self.conv5 = nn.Conv1d(64, 64, 3)
        self.conv6 = nn.Conv1d(64, 128, 3)
        self.conv7 = nn.Conv1d(128, 256, 3)
        self.conv8 = nn.Conv1d(256, 256, 3)

        self.bn0 = nn.BatchNorm1d(1)
        self.bn1 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn5 = nn.BatchNorm1d(64)
        self.bn7 = nn.BatchNorm1d(256)

        self.pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(256, 256)
        self.out = nn.Linear(256, n_out)

        self.gaus_noise = gaussian_noise

    def forward(self, X):
        """"""
        x = self.bn0(X)

        if self.training:
            e = torch.randn(X.shape)
            if X.is_cuda: e = e.cuda()
            x = x + e * self.gaus_noise

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.pool(F.relu(self.conv6(x)))
        x = self.pool(F.relu(self.bn7(self.conv7(x))))
        x = F.relu(self.conv8(x))

        x = F.dropout(x.view(X.size(0), -1), training=self.training, p=0.5)
        x = F.dropout(F.relu(self.fc1(x)), training=self.training, p=0.5)
        x = self.out(x)

        return x


class CNN1DASCAD(nn.Module):
    """"""
    def __init__(self, in_ch, n_out, n_len=700, gaussian_noise=0.5, mean=0, std=1):
        """"""
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 64, 11)
        self.conv2 = nn.Conv1d(64, 128, 11)
        self.conv3 = nn.Conv1d(128, 256, 11)
        self.conv4 = nn.Conv1d(256, 512, 11)
        self.conv5 = nn.Conv1d(512, 512, 11)

        self.pool = nn.AvgPool1d(2, stride=2)

        # infer conv output shape
        conv_out = self._conv_block(torch.randn(1, in_ch, n_len))
        conv_out = conv_out.view(1, -1)

        # init fc layer
        self.fc1 = nn.Linear(conv_out.shape[-1], 4096)
        # self.fc1 = nn.Linear(43008, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.out = nn.Linear(4096, n_out)

        self.gaus_noise = gaussian_noise
        self.mean = torch.Tensor([mean]).float()
        self.std = std

    def _conv_block(self, X):
        """"""
        x = self.pool(F.relu(self.conv1(X)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        return x

    def forward(self, X):
        """"""
        if self.training:
            e = self.std * torch.randn(X.shape)
            # e = e + self.mean.expand(X.size())
            X = X + e.cuda() * self.gaus_noise

        x = self._conv_block(X)
        x = x.view(X.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
