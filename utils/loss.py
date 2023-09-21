import os

import torch
import torch.nn as nn
import torch.nn.modules.loss as loss
import numpy as np
import torch.nn.functional as F
import scipy
from numpy import dot
from numpy.linalg import norm

import torch.utils.checkpoint as cp
from scipy.signal import find_peaks
from scipy.signal import butter, sosfiltfilt

def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def loss_fn(loss_name):
    """
    :param loss_fn: implement loss function for training
    :return: loss function module(class)
    """

    if loss_name == "MSE":
        return loss.MSELoss()
    elif loss_name == "fft":
        return fftLoss()
    elif loss_name == "L1":
        return loss.L1Loss()
    elif loss_name == "neg_pearson":
        return NegPearsonLoss()
    else:
        raise NotImplementedError("implement a custom function(%s) in loss.py" % loss_fn)


def neg_Pearson_Loss(predictions, targets):
    '''
    :param predictions: inference value of trained model
    :param targets: target label of input data
    :return: negative pearson loss
    '''
    rst = 0
    targets = targets[:, :]
    #predictions = torch.squeeze(predictions)
    # Pearson correlation can be performed on the premise of normalization of input data
    predictions = (predictions - torch.mean(predictions)) / torch.std(predictions)
    targets = (targets - torch.mean(targets)) / torch.std(targets)

    for i in range(predictions.shape[0]):
        sum_x = torch.sum(predictions[i])  # x
        sum_y = torch.sum(targets[i])  # y
        sum_xy = torch.sum(predictions[i] * targets[i])  # xy
        sum_x2 = torch.sum(torch.pow(predictions[i], 2))  # x^2
        sum_y2 = torch.sum(torch.pow(targets[i], 2))  # y^2
        N = predictions.shape[1] if len(predictions.shape) > 1 else 1
        pearson = (N * sum_xy - sum_x * sum_y) / (
            torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))

        rst += 1 - pearson

    rst = rst / predictions.shape[0]
    return rst

class NegPearsonLoss(nn.Module):
    def __init__(self):
        super(NegPearsonLoss, self).__init__()

    def forward(self, predictions, targets):
        return neg_Pearson_Loss(predictions, targets)


class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, predictions, targets):
        neg = neg_Pearson_Loss(predictions, targets)
        loss_func = nn.L1Loss()
        predictions = torch.fft.fft(predictions, dim=1, norm="forward")
        targets = torch.fft.fft(targets, dim=1, norm="forward")
        loss = loss_func(predictions, targets)
        return loss + neg
