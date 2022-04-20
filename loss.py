import torch
from torch import nn

import numpy as np

def kl_anneal_function(epoch, k, x0):
    # logistic
    return float(1/(1+np.exp(-k*(epoch-x0))))


def loss_fn(out, target, mu, logv, epoch, k, xo):
    MSE = nn.MSELoss()
    MSE_loss = MSE(out, target)

    # KL Divergence
    KL_loss = -0.5 * torch.sum(1+logv-mu.pow(2)-logv.exp())
    KL_weight = kl_anneal_function(epoch, k, xo)

    return MSE_loss, KL_loss, KL_weight