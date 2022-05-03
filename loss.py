import torch
from torch import nn

import numpy as np

def kl_anneal_function(epoch, k, x0):
    # logistic
    return float(1/(1+np.exp(-k*(epoch-x0))))

def bce_loss(out, target):
    BCE = nn.BCELoss()
    BCE_loss = BCE(out, target)
    return BCE_loss

def kl_loss(mu, logv, epoch, k, xo):
    KL_loss = -0.5 * torch.sum(1 + logv - mu.pow(2) - logv.exp())
    KL_weight = kl_anneal_function(epoch, k, xo)
    return KL_loss, KL_weight

def ce_loss(out, target):
    CE = nn.CrossEntropyLoss()
    CE_loss = CE(out, target)
    return CE_loss