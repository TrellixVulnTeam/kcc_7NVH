import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

def kl_anneal_function(epoch, k, x0):
    # logistic
    return float(1/(1+np.exp(-k*(epoch-x0))))

def bce_loss(out, target):
    BCE = nn.BCELoss()
    BCE_loss = BCE(out, target)
    return BCE_loss

def kl_loss_2(mu, logv, epoch, k, xo):
    KL_loss = -0.5 * torch.sum(1 + logv - mu.pow(2) - logv.exp())
    KL_weight = kl_anneal_function(epoch, k, xo)
    return KL_loss*KL_weight

def kl_loss(out, target, mean, logv, trg_pad_idx):
    # mean, logv; [batch, seq_len, latent_size]
    # reproduction_loss = F.cross_entropy(out, target, ignore_index=trg_pad_idx)
    KLD = (-0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp(), 1)).mean().squeeze()
    # KLD = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
    return KLD

def ce_loss(out, target):
    CE = nn.CrossEntropyLoss(ignore_index=0)
    CE_loss = CE(out, target)
    return CE_loss