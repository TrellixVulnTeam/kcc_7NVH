import os
import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .model import Encoder, TSTDecoder, NMTDecoder, StyleTransfer, StylizedNMT
from .loss import bce_loss, kl_loss, ce_loss
from .custom_dataset import custom_dataset

def tst_train_one_epoch(tst_model, epoch, data_loader, tst_optimizer, device):
    tst_model.train()

    train_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for _, (X, y) in enumerate(data_loader):
            X = X.to(device)
            y = y.to(device)

            tst_out, total_latent, content_c, content_mu, content_logv, style_a, style_mu, style_logv = tst_model(X)

            # TODO Loss 재정의
            bce_loss = bce_loss(tst_out, y)
            kl_loss, kl_weight = kl_loss(style_mu, style_logv, epoch, 0.0025, 2500)
            loss = (bce_loss + kl_loss*kl_weight)
            loss_value = loss.item()
            train_loss += loss_value

            tst_optimizer.zero_grad()   # optimizer 초기화
            loss.backward()
            tst_optimizer.step()    # Gradient Descent 시작
            pbar.update(1)

    return train_loss/total, total_latent

def nmt_train_one_epoch(nmt_model, data_loader, nmt_optimizer, device):
    nmt_model.train()

    # Freeze
    for name, param in nmt_model.named_parameters():
        if "encoder" in name:
            param.requires_grad = False

    train_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for _, (X, y) in enumerate(data_loader):
            X = X.to(device)
            y = y.to(device)

            nmt_out = nmt_model(X)

            ce_loss = ce_loss(nmt_out, y)
            loss = ce_loss
            loss_value = loss.item()
            train_loss += loss_value

            nmt_optimizer.zero_grad()   # optimizer 초기화
            loss.backward()
            nmt_optimizer.step()    # Gradient Descent 시작
            pbar.update(1)

    return train_loss/total

@torch.no_grad()    #no autograd (backpropagation X)
def tst_evaluate(tst_model, epoch, data_loader, device):
    tst_model.eval()

    valid_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for _, (X, y) in enumerate(data_loader):
            X = X.float().to(device)
            y = y.float().to(device)

            tst_out, total_latent, content_c, content_mu, content_logv, style_a, style_mu, style_logv = tst_model(X)

            bce_loss = bce_loss(tst_out, y)
            kl_loss, kl_weight = kl_loss(style_mu, style_logv, epoch, 0.0025, 2500)

            loss = (bce_loss + kl_loss * kl_weight)
            loss_value = loss.item()
            valid_loss += loss_value

            pbar.update(1)

    return valid_loss/total

@torch.no_grad()    #no autograd (backpropagation X)
def nmt_evaluate(nmt_model, data_loader, device):
    nmt_model.eval()

    valid_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for _, (X, y) in enumerate(data_loader):
            X = X.float().to(device)
            y = y.float().to(device)

            nmt_out = nmt_model(X)

            ce_loss = ce_loss(nmt_out, y)
            loss = ce_loss
            loss_value = loss.item()
            valid_loss += loss_value

            pbar.update(1)

    return valid_loss/total

def train():
    # Device Setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Initializing Device: {device}')

    # Data Setting





    # TST Train
    encoder = Encoder(d_model=512, d_hidden=1024, d_embed=256, n_layers=2, dropout=0.1)
    tst_decoder = TSTDecoder(d_model=512, d_hidden=1024, d_embed=256, n_layers=2, dropout=0.1)
    tst_model = StyleTransfer(encoder, tst_decoder, d_hidden=1024, style_ratio=0.3, device=device)
    tst_optimizer = torch.optim.AdamW(tst_model.parameters(), lr=0.001)

    start_epoch = 0
    epochs = 100
    print("Start TST Training..")
    for epoch in range(start_epoch, epochs+1):
        print(f"Epoch: {epoch}")
        epoch_loss, total_latent = tst_train_one_epoch(tst_model, epoch, data_loader, tst_optimizer, device)
        print(f"Training Loss: {epoch_loss:.5f}")

        valid_loss = tst_evaluate(tst_model, epoch, data_loader, device)
        print(f"Validation Loss: {valid_loss:.5f}")

    # NMT Train
    nmt_decoder = NMTDecoder(d_model=512, d_hidden=1024, d_embed=256, n_layers=2, dropout=0.1)
    nmt_model = StylizedNMT(encoder, nmt_decoder, total_latent=total_latent, device=device)
    nmt_optimizer = torch.optim.AdamW(nmt_model.parameters(), lr=0.001)

    start_epoch = 0
    epochs = 100
    print("Start NMT Training..")
    for epoch in range(start_epoch, epochs + 1):
        print(f"Epoch: {epoch}")
        epoch_loss = nmt_train_one_epoch(nmt_model, data_loader, nmt_optimizer, device)
        print(f"Training Loss: {epoch_loss:.5f}")

        valid_loss = nmt_evaluate(nmt_model, data_loader, device)
        print(f"Validation Loss: {valid_loss:.5f}")

