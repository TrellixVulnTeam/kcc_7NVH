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

from .model import StyleTransferTransformer, MachineTranslationTransformer, Encoder, TSTDecoder, NMTDecoder
from .loss import bce_loss, kl_loss
from .custom_dataset import custom_dataset

def train_one_epoch(tst_model, nmt_model, epoch, data_loader, tst_optimizer, nmt_optimizer, device):
    tst_model.train()

    train_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for _, (X, y) in enumerate(data_loader):
            X = X.float().to(device)
            y = y.float().to(device)

            tst_out, total_latent, content_c, content_mu, content_logv, style_a, style_mu, style_logv = tst_model(X)

            # TODO Loss 재정의
            BCE_loss = bce_loss(tst_out, y)
            KL_loss, KL_weight = kl_loss(style_mu, style_logv, epoch, 0.0025, 2500)

            loss = (BCE_loss + KL_loss*KL_weight)
            loss_value = loss.item()
            train_loss += loss_value

            tst_optimizer.zero_grad()   # optimizer 초기화
            loss.backward()
            tst_optimizer.step()    # Gradient Descent 시작
            pbar.update(1)

    return train_loss/total

@torch.no_grad()    #no autograd (backpropagation X)
def evaluate(tst_model, nmt_model, epoch, data_loader, device):
    tst_model.eval()

    valid_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for _, (X, y) in enumerate(data_loader):
            X = X.float().to(device)
            y = y.float().to(device)

            tst_out, total_latent, content_c, content_mu, content_logv, style_a, style_mu, style_logv = tst_model(X)

            BCE_loss = bce_loss(tst_out, y)
            KL_loss, KL_weight = kl_loss(style_mu, style_logv, epoch, 0.0025, 2500)

            loss = (BCE_loss + KL_loss * KL_weight)
            loss_value = loss.item()
            valid_loss += loss_value

            # y_list += y.detach().reshape(-1).tolist()
            # output_list += tst_out.detach().reshape(-1).tolist()
            pbar.update(1)

    return valid_loss/total

def train():
    # Device Setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Initializing Device: {device}')

    # Data Setting




    # Model Setting
    encoder = Encoder()
    tst_decoder = TSTDecoder()
    nmt_decoder = NMTDecoder()
    tst_model = StyleTransferTransformer(encoder, tst_decoder, d_model=512, d_hidden=1024, style_ratio=0.3)
    tst_optimizer = torch.optim.AdamW(tst_model.parameters(), lr=0.001)

    nmt_model = MachineTranslationTransformer(encoder, nmt_decoder, d_model=512, d_hidden=1024)
    nmt_optimizer = torch.optim.AdamW(nmt_model.parameters(), lr=0.001)

    # Train
    start_epoch = 0
    epochs = 100
    print("Start Training..")
    for epoch in range(start_epoch, epochs+1):
        print(f"Epoch: {epoch}")
        epoch_loss = train_one_epoch(tst_model, nmt_model, epoch, train_loader, tst_optimizer, nmt_optimizer, device)
        print(f"Training Loss: {epoch_loss:.5f}")

        valid_loss, y_list, output_list = evaluate(tst_model, nmt_model, epoch, valid_loader, device)
        # rmse = np.sqrt(valid_loss)
        print(f"Validation Loss: {valid_loss:.5f}")
        # print(f'RMSE is {rmse:.5f}')


