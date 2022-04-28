import os
import tqdm
import pickle

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
from .custom_dataset import CustomDataset

def tst_train_one_epoch(tst_model, epoch, data_loader, tst_optimizer, device):
    tst_model.train()

    train_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for _, (src, trg) in enumerate(data_loader):
            src = src.to(device)
            trg = trg.to(device)

            tst_out, total_latent, content_c, content_mu, content_logv, style_a, style_mu, style_logv = tst_model(src)

            # TODO Loss 재정의
            bce_loss = bce_loss(tst_out, trg)
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
        for _, (src, trg) in enumerate(data_loader):
            src = src.to(device)
            trg = trg.to(device)

            nmt_out = nmt_model(src)

            ce_loss = ce_loss(nmt_out, trg)
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
        for _, (src, trg) in enumerate(data_loader):
            src = src.float().to(device)
            trg = trg.float().to(device)

            tst_out, total_latent, content_c, content_mu, content_logv, style_a, style_mu, style_logv = tst_model(src)

            bce_loss = bce_loss(tst_out, trg)
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
        for _, (src, trg) in enumerate(data_loader):
            src = src.to(device)
            trg = trg.to(device)

            nmt_out = nmt_model(src)

            ce_loss = ce_loss(nmt_out, trg)
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
    with open("./data/processed/tokenized/spm_tokenized_data.pkl", "rb") as f:
        data = pickle.load(f)

    em_informal_train = data["train"]["em_informal"]
    em_formal_train = data["train"]["em_formal"]
    # fr_informal_train = data["train"]["fr_informal"]
    # fr_formal_train = data["train"]["fr_formal"]

    em_informal_test = data["test"]["em_informal"]
    em_formal_test = data["test"]["em_formal"]
    # fr_informal_test = data["test"]["fr_informal"]
    # fr_formal_test = data["test"]["fr_formal"]


    split_ratio = 0.8
    em_informal_train = em_informal_train[:int(len(em_informal_train)*split_ratio)]
    em_informal_valid = em_informal_train[int(len(em_informal_train)*split_ratio):]
    em_formal_train = em_formal_train[:int(len(em_formal_train) * split_ratio)]
    em_formal_valid = em_formal_train[int(len(em_formal_train) * split_ratio):]

    min_len, max_len = 2, 300
    batch_size = 16
    num_workers = 8

    train_data = CustomDataset(em_informal_train, em_formal_train, min_len, max_len)
    valid_data = CustomDataset(em_informal_valid, em_formal_valid, min_len, max_len)
    test_data = CustomDataset(em_informal_test, em_formal_test, min_len, max_len)

    train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers)


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
        epoch_loss, total_latent = tst_train_one_epoch(tst_model, epoch, train_loader, tst_optimizer, device)
        print(f"Training Loss: {epoch_loss:.5f}")

        valid_loss = tst_evaluate(tst_model, epoch, valid_loader, device)
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
        epoch_loss = nmt_train_one_epoch(nmt_model, train_loader, nmt_optimizer, device)
        print(f"Training Loss: {epoch_loss:.5f}")

        valid_loss = nmt_evaluate(nmt_model, valid_loader, device)
        print(f"Validation Loss: {valid_loss:.5f}")

