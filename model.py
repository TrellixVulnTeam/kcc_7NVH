import random

import torch
from torch import nn

class StyleTransfer(nn.Module):
    def __init__(self, encoder, tst_decoder, d_hidden, style_ratio, device):
        super(StyleTransfer, self).__init__()

        self.device = device

        self.encoder = encoder
        self.tst_decoder = tst_decoder
        self.style_ratio = style_ratio

        # TODO Size ?
        self.context2mu = nn.Linear(d_hidden, d_hidden)
        self.context2logv = nn.Linear(d_hidden, d_hidden)

    def reparameterization(self, hidden):
        mu = self.context2mean(hidden)
        log_v = self.context2logv(hidden)

        std = torch.exp(0.5 * log_v)
        eps = torch.randn_like(std)
        z = mu + (eps * std)

        return z, mu, log_v

    def forward(self, tst_src, tst_trg, teacher_forcing_ratio=0.5):
        tst_src = tst_src.transpose(0, 1)
        tst_trg = tst_trg.transpose(0, 1)
        embedded = self.src_embedding(tst_src)
        encoder_out, hidden, cell = self.encoder(embedded)

        style_index = int(len(hidden) * (1 - self.style_ratio))
        context_c, context_a = hidden[:style_index], hidden[style_index:]

        content_c, content_mu, content_logv = self.reparameterization(context_c)
        style_a, style_mu, style_logv = self.reparameterization(context_a)

        total_latent = torch.cat(content_c, style_a)

        # TODO cat? add? -> 일단은 total_latent로 진행
        # hidden = torch.add(hidden, total_latent)
        hidden = total_latent

        trg_len = tst_trg.shape[0]  # length of word
        batch_size = tst_trg.shape[1]  # batch size
        trg_vocab_size = self.tst_decoder.d_model
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        input = tst_trg[0, :]

        for i in range(1, trg_len):
            output, hidden, cell = self.tst_decoder(input, hidden, cell)
            outputs[i] = output
            top1 = output.argmax(1)

            teacher_force = random.random() < teacher_forcing_ratio
            input = tst_trg[i] if teacher_force else top1

        return outputs, total_latent, content_c, content_mu, content_logv, style_a, style_mu, style_logv


class StylizedNMT(nn.Module):
    def __init__(self, encoder, nmt_decoder, total_latent, device):
        super(StylizedNMT, self).__init__()

        self.device = device

        self.encoder = encoder
        self.nmt_decoder = nmt_decoder
        self.total_latent = total_latent

    def forward(self, nmt_src, nmt_trg, teacher_forcing_ratio=0.5):
        nmt_src = nmt_src.transpose(0, 1)
        nmt_trg = nmt_trg.transpose(0, 1)
        embedded = self.src_embedding(nmt_src)
        encoder_out, hidden, cell = self.encoder(embedded)

        hidden = torch.add(hidden, self.total_latent)

        trg_len = nmt_trg.shape[0]  # length of word
        batch_size = nmt_trg.shape[1]  # batch size
        trg_vocab_size = self.tst_decoder.d_model
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        input = nmt_trg[0, :]

        for i in range(1, trg_len):
            output, hidden, cell = self.tst_decoder(input, hidden, cell)
            outputs[i] = output
            top1 = output.argmax(1)

            teacher_force = random.random() < teacher_forcing_ratio
            input = nmt_trg[i] if teacher_force else top1

        return outputs


class Encoder(nn.Module):
    def __init__(self, d_model, d_hidden, d_embed, n_layers, dropout):
        super(Encoder, self).__init__()
        self.src_embedding = nn.Embedding(d_model, d_embed)
        self.encoder = nn.LSTM(input_size=d_embed, hidden_size=d_hidden, dropout=dropout,
                               num_layers=n_layers, bidirectional=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.src_embedding(src))
        outputs, hidden, cell = self.encoder(embedded)

        return hidden, cell


class TSTDecoder(nn.Module):
    def __init__(self, d_model, d_hidden, d_embed, n_layers, dropout):
        super(TSTDecoder, self).__init__()
        self.trg_embedding = nn.Embedding(d_model, d_embed)
        self.tst_decoder = nn.LSTM(input_size=d_embed, hidden_size=d_hidden, dropout=dropout,
                                   num_layers=n_layers, bidirectional=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_hidden, d_model)

    def forward(self, input, hidden, cell):
        input = input.insqueeze(0)
        embedded = self.dropout(self.trg_embedding(input))

        outputs, hidden, cell = self.tst_decoder(embedded, hidden, cell)
        tst_out = self.fc(outputs.squezze(0))

        return tst_out, hidden, cell


class NMTDecoder(nn.Module):
    def __init__(self, d_model, d_hidden, d_embed, n_layers, dropout):
        super(NMTDecoder, self).__init__()
        self.trg_embedding = nn.Embedding(d_model, d_embed)
        self.nmt_decoder = nn.LSTM(input_size=d_embed, hidden_size=d_hidden, dropout=dropout,
                                   num_layers=n_layers, bidirectional=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_hidden, d_model)

    def forward(self, input, hidden, cell):
        input = input.insqueeze(0)
        embedded = self.dropout(self.trg_embedding(input))

        outputs, hidden, cell = self.nmt_decoder(embedded, hidden, cell)
        nmt_out = self.fc(outputs.squezze(0))

        return nmt_out, hidden, cell
