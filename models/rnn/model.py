import random
import numpy as np

import torch
from torch import nn

class StyleTransfer(nn.Module):
    def __init__(self, encoder, tst_decoder, d_hidden, style_ratio, variational, device):
        super(StyleTransfer, self).__init__()

        self.device = device

        self.encoder = encoder
        self.tst_decoder = tst_decoder

        self.d_hidden = d_hidden
        self.style_ratio = style_ratio
        self.content_index = int(self.d_hidden * (1 - self.style_ratio))
        self.style_index = int(self.d_hidden-self.content_index)

        self.variational = variational


        # TODO Size ?
        self.half_hidden = nn.Linear(4, 2)
        self.content2mean = nn.Linear(self.content_index, d_hidden)
        self.content2logv = nn.Linear(self.content_index, d_hidden)

        self.style2mean = nn.Linear(self.style_index, d_hidden)
        self.style2logv = nn.Linear(self.style_index, d_hidden)

    def reparameterization(self, hidden, latent_type):
        hidden = hidden.transpose(0, -1)
        hidden = self.half_hidden(hidden)

        hidden = hidden.transpose(0, -1)
        if latent_type == "content":
            mean = self.content2mean(hidden).to(self.device)
            logv = self.content2logv(hidden).to(self.device)

        elif latent_type == "style":
            mean = self.style2mean(hidden).to(self.device)
            logv = self.style2logv(hidden).to(self.device)

        std = torch.exp(0.5 * logv)
        eps = torch.randn_like(std)
        z = mean + (eps * std)
        return z, mean, logv

    def forward(self, tst_src, tst_trg, teacher_forcing_ratio=0.5):
        tst_src = tst_src.to(self.device)
        tst_trg = tst_trg.to(self.device)

        encoder_out, hidden, cell = self.encoder(tst_src)

<<<<<<< HEAD:models/rnn/model.py
        if self.variational:
            context_c, context_a = hidden[:, :, :self.content_index], hidden[:, :, -self.style_index:]

            # TODO 따로 따로 reparameterize? 아니면 reparameterize 한 다음에 split?
            # TODO 나눈 후 size 맞추기 위해 content 밑에/style 위에 0으로 채워서 reparameterize?
            content_c, content_mu, content_logv = self.reparameterization(context_c, "content")
            style_a, style_mu, style_logv = self.reparameterization(context_a, "style")

            total_latent = torch.cat((content_c, style_a), 0)

            # TODO cat? add? -> 일단은 total_latent로 진행
            hidden = total_latent

            latent_variables = [total_latent, content_c, content_mu, content_logv, style_a, style_mu, style_logv,]

        else:
            latent_variables = [None for _ in range(7)]

=======
        context_c, context_a = hidden[:, :, :self.content_index], hidden[:, :, -self.style_index:]

        # TODO 따로 따로 reparameterize? 아니면 reparameterize 한 다음에 split?
        # TODO 나눈 후 size 맞추기 위해 content 밑에/style 위에 0으로 채워서 reparameterize?
        content_c, content_mu, content_logv = self.reparameterization(context_c, "content")
        style_a, style_mu, style_logv = self.reparameterization(context_a, "style")

        total_latent = torch.cat((content_c, style_a), 0)

        # TODO cat? add? -> 일단은 total_latent로 진행
        hidden = total_latent

>>>>>>> d6763e081e2c6aef0b836f9309640b7d6a8f81fd:model.py
        trg_len = tst_trg.shape[1]  # length of word
        batch_size = tst_trg.shape[0]  # batch size
        trg_vocab_size = self.tst_decoder.output_size
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

<<<<<<< HEAD:models/rnn/model.py
        input = tst_trg[:, 0]  # BOS 먼저
=======
        input = tst_trg[:, 0] # BOS 먼저
>>>>>>> d6763e081e2c6aef0b836f9309640b7d6a8f81fd:model.py

        output_list = []
        for i in range(1, trg_len):
            output, hidden, cell = self.tst_decoder(input, hidden, cell)
            outputs[:, i] = output
            output_list.append(torch.argmax(output, dim=1).tolist())
            top1 = output.argmax(1)

            teacher_force = random.random() < teacher_forcing_ratio
            input = tst_trg[:, i] if teacher_force else top1
<<<<<<< HEAD:models/rnn/model.py

        return outputs, latent_variables, output_list

class StylizedNMT(nn.Module):
    def __init__(self, nmt_encoder, nmt_decoder, d_hidden, total_latent, device):
=======

        return outputs, total_latent, content_c, content_mu, content_logv, style_a, style_mu, style_logv, output_list

class StylizedNMT(nn.Module):
    def __init__(self, nmt_decoder, d_hidden, total_latent, device):
>>>>>>> d6763e081e2c6aef0b836f9309640b7d6a8f81fd:model.py
        super(StylizedNMT, self).__init__()

        self.device = device

<<<<<<< HEAD:models/rnn/model.py
        self.nmt_encoder = nmt_encoder
=======
>>>>>>> d6763e081e2c6aef0b836f9309640b7d6a8f81fd:model.py
        self.nmt_decoder = nmt_decoder
        self.total_latent = total_latent

        self.hidden2concat = nn.Linear(d_hidden, d_hidden // 2)
        self.latent2concat = nn.Linear(d_hidden, d_hidden // 2)

<<<<<<< HEAD:models/rnn/model.py
    def forward(self, nmt_src, nmt_trg, teacher_forcing_ratio=0.5):

        # nmt_hidden = nmt_hidden.to(self.device)
        # nmt_cell = nmt_cell.to(self.device)
        nmt_src = nmt_src.to(self.device)
        nmt_trg = nmt_trg.to(self.device)

        encoder_out, hidden, cell = self.nmt_encoder(nmt_src)

        # TODO add 할 지, concat 할 지
        if not self.total_latent is None:
            hidden = self.hidden2concat(hidden)
            latent = self.latent2concat(self.total_latent)
            hidden = torch.cat((hidden, latent), 2)
=======
    def forward(self, nmt_hidden, nmt_cell, nmt_trg, teacher_forcing_ratio=0.5):

        nmt_hidden = nmt_hidden.to(self.device)
        nmt_cell = nmt_cell.to(self.device)
        nmt_trg = nmt_trg.to(self.device)

        # TODO add 할 지, concat 할 지
        hidden = self.hidden2concat(nmt_hidden)
        latent = self.latent2concat(self.total_latent)
        hidden = torch.cat((hidden, latent), 2)
>>>>>>> d6763e081e2c6aef0b836f9309640b7d6a8f81fd:model.py

        trg_len = nmt_trg.shape[1]  # length of word
        batch_size = nmt_trg.shape[0]  # batch size
        trg_vocab_size = self.nmt_decoder.output_size

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        input = nmt_trg[:, 0]

        output_list = []
        for i in range(1, trg_len):
<<<<<<< HEAD:models/rnn/model.py
            output, hidden, cell = self.nmt_decoder(input, hidden, cell)
=======
            output, hidden, cell = self.nmt_decoder(input, hidden, nmt_cell)
>>>>>>> d6763e081e2c6aef0b836f9309640b7d6a8f81fd:model.py
            outputs[:, i] = output
            output_list.append(torch.argmax(output, dim=1).tolist())
            top1 = output.argmax(1)

            teacher_force = random.random() < teacher_forcing_ratio
            input = nmt_trg[:, i] if teacher_force else top1

        return outputs, output_list


class Encoder(nn.Module):
    def __init__(self, input_size, d_hidden, d_embed, n_layers, dropout, device):
        super(Encoder, self).__init__()
        self.src_embedding = nn.Embedding(input_size, d_embed)

        # TODO num_layers=2 -> total_latent [8, batch_size, d_hidden] 이거 어떻게 해결?
        self.encoder = nn.LSTM(input_size=d_embed, hidden_size=d_hidden, dropout=dropout,
                               num_layers=n_layers, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.device = device

    def forward(self, src):
        embedded = self.dropout(self.src_embedding(src))
        outputs, (hidden, cell) = self.encoder(embedded)

        return outputs, hidden, cell


class TSTDecoder(nn.Module):
    def __init__(self, output_size, d_hidden, d_embed, n_layers, dropout, device):
        super(TSTDecoder, self).__init__()
        self.output_size = output_size
        self.trg_embedding = nn.Embedding(output_size, d_embed)
        self.tst_decoder = nn.LSTM(input_size=d_embed, hidden_size=d_hidden, dropout=dropout,
                                   num_layers=n_layers, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2*d_hidden, output_size)

        self.device = device

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.trg_embedding(input))

        outputs, (hidden, cell) = self.tst_decoder(embedded, (hidden, cell))

        tst_out = self.fc(outputs.squeeze(1))

        return tst_out, hidden, cell


class NMTDecoder(nn.Module):
    def __init__(self, output_size, d_hidden, d_embed, n_layers, dropout, device):
        super(NMTDecoder, self).__init__()
        self.output_size = output_size
        self.trg_embedding = nn.Embedding(output_size, d_embed)
        self.nmt_decoder = nn.LSTM(input_size=d_embed, hidden_size=d_hidden, dropout=dropout,
                                   num_layers=n_layers, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2*d_hidden, output_size)

        self.device = device

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.trg_embedding(input))
        outputs, (hidden, cell) = self.nmt_decoder(embedded, (hidden, cell))
        nmt_out = self.fc(outputs.squeeze(1))

        return nmt_out, hidden, cell
