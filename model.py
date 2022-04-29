import random

import torch
from torch import nn

class StyleTransfer(nn.Module):
    def __init__(self, encoder, tst_decoder, d_hidden, style_ratio, device):
        super(StyleTransfer, self).__init__()

        self.device = device

        self.encoder = encoder
        self.tst_decoder = tst_decoder

        self.d_hidden = d_hidden
        self.style_ratio = style_ratio
        self.content_index = int(self.d_hidden * (1 - self.style_ratio))
        self.style_index = int(self.d_hidden-self.content_index)


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
        tst_src = tst_src.transpose(0, 1)
        tst_trg = tst_trg.transpose(0, 1)

        tst_src = tst_src.to(self.device)
        tst_trg = tst_trg.to(self.device)
        # size tst_src & trg ; [max_len, batch]

        encoder_out, hidden, cell = self.encoder(tst_src)
        # hidden size = [n_layers*bi, batch, d_hidden]

        # print("style_ratio, style_index, content_index:", self.style_ratio, self.style_index, self.content_index)
        context_c, context_a = hidden[:, :, :self.content_index], hidden[:, :, -self.style_index:]
        # context_c&a size ; [n_layer*bi , batch, d_hidden]

        # TODO 따로 따로 reparameterize? 아니면 reparameterize 한 다음에 split?
        # TODO 나눈 후 size 맞추기 위해 content 밑에/style 위에 0으로 채워서 reparameterize?
        content_c, content_mu, content_logv = self.reparameterization(context_c, "content")
        style_a, style_mu, style_logv = self.reparameterization(context_a, "style")
        # content_c & style_a size ; [n_layer*bi , batch, d_hidden]

        total_latent = torch.cat((content_c, style_a), 0)
        # size ; [n_layer*bi , batch, d_hidden]

        # TODO cat? add? -> 일단은 total_latent로 진행
        # hidden = torch.add(hidden, total_latent)
        hidden = total_latent

        trg_len = tst_trg.shape[0]  # length of word
        batch_size = tst_trg.shape[1]  # batch size
        trg_vocab_size = self.tst_decoder.output_size
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        # size ; [max_len, batch, vocab_size]

        input = tst_trg[0, :]
        # size ; [16]

        for i in range(1, trg_len):
            output, hidden, cell = self.tst_decoder(input, hidden, cell)
            # size ; output [batch, vocab_size] / hidden [n_layer*bi, batch, hidden] / cell [n_layer*bi, batch, hidden]
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

        nmt_src = nmt_src.to(self.device)
        nmt_trg = nmt_trg.to(self.device)
        # embedded = self.src_embedding(nmt_src)
        encoder_out, hidden, cell = self.encoder(nmt_src)
        
        # TODO add 할 지, concat 할 지
        hidden = torch.add(hidden, self.total_latent)

        trg_len = nmt_trg.shape[0]  # length of word
        batch_size = nmt_trg.shape[1]  # batch size
        trg_vocab_size = self.nmt_decoder.output_size
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        input = nmt_trg[0, :]

        for i in range(1, trg_len):
            output, hidden, cell = self.nmt_decoder(input, hidden, cell)
            outputs[i] = output
            top1 = output.argmax(1)

            teacher_force = random.random() < teacher_forcing_ratio
            input = nmt_trg[i] if teacher_force else top1

        return outputs


class Encoder(nn.Module):
    def __init__(self, input_size, d_hidden, d_embed, n_layers, dropout, device):
        super(Encoder, self).__init__()
        self.src_embedding = nn.Embedding(input_size, d_embed)

        # TODO num_layers=2 -> total_latent [8, batch_size, d_hidden] 이거 어떻게 해결?
        self.encoder = nn.LSTM(input_size=d_embed, hidden_size=d_hidden, dropout=dropout,
                               num_layers=n_layers, bidirectional=True)
        # self.encoder = nn.LSTM(input_size=d_embed, hidden_size=d_hidden, dropout=dropout,
        #                        num_layers=1, bidirectional=True)
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
                                   num_layers=n_layers, bidirectional=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2*d_hidden, output_size)

        self.device = device

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        # input size ; [batch] -> [1, batch]
        embedded = self.dropout(self.trg_embedding(input))
        # embedded size ; [1, batch, d_embed]

        outputs, (hidden, cell) = self.tst_decoder(embedded, (hidden, cell))
        # size; output [1, batch, 2*d_hidden], hidden [n_layer*bi, batch, hidden], cell [n_layer*bi, batch, hidden]


        tst_out = self.fc(outputs.squeeze(0))
        # tst_out = self.fc(outputs)
        # size ; squeeze x [1, batch, vocab_size] / squeeze [batch, vocab_size]

        return tst_out, hidden, cell


class NMTDecoder(nn.Module):
    def __init__(self, output_size, d_hidden, d_embed, n_layers, dropout, device):
        super(NMTDecoder, self).__init__()
        self.output_size = output_size
        self.trg_embedding = nn.Embedding(output_size, d_embed)
        self.nmt_decoder = nn.LSTM(input_size=d_embed, hidden_size=d_hidden, dropout=dropout,
                                   num_layers=n_layers, bidirectional=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2*d_hidden, output_size)

        self.device = device

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.trg_embedding(input))

        outputs, (hidden, cell) = self.nmt_decoder(embedded, (hidden, cell))
        nmt_out = self.fc(outputs.squezze(0))

        return nmt_out, hidden, cell
