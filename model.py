# Import PyTorch
import torch
from torch import nn
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch.autograd import Variable


class StyleTransferTransformer(nn.Module):
    def __init__(self, encoder, tst_decoder, d_model, d_hidden, style_ratio, dropout):
        super(StyleTransferTransformer, self).__init__()

        self.encoder = nn.LSTM(input_size=d_model, hidden_size=d_hidden, dropout=dropout)
        self.tst_decoder = tst_decoder
        self.style_ratio = style_ratio

        self.context2mu = nn.Linear(d_model, d_hidden)
        self.context2logv = nn.Linear(d_model, d_hidden)

    def reparameterization(self, encoder_out):
        mu = self.context2mean(encoder_out)
        log_v = self.context2logv(encoder_out)

        std = torch.exp(0.5 * log_v)
        eps = torch.randn_like(std)
        z = mu + (eps * std)

        return z, mu, log_v

    def forward(self, src, tgt):
        encoder_out = self.encoder(src)

        style_index = int(len(encoder_out) * (1-self.style_ratio))
        context_c, context_a = encoder_out[:style_index], encoder_out[style_index:]

        content_c, content_mu, content_logv = self.reparameterization(context_c)
        style_a, style_mu, style_logv = self.reparameterization(context_a)

        total_latent = torch.cat(content_c, style_a)

        #TODO add가 맞나?
        encoder_out = torch.add(encoder_out, total_latent)

        tst_out = self.tst_decoder(tgt, encoder_out)

        return tst_out, total_latent, content_c, content_mu, content_logv, style_a, style_mu, style_logv


class MachineTranslationTransformer(nn.Module):
    def __init__(self, encoder, nmt_decoder, d_model, d_hidden):
        super(MachineTranslationTransformer, self).__init__()
        self.encoder = encoder
        self.nmt_decoder = nmt_decoder



    def forward(self, src, tgt):
        encoder_out = self.encoder(src)
        nmt_out = self.nmt_decoder(tgt, encoder_out)

        return nmt_out


#
# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         # self.encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
#         # self.encoder = TransformerEncoder(self.encoder_layer, num_layers=6)
#         self.encoder = nn.LSTM(input_size=embed_size, hidden_size=embed_size, dropout=dropout)
#
#     def forward(self, src):
#         encoder_out = self.encoder(src)
#
#         return encoder_out


class TSTDecoder(nn.Module):
    def __init__(self):
        super(TSTDecoder, self).__init__()
        self.decoder_layer = TransformerDecoderLayer(d_model=512, nhead=8)
        self.tst_decoder = TransformerDecoder(self.decoder_layer, num_layers=6)

    def forward(self, encoder_out, tgt):
        tst_out = self.tst_decoder(tgt, encoder_out)

        return tst_out


class NMTDecoder(nn.Module):
    def __init__(self):
        super(NMTDecoder, self).__init__()
        self.decoder_layer = TransformerDecoderLayer(d_model=512, nhead=8)
        self.nmt_decoder = TransformerDecoder(self.decoder_layer, num_layers=6)

    def forward(self, encoder_out, tgt):
        nmt_out = self.nmt_decoder(tgt, encoder_out)

        return nmt_out
