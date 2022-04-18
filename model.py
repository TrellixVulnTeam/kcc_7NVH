# Import PyTorch
import torch
from torch import nn
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch.autograd import Variable


class StyleTransferTransformer(nn.Module):
    def __init__(self, encoder, tst_decoder, style_ratio):
        super(StyleTransferTransformer, self).__init__()
        self.encoder = encoder
        self.tst_decoder = tst_decoder
        self.style_ratio = style_ratio

        self.context2mu = nn.Linear()
        self.context2logv = nn.Linear()

    def reparameterization(self, encoder_out):
        #TODO 이거 뭐쓸지 더 조사하기
        mu = self.context2mean(encoder_out)
        log_v = self.context2logv(encoder_out)

        std = log_v.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        z = eps.mul(std).add_(mu)

        return z, mu, log_v

    def forward(self, src, tgt):
        encoder_out = self.encoder(src)

        style_index = int(len(encoder_out) * (1-self.style_ratio))
        context_c, context_a = encoder_out[:style_index], encoder_out[style_index:]

        content_c, content_mu, content_logv = self.reparameterization(context_c)
        style_a, style_mu, style_logv = self.reparameterization(context_a)

        # TODO binary cross entropy
        # style_loss =

        total_latent = torch.cat(content_c, style_a)

        out = self.tst_decoder(total_latent, tgt)





class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.encoder_layer = TransformerEncoderLayer()
        self.encoder = TransformerEncoder()

    def forward(self):