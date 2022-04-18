# Import PyTorch
import torch
from torch import nn

class StyleTransferTransformer(nn.Module):
    def __init__(self, encoder, tst_decoder):
        super(StyleTransferTransformer, self).__init__()
        self.encoder = encoder
        self.tst_decoder = tst_decoder

    def forward(self, src, trg):


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()

    def forward(self):