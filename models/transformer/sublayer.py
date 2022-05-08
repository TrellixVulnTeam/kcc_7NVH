''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.transformer.module import ScaledDotProductAttention



class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        # size (q, k, v) : [max_len, batch, d_model]

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(1), q.size(0), k.size(0), v.size(0)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: lq x b x n x dv
        q = self.w_qs(q).view(len_q, sz_b, n_head, d_k)
        k = self.w_ks(k).view(len_k, sz_b, n_head, d_k)
        v = self.w_vs(v).view(len_v, sz_b, n_head, d_v)
        # size (q, k, v) : [max_len, batch, n_head, d_k]

        # Transpose for attention dot product: lq x n x b x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # size (q, k, v) : [max_len, n_head, batch, d_k]
        # print("transpose:", q.size())

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)
        # size q, attn : [max_len, n_head, batch, d_k]
        # print("attn q", q.size())

        # Transpose to move the head dimension back: lq x b x n x dv
        # Combine the last two dimensions to concatenate all the heads together: lq x b x (n*dv)
        q = q.transpose(2, 1).contiguous().view(len_q, sz_b, -1)
        # print("111 transpose", q.size())
        q = self.dropout(self.fc(q))
        # print("fc q", q.size())
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x