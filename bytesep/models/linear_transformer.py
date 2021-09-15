import torch
import torch.nn as nn
from torch.nn import Dropout, LayerNorm, Linear, Module
import torch.nn.functional as F


class LinearTransformerBlock(Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(LinearTransformerBlock, self).__init__()
        self.attention = LinearAttention(d_model, n_heads)
        self.linear1 = Linear(d_model, 4 * d_model)
        self.linear2 = Linear(4 * d_model, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = F.relu

    def forward(self, x):
        x = x + self.dropout(
            self.attention(
                x,
                x,
                x,
            )
        )
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x + y)


class LinearAttention(Module):
    def __init__(self, d_model, n_heads, eps=1e-6):
        super(LinearAttention, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        self.query_projection = Linear(d_model, d_keys * n_heads)
        self.key_projection = Linear(d_model, d_keys * n_heads)
        self.value_projection = Linear(d_model, d_values * n_heads)
        self.out_projection = Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

        self.elu_feature_map = lambda x: torch.nn.functional.elu(x) + 1
        self.eps = eps

    def forward(self, queries, keys, values):
        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        values = self.value_projection(values).view(N, S, H, -1)

        # -------- Compute the Linear Attention -------
        Q = self.elu_feature_map(queries)
        K = self.elu_feature_map(keys)
        KV = torch.einsum("nshd,nshm->nhmd", K, values)

        # Compute the normalizer
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)

        # Finally compute and return the new values
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z).contiguous()
        # -----------------------------------------------
        new_values = V.view(N, L, -1)
        # Project the output and return
        return self.out_projection(new_values)
