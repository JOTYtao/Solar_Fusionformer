
import math
from typing import Dict, List, Tuple, Union
import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, repeat
class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=160):
        super().__init__()
        assert d_model % 2 == 0, "model dimension has to be multiple of 2 (encode sin(pos) and cos(pos))"
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(0)
            pe = self.pe[:, :seq_len].view(seq_len, 1, self.d_model)
            x = x + pe
            return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = None, scale: bool = True):
        super(ScaledDotProductAttention, self).__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = dropout
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.permute(0, 2, 1))  # query-key overlap

        if self.scale:
            dimension = torch.as_tensor(k.size(-1), dtype=attn.dtype, device=attn.device).sqrt()
            attn = attn / dimension

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
        attn = self.softmax(attn)

        if self.dropout is not None:
            attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn

class TimeseriesMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, dropout: float = 0.0):
        super(TimeseriesMultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)
        self.v_layer = nn.Linear(self.d_model, self.d_v)
        self.q_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_q) for _ in range(self.n_head)])
        self.k_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_k) for _ in range(self.n_head)])
        self.attention = ScaledDotProductAttention()
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)
        self.init_weights()
    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)
    def forward(self, q, k, v, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        attns = []
        vs = self.v_layer(v)
        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            head, attn = self.attention(qs, ks, vs, mask)
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)
        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attn = torch.stack(attns, dim=2)
        outputs = torch.mean(head, dim=2) if self.n_head > 1 else head
        outputs = self.w_h(outputs)
        outputs = self.dropout(outputs)

        return outputs, attn
def rotate_every_two(x):
    x = rearrange(x, "... (d j) -> ... d j", j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d j -> ... (d j)")
# class SatelliteMultiHeadAttention(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, use_rotary=True, ):
#         super().__init__()
#         inner_dim = dim_head * heads
#         self.use_rotary = use_rotary
#         self.heads = heads
#         self.scale = dim_head**-0.5
#         self.attend = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(dropout)
#         self.to_q = nn.Linear(dim, inner_dim, bias=False)
#         self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
#         self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
#
#     def forward(self, x, pos_embedding):
#         q = self.to_q(x)
#         qkv = (q, *self.to_kv(x).chunk(2, dim=-1))
#         q, k, v = map(
#             lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=self.heads), qkv
#         )
#         if self.use_rotary:
#
#             sin, cos = map(
#                 lambda t: repeat(t, "b n d -> (b h) n d", h=self.heads), pos_embedding
#             )
#             dim_rotary = sin.shape[-1]
#             (q, q_pass), (k, k_pass) = map(
#                 lambda t: (t[..., :dim_rotary], t[..., dim_rotary:]), (q, k)
#             )
#             q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
#             q, k = map(lambda t: torch.cat(t, dim=-1), ((q, q_pass), (k, k_pass)))
#         dots = einsum("b i d, b j d -> b i j", q, k) * self.scale
#         attn = self.attend(dots)
#         attn = self.dropout(attn)
#
#         out = einsum("b i j, b j d -> b i d", attn, v)
#         out = rearrange(out, "(b h) n d -> b n (h d)", h=self.heads)
#         print("out", out.shape)
#         satellite_output = self.to_out(out)
#         print("satellite_output", satellite_output.shape)
#         return satellite_output, attn


class SatelliteMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int = 4, d_model: int = 16, dropout: float = 0.0,  use_rotary: bool = True):
        super(SatelliteMultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.use_rotary = use_rotary
        self.d_k = self.d_q = self.d_v = self.d_satellite = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)
        self.satellite_norm = nn.LayerNorm(self.d_model)
        self.v_layer = nn.Linear(self.d_model, self.d_v)
        self.satellite_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_satellite) for _ in range(self.n_head)])
        self.q_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_q) for _ in range(self.n_head)])
        self.k_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_k) for _ in range(self.n_head)])
        self.attention = ScaledDotProductAttention()
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)
        self.init_weights()
    def init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)
    def forward(self, satellite, satellite_pos_embedding, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        satellite = self.satellite_norm(satellite)
        heads = []
        attns = []
        v = self.v_layer(satellite)
        for i in range(self.n_head):
            q = self.q_layers[i](satellite)
            k = self.k_layers[i](satellite)

            sin_satellite = self.satellite_layers[i](satellite_pos_embedding[0])
            cos_satellite = self.satellite_layers[i](satellite_pos_embedding[1])
            if self.use_rotary:
                dim_rotary = sin_satellite.shape[-1]
                (q, q_pass), (k, k_pass) = map(lambda t: (t[..., :dim_rotary], t[..., dim_rotary:]), (q, k))
                q, k = map(lambda t: (t * cos_satellite) + (rotate_every_two(t) * sin_satellite), (q, k))
                q, k = map(lambda t: torch.cat(t, dim=-1), ((q, q_pass), (k, k_pass)))
            head, attn = self.attention(q, k, v, mask)
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)
        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attn = torch.stack(attns, dim=2)
        outputs = torch.mean(head, dim=2) if self.n_head > 1 else head
        outputs = self.w_h(outputs)
        outputs = self.dropout(outputs)
        return outputs, attn


class FeatureFusionMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int = 4, d_model: int = 16, dim_head: int = 64, dropout: float = 0.0,  use_rotary: bool = True):
        super(FeatureFusionMultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.use_rotary = use_rotary
        self.d_k = self.d_q = self.d_v = self.d_satellite = self.d_timeseries = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)
        self.satellite_norm = nn.LayerNorm(self.d_model)
        self.timeseries_norm = nn.LayerNorm(self.d_model)
        self.v_layer = nn.Linear(self.d_model, self.d_v)
        self.satellite_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_satellite) for _ in range(self.n_head)])
        self.timeseries_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_timeseries) for _ in range(self.n_head)])
        self.q_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_q) for _ in range(self.n_head)])
        self.k_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_k) for _ in range(self.n_head)])
        self.attention = ScaledDotProductAttention()
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)
        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)
    def forward(self, satellite, satellite_pos_embedding, timeseries, timeseries_pos_embedding, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        satellite = self.satellite_norm(satellite)
        timeseries = self.timeseries_norm(timeseries)
        heads = []
        attns = []
        v = self.v_layer(satellite)
        for i in range(self.n_head):
            q = self.q_layers[i](timeseries)
            k = self.k_layers[i](satellite)
            sin_satellite = self.satellite_layers[i](satellite_pos_embedding[0])
            cos_satellite = self.satellite_layers[i](satellite_pos_embedding[1])
            sin_timeseries = self.timeseries_layers[i](timeseries_pos_embedding[0])
            cos_timeseries = self.timeseries_layers[i](timeseries_pos_embedding[1])
            if self.use_rotary:
                # sin_satellite, cos_satellite = map(lambda t: repeat(t, "b n d -> (b h) n d", h=1),
                #                                    satellite_pos_embedding)
                # sin_timeseries, cos_timeseries = map(lambda t: repeat(t, "b n d -> (b h) n d", h=1),
                #                                      timeseries_pos_embedding)
                dim_rotary = sin_satellite.shape[-1]
                (q, q_pass), (k, k_pass) = map(lambda t: (t[..., :dim_rotary], t[..., dim_rotary:]), (q, k))
                q = (q * cos_timeseries) + (rotate_every_two(q) * sin_timeseries)
                k = (k * cos_satellite) + (rotate_every_two(k) * sin_satellite)
                q, k = map(lambda t: torch.cat(t, dim=-1), ((q, q_pass), (k, k_pass)))
            head, attn = self.attention(q, k, v, mask)
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)
        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attn = torch.stack(attns, dim=2)
        outputs = torch.mean(head, dim=2) if self.n_head > 1 else head
        outputs = self.w_h(outputs)
        outputs = self.dropout(outputs)
        return outputs, attn


