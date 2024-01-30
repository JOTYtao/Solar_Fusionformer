import numpy as np
import torch
from torch import nn
from einops import rearrange, repeat
from math import pi
from einops.layers.torch import Rearrange
class NeRF_embedding(nn.Module):
    def __init__(self, n_layers: int = 5):
        super().__init__()
        self.n_layers = n_layers
        self.dim = self.n_layers * 4

    def forward(self, spatial_coords: torch.Tensor):
        """
        Args:
            spatial_coords (torch.Tensor): Spatial coordinates of shape [B, 2, H, W]
        """
        embeddings = []
        for i in range(self.n_layers):
            embeddings += [
                torch.sin((2**i * torch.pi) * spatial_coords),
                torch.cos((2**i * torch.pi) * spatial_coords),
            ]
        embeddings = torch.cat(embeddings, axis=1)
        return embeddings


class Cyclical_embedding(nn.Module):
    def __init__(self, frequencies: list):
        super().__init__()
        self.frequencies = frequencies
        self.dim = len(self.frequencies) * 2

    def forward(self, time_coords: torch.Tensor):
        """
        Args:
            time_coords (torch.Tensor): Time coordinates of shape [B, T, C, H, W]
        """
        embeddings = []
        for i, frequency in enumerate(self.frequencies):
            embeddings += [
                torch.sin(2 * torch.pi * time_coords[:, :, i] / frequency),
                torch.cos(2 * torch.pi * time_coords[:, :, i] / frequency),
            ]
        embeddings = torch.stack(embeddings, axis=2)
        return embeddings


"""
Inspired by: https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/torch_encodings.py
"""


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, coords):
        """
        :param tensor: A 4d tensor of size (batch_size, ch, x, y)
        :param coords: A 4d tensor of size (batch_size, num_coords, x, y)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(coords.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        batch_size, _, x, y = coords.shape
        self.cached_penc = None
        pos_x = coords[:, 0, 0, :].type(self.inv_freq.type())  # batch, width
        pos_y = coords[:, 1, :, 0].type(self.inv_freq.type())  # batch, height
        sin_inp_x = torch.einsum("bi,j->bij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("bi,j->bij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(2)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb = torch.zeros(
            (batch_size, x, y, self.channels * 2), device=coords.device
        ).type(coords.type())
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y

        return emb
class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, freq_type="lucidrains", **kwargs):
        super().__init__()
        self.dim = dim
        self.freq_type = freq_type
        if freq_type == "lucidrains":
            scales = torch.linspace(1.0, kwargs["max_freq"] / 2, self.dim // 4)
        elif freq_type == "vaswani":
            scales = 1 / (
                kwargs["base"] ** (torch.arange(0, self.dim, 4).float() / self.dim)
            )
        else:
            NotImplementedError(
                f"Only 'lucidrains' and 'vaswani' frequencies are implemented, but you chose {freq_type}."
            )
        self.register_buffer("scales", scales)

    def forward(self, coords: torch.Tensor):
        """
        Assumes that coordinates do not change throughout the batches.
        Args:
            coords (torch.Tensor): Coordinates of shape [B, 2, H, W]
        """
        seq_x = coords[:, 0, 0, :]
        seq_x = seq_x.unsqueeze(-1)
        seq_y = coords[:, 1, :, 0]
        seq_y = seq_y.unsqueeze(-1)

        scales = self.scales[(*((None, None)), Ellipsis)]
        scales = scales.to(coords)

        if self.freq_type == "lucidrains":
            seq_x = seq_x * scales * pi
            seq_y = seq_y * scales * pi
        elif self.freq_type == "vaswani":
            seq_x = seq_x * scales
            seq_y = seq_y * scales

        x_sinu = repeat(seq_x, "b i d -> b i j d", j=seq_y.shape[1])
        y_sinu = repeat(seq_y, "b j d -> b i j d", i=seq_x.shape[1])

        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim=-1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim=-1)

        sin, cos = map(lambda t: rearrange(t, "b i j d -> b (i j) d"), (sin, cos))
        sin, cos = map(lambda t: repeat(t, "b n d -> b n (d j)", j=2), (sin, cos))
        return sin, cos