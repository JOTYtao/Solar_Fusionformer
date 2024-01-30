import math
from typing import Dict, List, Tuple, Union
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.nn.utils import rnn
import torch.nn.functional as F
from models.my_model.embedding import MultiEmbedding
class TimeDistributed(nn.Module):
    def __init__(self, module: nn.Module, batch_first: bool = True):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        b, t, h, w = x.size()
        x = x.view(b * t, h, w)
        y = self.module(x)
        y = y.view(b, t, y.size(1), y.size(2))
        return y

class TimeDistributedInterpolation(nn.Module):
    def __init__(self, output_size: int, batch_first: bool = False, trainable: bool = False):
        super().__init__()
        self.output_size = output_size
        self.batch_first = batch_first
        self.trainable = trainable
        if self.trainable:
            self.mask = nn.Parameter(torch.zeros(output_size, dtype=torch.float32))
            self.gate = nn.Sigmoid()
    def interpolate(self, x):
        upsampled = F.interpolate(x.unsqueeze(1), self.output_size, mode="linear", align_corners=True).squeeze(1)
        if self.trainable:
            upsampled = upsampled * self.gate(self.mask.unsqueeze(0)) * 2.0
        return upsampled
    def forward(self, x):
        # x shape (batch, time, channels, height, width)
        if len(x.size()) <= 2:
            return self.interpolate(x)
        b, t, h, w = x.size()
        x = x.contiguous().view(b * h * w, t)

        y = self.interpolate(x)
        y = y.contiguous().view(b, y.size(-1), h, w)

        return y

class AddNorm(nn.Module):
    def __init__(self, input_channels: int, skip_size: int, trainable_add: bool = True):
        super().__init__()

        self.input_channels = input_channels
        self.trainable_add = trainable_add
        self.skip_size = skip_size
        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(1, self.input_channels, 1, 1, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.BatchNorm2d(self.input_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):

        if self.trainable_add:
            skip = skip * self.gate(self.mask) * 2.0
        output = self.norm(x + skip)
        return output




class GatedConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float = None):
        super().__init__()
        if dropout is not None:
            self.dropout = nn.Dropout2d(dropout)
        else:
            self.dropout = None
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size, padding=kernel_size // 2)
        self.init_weights()
    def init_weights(self):
        for n, p in self.named_parameters():
            if 'bias' in n:
                nn.init.zeros_(p)
            else:
                nn.init.kaiming_uniform_(p, nonlinearity='sigmoid')
    def forward(self, x):
        if self.dropout:
            x = self.dropout(x)
        x = self.conv(x)
        x, gate = x.chunk(2, dim=1)
        return x * torch.sigmoid(gate)

class ResampleNorm(nn.Module):
    def __init__(self, input_channels: int, output_channels: int = None, batch_first: bool = True, trainable_add: bool = True):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.batch_first = batch_first
        self.trainable_add = trainable_add
        if self.input_channels != self.output_channels:
            self.resample = TimeDistributedInterpolation(self.output_channels, batch_first=True, trainable=False)

        if self.trainable_add:
            self.mask = nn.Parameter(torch.ones(1, output_channels, 64, 64, dtype=torch.float32))
            self.gate = nn.Sigmoid()

        self.norm = nn.BatchNorm2d(num_features=output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_channels != self.output_channels:
            x = self.resample(x)

        if self.trainable_add:
            x = x * self.gate(self.mask) * 2.0

        x = self.norm(x)
        return x

class GatedConvUnit(nn.Module):
    def __init__(self, input_channels: int, hidden_size: int, kernel_size: int, dropout: float = None):
        super().__init__()
        if dropout is not None:
            self.dropout = nn.Dropout2d(dropout)
        else:
            self.dropout = dropout
        self.conv = nn.Conv2d(input_channels, hidden_size * 2, kernel_size, stride=1, padding=kernel_size // 2)
        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                torch.nn.init.zeros_(p)
            elif "conv" in n:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        x = F.glu(x, dim=1)
        return x


class GateAddNorm(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_size: int = None,
        skip_size: int = None,
        trainable_add: bool = True,
        dropout: float = None,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_size = hidden_size or input_channels
        self.skip_size = skip_size or self.hidden_size
        self.dropout = dropout

        self.glu = GatedConvUnit(self.input_channels, hidden_size=self.hidden_size, kernel_size=3, dropout=self.dropout)
        self.add_norm = AddNorm(self.hidden_size, skip_size=self.skip_size, trainable_add=trainable_add)

    def forward(self, x, skip):
        output = self.glu(x)
        output = self.add_norm(output, skip)

        return output


class ConvGatedResidualNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        residual: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.dropout = dropout
        self.residual = residual
        if self.input_channels != self.output_channels and not self.residual:
            residual_size = self.input_channels
        else:
            residual_size = self.output_channels
        if self.output_channels != residual_size:
            self.resample_norm = ResampleNorm(residual_size, self.output_channels)
        self.conv1 = nn.Conv2d(self.input_channels, self.hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2)

        self.elu = nn.ELU()
        self.conv2 = nn.Conv2d(self.hidden_channels, self.hidden_channels*2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.init_weights()
        self.gate_norm = GateAddNorm(
            input_channels=self.hidden_channels*2,
            skip_size=self.output_channels,
            hidden_size=self.output_channels,
            dropout=self.dropout,
            trainable_add=False,
        )
    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" in name:
                torch.nn.init.zeros_(p)
            elif "conv1" in name or "conv2" in name:
                torch.nn.init.kaiming_normal_(p, a=0, mode="fan_in", nonlinearity="leaky_relu")
    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        if self.input_channels != self.output_channels and not self.residual:
            print()
            residual = self.resample_norm(residual)


        x = self.conv1(x)
        x = self.elu(x)
        x = self.conv2(x)
        x = self.gate_norm(x, residual)
        return x


class Spectral_Selection_Unit(nn.Module):

    def __init__(
        self,
        input_channels: Dict[str, int],
        hidden_size: int,
        input_embedding_flags: Dict[str, bool] = {},
        dropout: float = 0.1,
        single_variable_grns: Dict[str, ConvGatedResidualNetwork] = {},
        prescalers: Dict[str, nn.Linear] = {},
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.sample_size = [32, 32]
        self.input_channels = input_channels
        self.input_embedding_flags = input_embedding_flags
        self.dropout = dropout


        if self.num_inputs > 1:

            self.flattened_grn = ConvGatedResidualNetwork(
                input_channels=self.input_size_total,
                hidden_channels=self.input_size_total,
                output_channels=self.num_inputs,
                dropout=self.dropout,
                residual=False,
            )

        self.single_variable_grns = nn.ModuleDict()
        self.prescalers = nn.ModuleDict()
        for name, input_channels in self.input_channels.items():
            if name in single_variable_grns:

                self.single_variable_grns[name] = single_variable_grns[name]
            elif self.input_embedding_flags.get(name, False):

                self.single_variable_grns[name] = ResampleNorm(input_channels, self.sample_size)
            else:
                self.single_variable_grns[name] = ConvGatedResidualNetwork(
                    input_channels=input_channels,
                    hidden_channels=self.hidden_size,
                    output_channels=self.hidden_size,
                    dropout=self.dropout,
                )
            if name in prescalers:  # reals need to be first scaled up
                self.prescalers[name] = prescalers[name]
            elif not self.input_embedding_flags.get(name, False):
                self.prescalers[name] = nn.Linear(1, input_channels)

        self.softmax = nn.Softmax(dim=-1)

    @property
    def input_size_total(self):
        return sum(size if name in self.input_embedding_flags else size for name, size in self.input_channels.items())
    @property
    def num_inputs(self):
        return len(self.input_channels)

    def forward(self, x: Dict[str, torch.Tensor]):
        if self.num_inputs > 1:
            # transform single variables
            var_outputs = []
            weight_inputs = []
            for name in self.input_channels.keys():
                variable_embedding = x[name]
                weight_inputs.append(variable_embedding)
                var_outputs.append(self.single_variable_grns[name](variable_embedding))
            var_outputs = torch.stack(var_outputs, dim=-1)

            # calculate variable weights
            flat_embedding = torch.cat(weight_inputs, dim=1)
            sparse_weights = self.flattened_grn(flat_embedding)
            sparse_weights = self.softmax(sparse_weights).unsqueeze(-2)
            sparse_weights = sparse_weights.permute(0, 3, 2, 4, 1)
            outputs = var_outputs * sparse_weights
            outputs = outputs.sum(dim=-1)
        else:  # for one input, do not perform variable selection but just encoding
            name = next(iter(self.single_variable_grns.keys()))
            variable_embedding = x[name]
            if name in self.prescalers:
                variable_embedding = self.prescalers[name](variable_embedding)
            outputs = self.single_variable_grns[name](variable_embedding)  # fast forward if only one variable
            if outputs.ndim == 3:  # -> batch size, time, hidden size, n_variables
                sparse_weights = torch.ones(outputs.size(0), outputs.size(1), 1, 1, device=outputs.device)  #
            else:  # ndim == 2 -> batch size, hidden size, n_variables
                sparse_weights = torch.ones(outputs.size(0), 1, 1, device=outputs.device)
        return outputs, sparse_weights





