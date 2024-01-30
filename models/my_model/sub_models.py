import math
from typing import Dict, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.FA_Block import HilbertBlock
import numpy as np
from models.layers.NeuralBasis_Decomp import SeasonalDecomp, TrendDecomp, Non_stationaryDecomp
from pytorch_lightning import LightningModule


class Layernorm(nn.Module):


    def __init__(self, channels):
        super(Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class TimeDistributed(nn.Module):
    def __init__(self, module: nn.Module, batch_first: bool = False):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y

class TimeDistributedInterpolation(nn.Module):
    def __init__(self, output_size: int, batch_first: bool = False, trainable: bool = False):
        super().__init__()
        self.output_size = output_size
        self.batch_first = batch_first
        self.trainable = trainable
        if self.trainable:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float32))
            self.gate = nn.Sigmoid()

    def interpolate(self, x):
        upsampled = F.interpolate(x.unsqueeze(1), self.output_size, mode="linear", align_corners=True).squeeze(1)
        if self.trainable:
            upsampled = upsampled * self.gate(self.mask.unsqueeze(0)) * 2.0
        return upsampled

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.interpolate(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.interpolate(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class GatedLinearUnit(nn.Module):
    """Gated Linear Unit"""

    def __init__(self, input_size: int, hidden_size: int = None, dropout: float = None):
        super().__init__()

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout
        self.hidden_size = hidden_size or input_size
        self.fc = nn.Linear(input_size, self.hidden_size * 2)

        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                torch.nn.init.zeros_(p)
            elif "fc" in n:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        x = F.glu(x, dim=-1)
        return x


ACTIVATIONS = ['ReLU',
               'Softplus',
               'Tanh',
               'SELU',
               'LeakyReLU',
               'PReLU',
               'Sigmoid']

class Decomp_Block(LightningModule):
    def __init__(self,
                 input_size: int,
                 n_theta: int,
                 mlp_units: list,
                 basis: nn.Module,
                 dropout_prob: float,
                 activation: str):
        """
        """
        super().__init__()

        self.dropout_prob = dropout_prob

        assert activation in ACTIVATIONS, f'{activation} is not in {ACTIVATIONS}'
        activ = getattr(nn, activation)()

        hidden_layers = [nn.Linear(in_features=input_size,
                                   out_features=mlp_units[0][0])]
        for layer in mlp_units:
            hidden_layers.append(nn.Linear(in_features=layer[0],
                                           out_features=layer[1]))
            hidden_layers.append(activ)

            if self.dropout_prob > 0:
                raise NotImplementedError('dropout')

        output_layer = [nn.Linear(in_features=mlp_units[-1][1], out_features=n_theta)]
        layers = hidden_layers + output_layer
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(self, insample_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute local projection weights and projection
        theta = self.layers(insample_y)
        backcast, forecast = self.basis(theta)
        return backcast, forecast
class High_FrequencyBlock(LightningModule):
    def __init__(self,
                stack_types: List[str] = ["nonstationary"],
                d_model=512,
                modes=64,
                mode_select='random',
                width=1024,
                num_blocks=3,
                num_block_layers=4,
                expansion_coefficient_lengths=128,
                backcast_length=48,
                forecast_length=48,
                seq_len=96,
                dropout=0.1,
                ):
        self.save_hyperparameters()
        super(High_FrequencyBlock, self).__init__()
        self.d_model = d_model
        self.modes = modes
        self.mode_select = mode_select
        self.seq_len = seq_len
        self.num_blocks = num_blocks
        self.expansion_coefficient_length = expansion_coefficient_lengths
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.dropout = dropout
        self.num_block_layers = num_block_layers
        self.width = width
        self.FA_Block = HilbertBlock(in_channels=self.d_model, out_channels=self.d_model, seq_len=self.seq_len, modes=self.modes, mode_select_method=self.mode_select)
        self.net_blocks = nn.ModuleList()
        self.conv = nn.Conv1d(in_channels=512, out_channels=16, kernel_size=1)

        for stack_type in stack_types:
            for _ in range(num_blocks):
                if stack_type == "nonstationary":
                    net_block = Non_stationaryDecomp(
                        units=self.width,
                        thetas_dim=self.expansion_coefficient_length,
                        num_block_layers=self.num_block_layers,
                        backcast_length=self.backcast_length,
                        forecast_length=self.forecast_length,
                        dropout=self.dropout,
                    )
                else:
                    raise ValueError(f"Unknown stack type {stack_type}")
                self.net_blocks.append(net_block)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:

        target, _ = self.FA_Block(x)
        target = target.transpose(1, 2)
        target = self.conv(target)
        target = target.transpose(1, 2)
        num_channels = target.shape[-1]

        timesteps = self.backcast_length + self.forecast_length
        nonstationary_forecast = [torch.zeros((target.size(0), timesteps), dtype=torch.float32, device=self.device) for _ in range(num_channels)]
        forecast = torch.zeros((target.size(0), self.forecast_length, num_channels), dtype=torch.float32, device=self.device)
        for ch in range(num_channels):
            backcast = target[:, :, ch]
            for i, block in enumerate(self.net_blocks):
            # evaluate block
                backcast_block, forecast_block = block(backcast)
                full = torch.cat([backcast_block.detach(), forecast_block.detach()], dim=1)
                if isinstance(block, Non_stationaryDecomp):
                    nonstationary_forecast[ch] = full
                backcast = (backcast - backcast_block)
                forecast[:, :, ch] = forecast[:, :, ch] + forecast_block
        nonstationary_forecast = torch.stack(nonstationary_forecast, dim=-1)

        return forecast, nonstationary_forecast





class Low_FrequencyBlock(LightningModule):
    def __init__(self,
                stack_types: List[str] = ["trend", "seasonality"],
                d_model=512,
                modes=64,
                mode_select='random',
                width=[32, 512],
                num_blocks=[4, 8],
                num_block_layers=[4, 4],
                expansion_coefficient_lengths:  List[int] = [3, 7],
                backcast_length=48,
                forecast_length=48,
                seq_len=96,
                dropout=0.1,
                min_period = [3, 7],

                ):
        self.save_hyperparameters()
        super(Low_FrequencyBlock, self).__init__()
        self.d_model = d_model
        self.modes = modes
        self.mode_select = mode_select
        self.width = width
        self.seq_len = seq_len
        self.num_blocks = num_blocks
        self.expansion_coefficient_length = expansion_coefficient_lengths
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.dropout = dropout
        self.num_block_layers = num_block_layers
        self.width = width
        self.min_period = min_period
        self.net_blocks = nn.ModuleList()
        self.conv = nn.Conv1d(in_channels=7, out_channels=64, kernel_size=1)

        for stack_id, stack_type in enumerate(stack_types):
            for _ in range(num_blocks[stack_id]):
                if stack_type == "seasonality":
                    net_block = SeasonalDecomp(
                        units=self.width[stack_id],
                        thetas_dim=self.expansion_coefficient_length[stack_id],
                        num_block_layers=self.num_block_layers[stack_id],
                        min_period=self.min_period[stack_id],
                        backcast_length=self.backcast_length,
                        forecast_length=self.forecast_length,
                        dropout=self.dropout,
                    )
                elif stack_type == "trend":
                    net_block = TrendDecomp(
                        units=self.width[stack_id],
                        thetas_dim=self.expansion_coefficient_length[stack_id],
                        num_block_layers=self.num_block_layers[stack_id],
                        backcast_length=self.backcast_length,
                        forecast_length=self.forecast_length,
                        dropout=self.dropout,
                    )
                else:
                    raise ValueError(f"Unknown stack type {stack_type}")
                self.net_blocks.append(net_block)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        target = x.transpose(1, 2)
        target = self.conv(target)
        x = target.transpose(1, 2)


        num_channels = x.shape[-1]
        timesteps = self.backcast_length + self.forecast_length
        seasonal_forecast = [torch.zeros((x.size(0), timesteps), dtype=torch.float32, device=self.device) for _ in range(num_channels)]
        trend_forecast = [torch.zeros((x.size(0), timesteps), dtype=torch.float32, device=self.device) for _ in range(num_channels)]
        forecast = torch.zeros((x.size(0), self.forecast_length, num_channels), dtype=torch.float32, device=self.device)
        for ch in range(num_channels):
            backcast = x[:, :, ch]
            for i, block in enumerate(self.net_blocks):
            # evaluate block
                backcast_block, forecast_block = block(backcast)
                full = torch.cat([backcast_block.detach(), forecast_block.detach()], dim=1)
                if isinstance(block, SeasonalDecomp):
                    seasonal_forecast[ch] = full
                elif isinstance(block, TrendDecomp):
                    trend_forecast[ch] = full
                backcast = (backcast - backcast_block)
                forecast[:, :, ch] = forecast[:, :, ch] + forecast_block
        seasonal_forecast = torch.stack(seasonal_forecast, dim=-1)
        trend_forecast = torch.stack(trend_forecast, dim=-1)

        return forecast, seasonal_forecast, trend_forecast











