import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from models.layers.Wave_decomposition import WaveletDeconvolution
from pytorch_lightning import LightningModule
import torch.nn as nn
from models.layers.Embed import DataEmbedding
from models.my_model.sub_models import High_FrequencyBlock, Low_FrequencyBlock, Layernorm
from typing import Dict, List, Tuple, Union
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Model(LightningModule):
    def __init__(self,
                seq_len=144,
                forecast_length=144,
                backcast_length=144,
                kernel_length=25,
                nb_widths=6,
                padding='same',
                d_model=7,
                embed='fixed',
                c_out=7,
                freq='t',
                feature=7,
                modes=64,
                Hwidth=1024,
                hidden_size=64,
                Hnum_blocks=5,
                dropout=0.1,
                Hnum_block_layers=5,
                Hexpansion_coefficient_lengths=512,
                Hstack_types: List[str] = ["nonstationary"],
                Lstack_types: List[str] = ["trend", "seasonality"],
                Lnum_blocks: List[int] = [4, 8],
                Lwidth: List[int] = [32, 512],
                Lnum_block_layers: List[int] = [4, 4],
                Lexpansion_coefficient_lengths: List[int] = [3, 7],
                ):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.kernel_length = kernel_length
        self.nb_widths = nb_widths
        self.embed = embed
        self.freq = freq
        self.Hstack_types = list(Hstack_types)
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.modes = modes
        self.feature = feature
        self.dropout = dropout
        self.c_out = c_out
        self.Hwidth = Hwidth
        self.Hnum_blocks = Hnum_blocks
        self.Hnum_block_layers = Hnum_block_layers
        self.Hexpansion_coefficient_lengths = Hexpansion_coefficient_lengths
        self.Lstack_types = list(Lstack_types)
        self.Lwidth = list(Lwidth)
        self.Lnum_blocks=list(Lnum_blocks)
        self.Lnum_block_layers = list(Lnum_block_layers)
        self.Lexpansion_coefficient_lengths = list(Lexpansion_coefficient_lengths)



        # self.wave_decomp = WaveletDeconvolution(nb_widths=self.nb_widths, seq_len=self.seq_len, kernel_length=self.kernel_length).to(device)
        # self.enc_embedding = DataEmbedding(self.feature, self.d_model, self.embed, self.freq, self.dropout)
        # self.net_blocks = nn.ModuleList()

        # self.High_FrequencyBlock = High_FrequencyBlock(stack_types=self.Hstack_types, d_model=self.d_model, width=self.Hwidth, num_blocks=self.Hnum_blocks, num_block_layers=self.Hnum_block_layers,
        #                                                expansion_coefficient_lengths=self.Hexpansion_coefficient_lengths, seq_len=self.seq_len, backcast_length=self.backcast_length,
        #                                                forecast_length=self.forecast_length)
        # self.Low_FrequencyBlock = nn.ModuleList()
        # for num_blocks, width, num_block_layers, expansion_coefficient_length in zip(self.Lnum_blocks, self.Lwidth,
        #                                                                              self.Lnum_block_layers,
        #                                                                              self.Lexpansion_coefficient_lengths):
        #     L_block = Low_FrequencyBlock(stack_types=self.Lstack_types, num_blocks=num_blocks,
        #                                  num_block_layers=num_block_layers, width=width,
        #                                  expansion_coefficient_lengths=expansion_coefficient_length,
        #                                  seq_len=self.seq_len, backcast_length=self.backcast_length,
        #                                  forecast_length=self.forecast_length)
        #     self.Low_FrequencyBlock.append(L_block)
        self.Low_FrequencyBlock = Low_FrequencyBlock(stack_types=self.Lstack_types,num_blocks=self.Lnum_blocks,num_block_layers=self.Lnum_block_layers, width=self.Lwidth,
                                                       expansion_coefficient_lengths=self.Lexpansion_coefficient_lengths, seq_len=self.seq_len, backcast_length=self.backcast_length,
                                                       forecast_length=self.forecast_length)
        self.Layernorm = Layernorm(self.hidden_size)
        self.projection = nn.Linear(self.hidden_size, self.c_out, bias=True)

    def forward(self, x):
        #
        # x_HighF = self.wave_decomp(x)
        # x_LowF = self.wave_decomp(x)
        # x_HighF = self.enc_embedding(x, x_timestamp)

        # x_LowF = self.enc_embedding(x, x_timestamp)
        # print(x.shape)

        # nonstationary_forecasting, _ = self.High_FrequencyBlock(x)
        seasonal_forecasting, _, _ = self.Low_FrequencyBlock(x)
        # nonstationary_forecasting = self.Layernorm(nonstationary_forecasting)
        # nonstationary_forecasting = self.projection(nonstationary_forecasting)

        seasonal_forecasting = self.Layernorm(seasonal_forecasting)
        seasonal_forecasting = self.projection(seasonal_forecasting)
        # results = (seasonal_forecasting + nonstationary_forecasting)/2
        results = seasonal_forecasting

        return results[:, -self.forecast_length:, :]





