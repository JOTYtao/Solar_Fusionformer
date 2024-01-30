from typing import Dict, List, Tuple, Union
import warnings
warnings.filterwarnings("ignore")
from einops.layers.torch import Rearrange
import torch
import pytorch_lightning as pl
from torch import nn
from einops import rearrange, repeat
from models.my_model.positional_encoding import Cyclical_embedding
from models.my_model.ConvLSTM import ConvLSTM
from models.my_model.Attention import TimeseriesMultiHeadAttention
from models.my_model.LSTM import LSTM
from models.my_model.CVselection_network import Spectral_Selection_Unit
from models.my_model.TSselection_network import AddNorm, GateAddNorm, GatedLinearUnit, Linear_Residual_Unit,  Weather_feather_Selection_Unit
from models.my_model.GKConvLSTM import GKConvLSTM
class FusionFormer(pl.LightningModule):
    def __init__(
            self,
            image_size: Union[List[int], Tuple[int]],
            patch_size: Union[List[int], Tuple[int]],
            frequencies: [12, 31, 24, 60],
            bands_number: int = 11,
            dim_head: int = 64,
            dim: int = 512,
            batch_size: int = 16,
            attention_head_size: int = 4,
            satellite_masking_ratio: float = 0.9,
            timeseries_masking_ratio: float =0.9,
            output_size: int = 1,
            freq_type: str = "lucidrains",
            encoder_input_satellite: Dict[str, int] = {"0": 48, "1": 48, "2": 48, "3": 48, "4": 48, "5": 48, "6": 48, "7": 48, "8": 48, "9": 48, "10": 48},
            satellite_flags: Dict[str, bool] = {"0": False, "1": False, "2": False, "3": False, "4": False, "5": False, "6": False, "7": False, "8": False, "9": False, "10": False},
            encoder_input_timeseries: Dict[str, int] = {"0": 48, "1": 48, "2": 48, "3": 48, "4": 48, "5": 48, "6": 48, "7": 48, "8": 48, "9": 48, "10": 48},
            timeseries_flags: Dict[str, bool] = {"0": True, "1": True, "2": True, "3": True, "4": True, "5": True,"6": True, "7": True, "8": True, "9": True, "10": True},
            input_length: int = 48,
            output_length: int = 24,
            lstm_layers: int = 3,
            pe_type: str = "learned",
            dropout: float = 0.1,
            **kwargs,
    ):
        super(FusionFormer, self).__init__()
        self.bands_number = bands_number
        self.frequencies = frequencies
        self.patch_size = patch_size
        self.image_size = image_size
        self.save_hyperparameters()
        self.hidden_size = input_length
        self.output_length = output_length
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.dim = dim
        self.output_size = output_size
        self.attention_head_size = attention_head_size
        self.batch_size = batch_size
        self.satellite_masking_ratio = satellite_masking_ratio
        self.timeseries_masking_ratio = timeseries_masking_ratio
        self.pe_type = pe_type
        self.time_coords_encoder = Cyclical_embedding(frequencies=self.frequencies)
        self.ts_mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.Spectral_Selection_Unit = Spectral_Selection_Unit(input_channels=encoder_input_satellite, input_embedding_flags=satellite_flags, hidden_size=self.hidden_size)

        self.ctx_embedding = nn.Sequential(
            Rearrange(
                "b t c h w -> b t h w c"
            ),
            nn.Linear(9, 16)
        )
        self.convlstm = ConvLSTM(input_channel=16, image_size=64, seq_len=self.output_length)
        self.GKConvLSTM = GKConvLSTM(input_channel=16, image_size=64, seq_len=self.output_length)
        self.linear = nn.Linear(64*16*16, 64)
        self.LRU_1 = Linear_Residual_Unit(input_size=64, hidden_size=self.dim, output_size=self.dim, dropout=self.dropout, residual=True)
        self.LRU_2 = Linear_Residual_Unit(input_size=64, hidden_size=self.dim, output_size=self.dim,
                                          dropout=self.dropout, residual=True)



        self.ts_embedding = nn.Linear(self.hidden_size, self.dim)
        self.Weather_feather_Selection_Unit = Weather_feather_Selection_Unit(input_sizes=encoder_input_timeseries, input_embedding_flags=timeseries_flags, hidden_size=self.hidden_size)
        self.initial_hidden_lstm = Linear_Residual_Unit(input_size=self.hidden_size, hidden_size=self.dim, output_size=self.dim, dropout=self.dropout,)
        self.initial_cell_lstm = Linear_Residual_Unit(input_size=self.hidden_size, hidden_size=self.dim, output_size=self.dim, dropout=self.dropout,)
        self.grn = Linear_Residual_Unit(input_size=self.dim, hidden_size=self.dim, output_size=self.dim, dropout=self.dropout)
        self.lstm_encoder = LSTM(input_size=self.dim, hidden_size=self.dim, num_layers=self.lstm_layers, dropout=0.1, batch_first=True)
        self.post_lstm_gate_encoder = GatedLinearUnit(self.dim, dropout=self.dropout)
        self.post_lstm_add_norm_encoder = AddNorm(self.dim, trainable_add=False)
        self.timeseries_attention = TimeseriesMultiHeadAttention(d_model=self.dim, n_head=self.attention_head_size, dropout=self.dropout)
        self.timeseries_attn_gate_norm = GateAddNorm(self.dim, dropout=self.dropout, trainable_add=False)
        self.output_layer = nn.Linear(self.dim, self.output_size)
        self.Adaptive_output = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(self.output_length),
            nn.Conv1d(self.dim, self.dim, kernel_size=3, padding=1),
            nn.ReLU(),
            Rearrange("b c t -> b t c"),
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.output_size),
        )
    def forward(self,optical_flow: torch.Tensor, ctx: torch.Tensor, ctx_coords: torch.Tensor, ts: torch.Tensor, ts_coords: torch.Tensor, time_coords: torch.Tensor, mask: bool = True):

        """
        Args:
            optical_flow (torch.Tensor): optical_flow  frame of shape [B, T, C×2, H, W]
            ctx (torch.Tensor): Context frames of shape [B, T, C, H, W]
            ctx_coords (torch.Tensor): Coordinates of context frames of shape [B, 2, H, W]
            ts (torch.Tensor): Station timeseries of shape [B, T, C]
            ts_coords (torch.Tensor): Station coordinates of shape [B, 2, 1, 1]
            time_coords (torch.Tensor): Time coordinates of shape [B, T, C, H, W]
            mask (bool): Whether to mask or not. Useful for inference
        Returns:
        """
        B, T, C, H, W = ctx.shape
        time_coords = self.time_coords_encoder(time_coords)
        ctx = torch.cat([ctx, time_coords], axis=2)
        ts = torch.cat([ts, time_coords[..., 0, 0]], axis=-1)

        # ******************************satellite feature part****************************************

        ctx_dict = {str(c): ctx[:, :, c, :, :] for c in range(ctx.shape[2])}
        ctx_output, bands_weights = self.Spectral_Selection_Unit(ctx_dict) #ctx_output.shape : B, T, H, W  satellite_weights : B, 1, H, W, bands_number
        ctx_output = ctx_output.unsqueeze(2) #ctx_output.shape : B, T, 1, H, W
        ctx_output = torch.cat([ctx_output, time_coords], axis=2)
        ctx_output = self.ctx_embedding(ctx_output)
        ctx_output = rearrange(ctx_output, "b t h w c -> b t c h w")
        Convlstm_output = self.convlstm(ctx_output)
        Convlstm_output = rearrange(Convlstm_output, "t b c h w -> b t c h w")
        batch_size, seq_len, channels, height, width = Convlstm_output.size()
        flattened_output = Convlstm_output.view(batch_size, seq_len, -1)
        flattened_output = self.linear(flattened_output)
        q_1_output = self.LRU_1(flattened_output)

        # ******************************optical_flow feature part****************************************

        bands_weights_expanded = bands_weights.repeat(1, T, 1, 1, 1)
        bands_weights_expanded = bands_weights_expanded.unsqueeze(4).repeat(1, 1, 1, 1, 2)
        bands_weights_expanded = bands_weights_expanded.permute(0, 1, 4, 2, 3).reshape(B, T, C * 2, H, W)
        optical_flow_feature = optical_flow * bands_weights_expanded
        optical_flow_input = torch.cat([optical_flow_feature, time_coords], axis=2)
        optical_flow_input = self.ctx_embedding(optical_flow_input)
        optical_flow_input = rearrange(optical_flow_input, "b t h w c -> b t c h w")
        optical_flow_output = self.GKConvLSTM(optical_flow_input)
        optical_flow_output = rearrange(optical_flow_output, "t b c h w -> b t c h w")
        opflattened_output = optical_flow_output.view(batch_size, seq_len, -1)
        opflattened_output = self.linear(opflattened_output)
        q_opflattened_output = self.LRU_2(opflattened_output)
        q_attn_output, q_attn_output_weights = self.timeseries_attention(
            q=q_opflattened_output,
            k=q_1_output,
            v=q_1_output,
            mask=None,
        )

        # ******************************time series part****************************************
        '''
        ts (torch.Tensor): Station timeseries of shape [B, T, C]
        '''
        B, T, _ = ts.shape
        encoder_lengths = torch.full((B,), T).to('cuda:0')
        ts = {str(c): ts[:, :, c].unsqueeze(2) for c in range(ts.shape[2])}
        static_embedding = torch.zeros((B, self.hidden_size), device='cuda:0')
        embeddings_varying_encoder, feature_weights = self.Weather_feather_Selection_Unit(ts)
        input_hidden = self.initial_hidden_lstm(static_embedding).expand(self.lstm_layers, -1, -1)
        input_cell = self.initial_cell_lstm(static_embedding).expand(self.lstm_layers, -1, -1)
        embeddings_varying_encoder = self.ts_embedding(embeddings_varying_encoder)
        encoder_output, (hidden, cell) = self.lstm_encoder(embeddings_varying_encoder, (input_hidden, input_cell), lengths=encoder_lengths, enforce_sorted=False)
        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(lstm_output_encoder, embeddings_varying_encoder)
        attn_output, attn_output_weights = self.timeseries_attention(
            q=q_attn_output,
            k=lstm_output_encoder,
            v=lstm_output_encoder,
            mask=None,
        )
        timeseries_attn_output = self.timeseries_attn_gate_norm(attn_output, lstm_output_encoder)
        output = rearrange(timeseries_attn_output, "b t c -> (b t) () c")
        output = self.grn(output)
        output = rearrange(output, "(b t) () c -> b t c", b=B, t=T)

        output = rearrange(output.detach(), "b t c -> b c t")
        output = self.Adaptive_output(output)
        return (output, bands_weights, feature_weights, attn_output_weights)
