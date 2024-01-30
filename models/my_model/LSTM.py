
from typing import Dict, List, Tuple, Union
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.nn.utils import rnn
HiddenState = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]


class RNN(ABC, nn.RNNBase):
    @abstractmethod
    def handle_no_encoding(
        self, hidden_state: HiddenState, no_encoding: torch.BoolTensor, initial_hidden_state: HiddenState
    ) -> HiddenState:
        pass

    @abstractmethod
    def init_hidden_state(self, x: torch.Tensor) -> HiddenState:
        pass

    @abstractmethod
    def repeat_interleave(self, hidden_state: HiddenState, n_samples: int) -> HiddenState:

        pass

    def forward(
        self,
        x: Union[rnn.PackedSequence, torch.Tensor],
        hx: HiddenState = None,
        lengths: torch.LongTensor = None,
        enforce_sorted: bool = True,
    ) -> Tuple[Union[rnn.PackedSequence, torch.Tensor], HiddenState]:
        if isinstance(x, rnn.PackedSequence) or lengths is None:
            assert lengths is None, "cannot combine x of type PackedSequence with lengths argument"
            return super().forward(x, hx=hx)
        else:
            min_length = lengths.min()
            max_length = lengths.max()
            assert min_length >= 0, "sequence lengths must be great equals 0"
            if max_length == 0:
                hidden_state = self.init_hidden_state(x)
                if self.batch_first:
                    out = torch.zeros(lengths.size(0), x.size(1), self.hidden_size, dtype=x.dtype, device=x.device)
                else:
                    out = torch.zeros(x.size(0), lengths.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
                return out, hidden_state
            else:
                pack_lengths = lengths.where(lengths > 0, torch.ones_like(lengths))
                packed_out, hidden_state = super().forward(
                    rnn.pack_padded_sequence(
                        x, pack_lengths.cpu(), enforce_sorted=enforce_sorted, batch_first=self.batch_first
                    ),
                    hx=hx,
                )
                # replace hidden cell with initial input if encoder_length is zero to determine correct initial state
                if min_length == 0:
                    no_encoding = (lengths == 0)[
                        None, :, None
                    ]  # shape: n_layers * n_directions x batch_size x hidden_size
                    if hx is None:
                        initial_hidden_state = self.init_hidden_state(x)
                    else:
                        initial_hidden_state = hx
                    # propagate initial hidden state when sequence length was 0
                    hidden_state = self.handle_no_encoding(hidden_state, no_encoding, initial_hidden_state)
                out, _ = rnn.pad_packed_sequence(packed_out, batch_first=self.batch_first)
                return out, hidden_state
class LSTM(RNN, nn.LSTM):
    def handle_no_encoding(
        self, hidden_state: HiddenState, no_encoding: torch.BoolTensor, initial_hidden_state: HiddenState
    ) -> HiddenState:
        hidden, cell = hidden_state
        hidden = hidden.masked_scatter(no_encoding, initial_hidden_state[0])
        cell = cell.masked_scatter(no_encoding, initial_hidden_state[0])
        return hidden, cell
    def init_hidden_state(self, x: torch.Tensor) -> HiddenState:
        num_directions = 2 if self.bidirectional else 1
        if self.batch_first:
            batch_size = x.size(0)
        else:
            batch_size = x.size(1)
        hidden = torch.zeros(
            (self.num_layers * num_directions, batch_size, self.hidden_size),
            device=x.device,
            dtype=x.dtype,
        )
        cell = torch.zeros(
            (self.num_layers * num_directions, batch_size, self.hidden_size),
            device=x.device,
            dtype=x.dtype,
        )
        return hidden, cell

    def repeat_interleave(self, hidden_state: HiddenState, n_samples: int) -> HiddenState:
        hidden, cell = hidden_state
        hidden = hidden.repeat_interleave(n_samples, 1)
        cell = cell.repeat_interleave(n_samples, 1)
        return hidden, cell