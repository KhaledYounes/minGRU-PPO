from typing import Tuple, Optional

import torch
import torch.nn as nn


class GRUCellRef(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.W_ih = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.W_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

    def forward(self, x, h):
        i_r, i_z, i_n = self.W_ih(x).chunk(3, 1)
        h_r, h_z, h_n = self.W_hh(h).chunk(3, 1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return (1 - z) * n + z * h


class GRURef(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            bias: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        cells = []
        for layer in range(num_layers):
            in_sz = input_size if layer == 0 else hidden_size
            cells.append(GRUCellRef(in_sz, hidden_size, bias=bias))
        self.cells = nn.ModuleList(cells)

    def forward(
            self,
            x: torch.Tensor,
            h0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        if h0 is None:
            h_t = [x.new_zeros(B, self.hidden_size) for _ in range(self.num_layers)]
        else:
            h_t = list(h0.unbind(0))
        outputs = []
        for t in range(T):
            inp = x[:, t, :]
            for layer, cell in enumerate(self.cells):
                h_t[layer] = cell(inp, h_t[layer])
                inp = h_t[layer]
            outputs.append(inp)
        y = torch.stack(outputs, dim=1)
        h_n = torch.stack(h_t)
        return y, h_n
