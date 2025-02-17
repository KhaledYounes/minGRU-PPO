import torch
import torch.nn as nn

from .min_gru import MinGRU


class MinGRUBlock(nn.Module):
    def __init__(self, input_size, hidden_size, use_norm=False, use_residual=False):
        super().__init__()
        self.use_norm = use_norm
        self.use_residual = use_residual

        self.min_gru = MinGRU(input_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size) if use_norm else nn.Identity()
        self.res_proj = nn.Linear(input_size, hidden_size) if (
                use_residual and input_size != hidden_size) else nn.Identity()

    def forward(self, x, h=None, step=False):
        out, hidden = self.min_gru(x, h, step)
        out = self.norm(out)
        out = out + self.res_proj(x) if self.use_residual else 0
        return out, hidden


class StackedMinGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, use_norm=False, use_residual=False):
        super().__init__()
        self.num_layers = num_layers

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            layer_input = input_size if i == 0 else hidden_size
            block = MinGRUBlock(
                input_size=layer_input,
                hidden_size=hidden_size,
                use_norm=use_norm,
                use_residual=use_residual
            )
            self.blocks.append(block)

    def forward(self, x, h0=None):
        if h0 is None:
            raise ValueError("Initial hidden state h0 must be provided")
        if h0.size(0) != self.num_layers:
            raise ValueError("h0 should have shape (num_layers, batch_size, hidden_size)")
        next_hiddens = []
        current_out = x
        for i, (block, h_init) in enumerate(zip(self.blocks, h0)):
            current_out, current_hidden = block(current_out, h_init.unsqueeze(1), step=x.size(1) == 1)
            next_hiddens.append(current_hidden)
        return current_out, torch.stack(next_hiddens, dim=0)
