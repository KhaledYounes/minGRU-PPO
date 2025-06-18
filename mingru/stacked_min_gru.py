import torch
import torch.nn as nn

from .min_gru import MinGRU
from .min_gru_triton import FusedMinGRU


class MinGRUBlock(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            *,
            use_norm: bool = False,
            use_residual: bool = False,
            use_fused_kernel: bool = True,
    ):
        super().__init__()
        self.use_norm = use_norm
        self.use_residual = use_residual

        self.norm = nn.LayerNorm(input_size, eps=1e-10) if use_norm else nn.Identity()

        self.min_gru = FusedMinGRU(input_size, hidden_size) if use_fused_kernel else MinGRU(input_size, hidden_size)

        self.res_proj = (
            nn.Identity()
            if input_size == hidden_size
            else nn.Linear(input_size, hidden_size, bias=False)
        )

    def forward(self, x, h=None, step=False):
        out = self.norm(x)
        out, hidden = self.min_gru(out, h, step)
        if self.use_residual:
            out = out + self.res_proj(x)
        return out, hidden


class StackedMinGRU(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            *,
            use_norm: bool = False,
            use_residual: bool = False,
            use_fused_kernel: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.blocks = nn.ModuleList(
            [
                MinGRUBlock(
                    input_size if i == 0 else hidden_size,
                    hidden_size,
                    use_norm=use_norm,
                    use_residual=use_residual,
                    use_fused_kernel=use_fused_kernel
                )
                for i in range(num_layers)
            ]
        )

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-10)

    def forward(
            self,
            x: torch.Tensor,
            h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if h0 is None:
            raise ValueError("Initial hidden state h0 must be provided")
        if h0.size(0) != self.num_layers:
            raise ValueError("h0 should have shape (num_layers, batch_size, hidden_size)")
        batch_size = x.size(0)
        next_hiddens = torch.empty(self.num_layers, batch_size, self.hidden_size, device=x.device)
        current_out = x
        for i, block in enumerate(self.blocks):
            current_out, current_hidden = block(current_out, h0[i].unsqueeze(1), step=x.size(1) == 1)
            next_hiddens[i] = current_hidden.squeeze(1)
        return self.last_norm(current_out), next_hiddens
