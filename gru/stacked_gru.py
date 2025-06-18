import torch
import torch.nn as nn


class GRUBlock(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            *,
            use_norm: bool = False,
            use_residual: bool = False
    ):
        super().__init__()
        self.use_residual = use_residual

        self.norm = nn.LayerNorm(input_size, eps=1e-10) if use_norm else nn.Identity()

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.res_proj = (
            nn.Identity()
            if input_size == hidden_size
            else nn.Linear(input_size, hidden_size, bias=False)
        )

    def forward(
            self,
            x: torch.Tensor,
            h0: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.norm(x)
        y, h = self.gru(x, h0)
        if self.use_residual:
            y = y + self.res_proj(x)
        return y, h


class StackedGRU(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            *,
            use_norm: bool = False,
            use_residual: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.blocks = nn.ModuleList(
            [
                GRUBlock(
                    input_size if i == 0 else hidden_size,
                    hidden_size,
                    use_norm=use_norm,
                    use_residual=use_residual
                )
                for i in range(num_layers)
            ]
        )

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-10)

    def forward(
            self,
            x: torch.Tensor,
            h0: torch.Tensor | None = None,
            use_fused_kernel: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if h0 is None:
            raise ValueError("Initial hidden state h0 must be provided")
        if h0.size(0) != self.num_layers:
            raise ValueError("h0 should have shape (num_layers, batch_size, hidden_size)")
        batch_size = x.size(0)
        next_hiddens = torch.empty(self.num_layers, batch_size, self.hidden_size, device=x.device)
        current_out = x
        for i, block in enumerate(self.blocks):
            current_out, current_hidden = block(current_out, h0[i].unsqueeze(0))
            next_hiddens[i] = current_hidden.squeeze(0)
        return self.last_norm(current_out), next_hiddens
