import torch
import torch.nn as nn
import torch.nn.functional as F


def g(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))


def log_g(x):
    return torch.where(x >= 0, torch.log(F.relu(x) + 0.5), -F.softplus(-x))


def parallel_scan(log_coeffs, log_values):
    a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    log_h = a_star + log_h0_plus_b_star
    return torch.exp(log_h)[:, 1:]


class MinGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.linear_z = nn.Linear(input_size, hidden_size)
        self.linear_h = nn.Linear(input_size, hidden_size)

    def forward(self, x, h=None, step=False):
        if step:
            return self.forward_step(x, h)
        else:
            return self.forward_parallel(x, h)

    def forward_step(self, x_t, h_prev):
        z_t = torch.sigmoid(self.linear_z(x_t))
        h_tilde = g(self.linear_h(x_t))
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        return h_t, h_t[:, -1:]

    def forward_parallel(self, x, h0):
        seq_len = x.size(1)
        k = self.linear_z(x)
        log_z = -F.softplus(-k)
        log_coeffs = -F.softplus(k)
        log_h0 = log_g(h0)
        log_tilde_h = log_g(self.linear_h(x))
        log_values = torch.cat([log_h0, log_z + log_tilde_h], dim=1)
        h_0_t = parallel_scan(log_coeffs, log_values)
        return h_0_t[:, -seq_len:], h_0_t[:, -1:]
