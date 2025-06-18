import torch
import torch.nn.functional as F


def log_g(x):
    return torch.where(x >= 0, torch.log(F.relu(x) + 0.5), -F.softplus(-x))


class CustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_coeffs, log_values):
        a_star = log_coeffs.cumsum(dim=1)
        temp = log_values - a_star
        z = temp.logcumsumexp(dim=1)
        log_h = a_star + z
        h = log_h.exp()

        ctx.save_for_backward(temp, z, h)
        return h

    @staticmethod
    def backward(ctx, grad_output):
        temp, z, h = ctx.saved_tensors
        grad_log_h = grad_output * h

        # Split gradient
        grad_a1 = grad_log_h
        grad_z = grad_log_h

        # Compute z using logcumsumexp for stability
        ell = torch.log(grad_z.abs()) - z
        log_pos = torch.where(grad_z > 0, ell, -torch.inf)
        log_neg = torch.where(grad_z < 0, ell, -torch.inf)
        rev_cumsum_pos = torch.logcumsumexp(log_pos.flip(dims=[1]), dim=1).flip(dims=[1])
        rev_cumsum_neg = torch.logcumsumexp(log_neg.flip(dims=[1]), dim=1).flip(dims=[1])
        grad_temp = torch.exp(temp + rev_cumsum_pos) - torch.exp(temp + rev_cumsum_neg)

        grad_y = grad_temp
        grad_a2 = -grad_temp

        grad_a_star = grad_a1 + grad_a2

        grad_log_coeffs = torch.cumsum(grad_a_star.flip(dims=[1]), dim=1).flip(dims=[1])

        return grad_log_coeffs, grad_y


def fn(x, y):
    return CustomFunction.apply(x, y)


def reference_forward(log_coeffs, log_values):
    a_star = log_coeffs.cumsum(dim=1)
    temp = log_values - a_star
    z = temp.logcumsumexp(dim=1)
    return (a_star + z).exp()


def compare(batch=64, T=1024, hidden=512, tol=1e-3):
    torch.manual_seed(111)
    # Prepare sample inputs
    x0 = log_g(torch.randn(batch, T, hidden, device='cpu'))
    y0 = log_g(torch.randn(batch, T, hidden, device='cpu'))

    x_ref = x0.clone().requires_grad_(True)
    y_ref = y0.clone().requires_grad_(True)
    x_cus = x0.clone().requires_grad_(True)
    y_cus = y0.clone().requires_grad_(True)

    out_ref = reference_forward(x_ref, y_ref)
    loss_ref = out_ref.sum()
    loss_ref.backward()

    out_cus = fn(x_cus, y_cus)
    loss_cus = out_cus.sum()
    loss_cus.backward()

    print("Max abs diff output:", f"{(out_ref - out_cus).abs().max().item():.12e}")
    print("Max abs diff coeffs:", f"{(x_ref.grad - x_cus.grad).abs().max().item():.12e}")
    print("Max abs diff values:", f"{(y_ref.grad - y_cus.grad).abs().max().item():.12e}")

    out_close = torch.allclose(out_ref, out_cus, atol=tol, rtol=tol)
    print(f"[Check] Outputs match: {out_close}")
    if not out_close:
        print("  Max abs diff output:", (out_ref - out_cus).abs().max().item())
    # Grads
    grad_coeffs_close = torch.allclose(x_ref.grad, x_cus.grad, atol=tol, rtol=tol)
    grad_values_close = torch.allclose(y_ref.grad, y_cus.grad, atol=tol, rtol=tol)
    print(f"[Check] Grad w.r.t. coeffs match: {grad_coeffs_close}")
    if not grad_coeffs_close:
        print("  Max abs diff coeffs:", (x_ref.grad - x_cus.grad).abs().max().item())
    print(f"[Check] Grad w.r.t. values match: {grad_values_close}")
    if not grad_values_close:
        print("  Max abs diff values:", (y_ref.grad - y_cus.grad).abs().max().item())


if __name__ == "__main__":
    compare()
