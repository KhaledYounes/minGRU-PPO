import torch
import torch.nn.functional as F

from triton_fused_heinsen_scan import prefix_scan


def log_g(x):
    return torch.where(x >= 0, torch.log(F.relu(x) + 0.5), -F.softplus(-x))


@torch.jit.script
def reference_forward(log_coeffs, log_values):
    a_star = log_coeffs.cumsum(dim=1)
    temp = log_values - a_star
    z = temp.logcumsumexp(dim=1)
    return (a_star + z).exp()


def compare(batch=4, T=2048, hidden=8, n_runs=20):
    print()
    print("Run with: batch= ", batch, ", T = ", T, ", hidden_states= ", hidden)
    print()

    torch.manual_seed(111)
    x0 = log_g(torch.randn(batch, T, hidden, device='cuda'))
    y0 = log_g(torch.randn(batch, T, hidden, device='cuda'))

    fwd_times_ref = []
    fwd_times_cus = []
    bwd_times_ref = []
    bwd_times_cus = []

    for i in range(n_runs):
        x_ref = x0.clone().requires_grad_(True)
        y_ref = y0.clone().requires_grad_(True)
        x_cus = x0.clone().requires_grad_(True)
        y_cus = y0.clone().requires_grad_(True)

        start_f = torch.cuda.Event(enable_timing=True)
        end_f = torch.cuda.Event(enable_timing=True)
        start_b = torch.cuda.Event(enable_timing=True)
        end_b = torch.cuda.Event(enable_timing=True)

        # ----- Reference forward + backward -----
        # Forward timing
        torch.cuda.synchronize()
        start_f.record()
        out_ref = reference_forward(x_ref, y_ref)
        end_f.record()
        torch.cuda.synchronize()
        fwd_times_ref.append(start_f.elapsed_time(end_f))  # ms

        # Backward timing
        loss_ref = out_ref.sum()
        torch.cuda.synchronize()
        start_b.record()
        loss_ref.backward()
        end_b.record()
        torch.cuda.synchronize()
        bwd_times_ref.append(start_b.elapsed_time(end_b))

        # ----- Custom forward + backward -----
        # Forward timing
        torch.cuda.synchronize()
        start_f.record()
        out_cus = prefix_scan(x_cus, y_cus)
        end_f.record()
        torch.cuda.synchronize()
        fwd_times_cus.append(start_f.elapsed_time(end_f))

        # Backward timing
        loss_cus = out_cus.sum()
        torch.cuda.synchronize()
        start_b.record()
        loss_cus.backward()
        end_b.record()
        torch.cuda.synchronize()
        bwd_times_cus.append(start_b.elapsed_time(end_b))

        if i == 0:
            print("Max abs diff output:", f"{(out_ref - out_cus).abs().max().item():.12e}")
            print("Max abs diff coeffs:", f"{(x_ref.grad - x_cus.grad).abs().max().item():.12e}")
            print("Max abs diff values:", f"{(y_ref.grad - y_cus.grad).abs().max().item():.12e}")

        if i < 10:
            fwd_times_ref = []
            fwd_times_cus = []
            bwd_times_ref = []
            bwd_times_cus = []

    # Compute averages
    def summarize(name, times_ref, times_cus):
        avg_ref = sum(times_ref) / len(times_ref)
        avg_cus = sum(times_cus) / len(times_cus)
        print(
            f"{name:<20} | reference: {avg_ref:7.3f} ms | custom: {avg_cus:7.3f} ms | speedup: {avg_ref / avg_cus:5.2f}×")

    print("\nAverage over {} runs:".format(n_runs - 1))
    summarize("Forward pass", fwd_times_ref, fwd_times_cus)
    summarize("Backward pass", bwd_times_ref, bwd_times_cus)


if __name__ == "__main__":
    print("#######")
    print("#######")
    print()
    print("#######")
    print("#######")
    compare(batch=8, T=64, hidden=128)
    compare(batch=8, T=128, hidden=128)
    compare(batch=8, T=256, hidden=128)
    compare(batch=8, T=512, hidden=128)
    compare(batch=8, T=1024, hidden=128)
    compare(batch=8, T=2048, hidden=128)
    compare(batch=8, T=4096, hidden=128)
