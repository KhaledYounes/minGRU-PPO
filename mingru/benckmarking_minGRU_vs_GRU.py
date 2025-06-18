import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

from GRURef import GRURef
from min_gru import MinGRU
from min_gru_triton import FusedMinGRU

REFERENCE_GRU = "GRU (reference)"
BUILD_IN_GRU = "PyTorch GRU"
PYTORCH_MIN_GRU = "MinGRU (py)"
FUSED_MIN_GRU = "MinGRU (fused)"


def benchmark_four_variants(
        batch_size: int = 8,
        input_size: int = 4096,
        hidden_size: int = 8,
        seq_lengths: list[int] | None = None,
        n_iters: int = 25,
        device: str | None = None,
):
    if seq_lengths is None:
        seq_lengths = list(range(1000, 10001, 1000))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    nn_reference_gru = GRURef(input_size, hidden_size).to(device)
    nn_gru = nn.GRU(input_size, hidden_size, batch_first=True).to(device)
    min_gru_py = MinGRU(input_size, hidden_size).to(device)
    min_gru_fused = FusedMinGRU(input_size, hidden_size).to(device)

    models: dict[str, nn.Module] = {
        REFERENCE_GRU: nn_reference_gru,
        BUILD_IN_GRU: nn_gru,
        PYTORCH_MIN_GRU: min_gru_py,
        FUSED_MIN_GRU: min_gru_fused,
    }

    records: list[dict] = []
    for name, model in models.items():
        if name in [PYTORCH_MIN_GRU, FUSED_MIN_GRU]:
            h0 = torch.randn(batch_size, 1, hidden_size, device=device)
        else:
            h0 = torch.randn(1, batch_size, hidden_size, device=device)
        for T in seq_lengths:
            x = torch.randn(batch_size, T, input_size, device=device)

            for _ in range(10):
                model.zero_grad(set_to_none=True)
                out, _ = model(x, h0)
                out.sum().backward()

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iters):
                model.zero_grad(set_to_none=True)
                out, _ = model(x, h0)
                out.sum().backward()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            fb_ms = (time.perf_counter() - t0) / n_iters * 1e3

            records.append({"Model": name, "T": T, "Fwd+Back (ms)": fb_ms})
            print({"Model": name, "T": T, "Fwd+Back (ms)": fb_ms})

    return pd.DataFrame.from_records(records)


def _save_plot(df: pd.DataFrame, labels: list[str], title: str, filename: str):
    plt.figure()
    for lbl in labels:
        sub = df[df["Model"] == lbl]
        plt.plot(sub["T"], sub["Fwd+Back (ms)"], marker="o", label=lbl)
    plt.xlabel("Sequence Length T")
    plt.ylabel("Forward+Backward Time (ms)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def generate_artifacts(df: pd.DataFrame, out_path: str = "."):
    csv_file = f"{out_path}/benchmark_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"CSV written → {csv_file}")

    _save_plot(df, [REFERENCE_GRU, PYTORCH_MIN_GRU],
               "Reference GRU vs MinGRU (python scan)",
               f"{out_path}/plot_ref_vs_minpy.png")

    _save_plot(df, [BUILD_IN_GRU, PYTORCH_MIN_GRU],
               "PyTorch GRU vs MinGRU (python scan)",
               f"{out_path}/plot_builtin_vs_minpy.png")

    _save_plot(df, [BUILD_IN_GRU, FUSED_MIN_GRU],
               "PyTorch GRU vs MinGRU (fused)",
               f"{out_path}/plot_builtin_vs_minfused.png")


if __name__ == "__main__":
    df_bench = benchmark_four_variants()
    generate_artifacts(df_bench)
