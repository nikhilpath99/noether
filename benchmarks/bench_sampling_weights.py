#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Benchmark: FieldGradientWeightSampleProcessor vs FieldEigenWeightSampleProcessor.

Run with:
    uv run python benchmarks/bench_sampling_weights.py
"""

import time

import torch

from noether.data.pipeline.sample_processors.field_gradient_weight import (
    FieldGradientWeightSampleProcessor,
)
from noether.data.pipeline.sample_processors.field_eigen_weight import (
    FieldEigenWeightSampleProcessor,
)

CONFIGS = [
    {"N": 5_000,  "label": "small  (5k  pts)"},
    {"N": 10_000, "label": "medium (10k pts)"},
    {"N": 20_000, "label": "large  (20k pts)"},
]
K = 16
REPEATS = 5


def make_sample(N: int, device: torch.device) -> dict:
    return {
        "volume_position": torch.rand(N, 2, device=device),
        "volume_velocity": torch.rand(N, 2, device=device),
        "volume_pressure": torch.rand(N, 1, device=device),
    }


def bench(proc, sample: dict, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        s = {k: v.clone() for k, v in sample.items()}  # fresh copy each run
        t0 = time.perf_counter()
        proc(s)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    print(f"{'Config':<25}  {'Gradient (ms)':>14}  {'Eigen-trace (ms)':>17}  {'Overhead':>9}")
    print("-" * 72)

    for cfg in CONFIGS:
        N = cfg["N"]
        sample = make_sample(N, device)

        grad_proc = FieldGradientWeightSampleProcessor(
            position_item="volume_position",
            field_items=["volume_velocity", "volume_pressure"],
            weight_key="volume_importance_weights",
            k=K,
        )
        eigen_proc = FieldEigenWeightSampleProcessor(
            position_item="volume_position",
            field_items=["volume_velocity", "volume_pressure"],
            weight_key="volume_importance_weights",
            k=K,
            mode="trace",
        )

        # Warmup
        bench(grad_proc, sample, repeats=2)
        bench(eigen_proc, sample, repeats=2)

        t_grad = bench(grad_proc, sample, repeats=REPEATS) * 1000
        t_eigen = bench(eigen_proc, sample, repeats=REPEATS) * 1000
        overhead = f"{t_eigen / t_grad:.2f}x"

        print(f"{cfg['label']:<25}  {t_grad:>14.1f}  {t_eigen:>17.1f}  {overhead:>9}")


if __name__ == "__main__":
    main()
