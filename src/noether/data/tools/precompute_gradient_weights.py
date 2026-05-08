#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

"""
Pre-compute gradient-based importance weights for a dataset and save them
alongside the existing .pt files.

Usage:
    python -m noether.data.tools.precompute_gradient_weights \\
        --dataset-root ./data/drivaerml \\
        --position-file volume_cell_position.pt \\
        --field-files volume_cell_velocity.pt volume_cell_totalpcoeff.pt \\
        --output-file volume_importance_weights.pt \\
        --k 16

Each run_* directory gets its own weights file written next to the existing tensors.
The weights are float32 tensors of shape (N,).
"""

import argparse
import time
from pathlib import Path

import torch

from noether.data.pipeline.sample_processors import FieldGradientWeightSampleProcessor


def precompute_for_run(
    run_dir: Path,
    position_file: str,
    field_files: list[str],
    output_file: str,
    k: int,
    overwrite: bool,
) -> None:
    out_path = run_dir / output_file
    if out_path.exists() and not overwrite:
        print(f"  [skip] {out_path} already exists")
        return

    pos_path = run_dir / position_file
    if not pos_path.exists():
        print(f"  [skip] {run_dir.name}: {position_file} not found")
        return

    position = torch.load(pos_path, weights_only=True)

    fields = []
    for fname in field_files:
        fpath = run_dir / fname
        if not fpath.exists():
            print(f"  [skip] {run_dir.name}: {fname} not found")
            return
        fields.append(torch.load(fpath, weights_only=True))

    # Build a minimal sample dict and run the processor
    sample: dict = {"_pos": position}
    for i, f in enumerate(fields):
        sample[f"_field_{i}"] = f if f.ndim == 2 else f.unsqueeze(1)

    proc = FieldGradientWeightSampleProcessor(
        position_item="_pos",
        field_items=[f"_field_{i}" for i in range(len(fields))],
        weight_key="_weights",
        k=k,
    )

    t0 = time.perf_counter()
    out = proc(sample)
    elapsed = time.perf_counter() - t0

    weights: torch.Tensor = out["_weights"]
    torch.save(weights, out_path)
    print(f"  [done] {run_dir.name}: shape={tuple(weights.shape)}  time={elapsed:.2f}s  -> {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute gradient importance weights.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--position-file", default="volume_cell_position.pt")
    parser.add_argument("--field-files", nargs="+", default=["volume_cell_velocity.pt", "volume_cell_totalpcoeff.pt"])
    parser.add_argument("--output-file", default="volume_importance_weights.pt")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    run_dirs = sorted(root.glob("run_*"))
    if not run_dirs:
        print(f"No run_* directories found in {root}")
        return

    print(f"Processing {len(run_dirs)} runs in {root}")
    for run_dir in run_dirs:
        precompute_for_run(
            run_dir=run_dir,
            position_file=args.position_file,
            field_files=args.field_files,
            output_file=args.output_file,
            k=args.k,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
