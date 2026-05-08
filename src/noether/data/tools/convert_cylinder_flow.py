#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Convert DeepMind MeshGraphNets cylinder_flow tfrecord dataset to .pt tensors.

Reads the raw tfrecord files produced by the DeepMind MeshGraphNets data
generation pipeline and saves each trajectory as a directory of .pt tensors
that can be loaded by :class:`~noether.data.datasets.cfd.cylinder_flow.CylinderFlowDataset`.

Directory layout produced::

    <output-root>/
        manifest.json              # {"train": [...], "test": [...], "valid": [...]}
        train_000000/
            mesh_pos.pt            # (N, 2)   float32  mesh node x/y positions
            velocity.pt            # (T, N, 2) float32  velocity field
            pressure.pt            # (T, N, 1) float32  pressure field
            node_type.pt           # (N,)      int32    node type flags
            cells.pt               # (E, 3)    int32    triangular connectivity
        train_000001/
            ...

Node type values (from MeshGraphNets paper):
    0 = interior fluid node
    4 = no-slip wall
    5 = inflow boundary
    6 = outflow boundary

Usage::

    uv run python -m noether.data.tools.convert_cylinder_flow \\
        --dataset-root /data/cylinder_flow \\
        --output-root  /data/cylinder_flow_pt \\
        --splits train test valid

Requires ``tensorflow`` (CPU-only build is sufficient)::

    uv add tensorflow-cpu
"""

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm


def _parse_meta(dataset_root: Path) -> dict:
    meta_path = dataset_root / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"meta.json not found in {dataset_root}. "
            "Make sure you point to the raw MeshGraphNets cylinder_flow directory."
        )
    with open(meta_path) as f:
        return json.load(f)


def _iter_trajectories(tfrecord_path: Path, meta: dict):
    """Yield one dict of numpy arrays per trajectory.

    Args:
        tfrecord_path: Path to a ``<split>.tfrecord`` file.
        meta: Parsed ``meta.json`` dict.

    Yields:
        Dict mapping field name -> numpy array with the leading static batch
        dimension already squeezed out.
    """
    try:
        import tensorflow as tf  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "TensorFlow is required for tfrecord conversion.\n"
            "Install with:  uv add tensorflow-cpu"
        ) from e

    feature_lists = {k: tf.io.VarLenFeature(tf.string) for k in meta["features"]}

    def _parse(proto):
        features = tf.io.parse_single_example(proto, feature_lists)
        out = {}
        for key, field in meta["features"].items():
            data = tf.io.decode_raw(features[key].values, getattr(tf, field["dtype"]))
            data = tf.reshape(data, field["shape"])
            out[key] = data
        return out

    ds = tf.data.TFRecordDataset(str(tfrecord_path))
    for record in ds.map(_parse):
        yield {k: v.numpy() for k, v in record.items()}


def _trajectory_to_tensors(traj: dict) -> dict[str, torch.Tensor]:
    """Convert a raw trajectory dict to a clean tensor dict.

    Squeezes the leading static-field batch dimension (always 1) and
    normalises field names to the Noether convention.

    Args:
        traj: Raw dict from :func:`_iter_trajectories`.

    Returns:
        Dict with keys: mesh_pos, velocity, pressure, node_type, cells.
    """
    import numpy as np

    def _to_tensor(arr, dtype=None):
        t = torch.from_numpy(np.array(arr))
        return t.to(dtype) if dtype is not None else t

    # Static fields have shape (1, N, ...) — squeeze the leading dim
    mesh_pos = _to_tensor(traj["mesh_pos"].squeeze(0))      # (N, 2)
    node_type = _to_tensor(traj["node_type"].squeeze(0))    # (N, 1) or (N,)
    cells = _to_tensor(traj["cells"].squeeze(0))            # (E, 3)

    if node_type.dim() == 2 and node_type.shape[1] == 1:
        node_type = node_type.squeeze(1)                    # (N,)

    # Dynamic fields have shape (T, N, ...) — keep as-is
    velocity = _to_tensor(traj["velocity"])                 # (T, N, 2)
    pressure = _to_tensor(traj["pressure"])                 # (T, N, 1) or (T, N)
    if pressure.dim() == 2:
        pressure = pressure.unsqueeze(-1)                   # (T, N, 1)

    return {
        "mesh_pos": mesh_pos.float(),
        "velocity": velocity.float(),
        "pressure": pressure.float(),
        "node_type": node_type.int(),
        "cells": cells.int(),
    }


def convert_split(
    dataset_root: Path,
    output_root: Path,
    split: str,
    meta: dict,
    overwrite: bool,
) -> list[str]:
    """Convert one split's tfrecord file, returning the list of trajectory IDs.

    Args:
        dataset_root: Directory containing the raw tfrecord files.
        output_root: Root of the output directory tree.
        split: Split name, e.g. ``"train"``.
        meta: Parsed meta.json.
        overwrite: Re-convert existing trajectories when True.

    Returns:
        List of trajectory directory names (relative to *output_root*).
    """
    tfrecord_path = dataset_root / f"{split}.tfrecord"
    if not tfrecord_path.exists():
        raise FileNotFoundError(f"tfrecord not found: {tfrecord_path}")

    trajectory_ids: list[str] = []

    for idx, raw in enumerate(
        tqdm(_iter_trajectories(tfrecord_path, meta), desc=f"  {split}", unit="traj")
    ):
        traj_name = f"{split}_{idx:06d}"
        traj_dir = output_root / traj_name
        sentinel = traj_dir / "velocity.pt"

        if sentinel.exists() and not overwrite:
            trajectory_ids.append(traj_name)
            continue

        traj_dir.mkdir(parents=True, exist_ok=True)
        tensors = _trajectory_to_tensors(raw)
        for name, tensor in tensors.items():
            torch.save(tensor, traj_dir / f"{name}.pt")

        trajectory_ids.append(traj_name)

    return trajectory_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DeepMind MeshGraphNets cylinder_flow tfrecords to .pt tensors."
    )
    parser.add_argument(
        "--dataset-root", required=True, type=Path,
        help="Directory containing train.tfrecord, test.tfrecord, meta.json.",
    )
    parser.add_argument(
        "--output-root", required=True, type=Path,
        help="Output directory. Will be created if it does not exist.",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "test", "valid"],
        help="Which splits to convert (default: train test valid).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-convert trajectories that already have .pt files.",
    )
    args = parser.parse_args()

    meta = _parse_meta(args.dataset_root)
    traj_length = meta.get("trajectory_length", "?")
    print(f"Dataset: trajectory_length={traj_length}, fields={list(meta['features'].keys())}")

    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, list[str]] = {}
    t0 = time.perf_counter()

    for split in args.splits:
        tfrecord = args.dataset_root / f"{split}.tfrecord"
        if not tfrecord.exists():
            print(f"Skipping split '{split}': {tfrecord} not found.")
            continue
        print(f"Converting '{split}'...")
        manifest[split] = convert_split(
            args.dataset_root, args.output_root, split, meta, args.overwrite
        )
        print(f"  {len(manifest[split])} trajectories")

    manifest_path = args.output_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest -> {manifest_path}")

    elapsed = time.perf_counter() - t0
    total = sum(len(v) for v in manifest.values())
    print(f"Done in {elapsed:.1f}s — {total} trajectories total.")


if __name__ == "__main__":
    main()
