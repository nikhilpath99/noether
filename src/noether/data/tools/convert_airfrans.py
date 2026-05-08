#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

"""Convert raw AirFRANS VTK files to PyTorch .pt tensors for use with Noether.

Each simulation folder is converted in-place or to a mirrored output directory.
The following files are produced per simulation:

    volume_position.pt      (N, 2)  x/y coordinates of volume mesh points
    volume_velocity.pt      (N, 2)  U_x / U_y velocity components
    volume_pressure.pt      (N, 1)  static pressure
    surface_position.pt     (M, 2)  x/y coordinates of aerofoil surface points
    surface_pressure.pt     (M, 1)  surface pressure
    surface_normals.pt      (M, 2)  outward surface normals (x/y components)
    design_parameters.pt    (6,)    [velocity, aoa_deg, p1, p2, p3, p4]
                                    p4 = 0 for NACA 4-digit airfoils

Usage:
    python -m noether.data.tools.convert_airfrans \\
        --dataset-root E:/Code/experiment_runner/Data/Dataset \\
        --output-root  E:/Code/experiment_runner/Data/Dataset_converted \\
        --workers 4
"""

import argparse
import json
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Folder-name parsing
# ---------------------------------------------------------------------------

def parse_sim_name(name: str) -> dict:
    """Parse an AirFRANS simulation folder name into its boundary conditions.

    Format: ``airFoil2D_SST_{velocity}_{aoa}_{naca_p1}[_{naca_p2}_{naca_p3}[_{naca_p4}]]``

    Args:
        name: Folder name, e.g. ``airFoil2D_SST_43.597_5.932_3.551_3.1_1.0_18.252``.

    Returns:
        Dict with keys ``velocity``, ``aoa``, ``naca_params`` (list[float]).

    Raises:
        ValueError: If the name cannot be parsed.
    """
    parts = name.split("_")
    if len(parts) < 6 or parts[0] != "airFoil2D" or parts[1] != "SST":
        raise ValueError(f"Unexpected folder name format: {name!r}")
    velocity = float(parts[2])
    aoa = float(parts[3])
    naca_params = [float(p) for p in parts[4:]]
    return {"velocity": velocity, "aoa": aoa, "naca_params": naca_params}


def design_parameters_tensor(velocity: float, aoa: float, naca_params: list[float]) -> torch.Tensor:
    """Build the (6,) design-parameter tensor, zero-padding 4-digit NACA airfoils.

    Layout: ``[velocity, aoa_deg, naca_p1, naca_p2, naca_p3, naca_p4]``

    Args:
        velocity: Inlet velocity magnitude in m/s.
        aoa: Angle of attack in degrees.
        naca_params: NACA shape parameters (3 values for 4-digit, 4 for 5-digit).

    Returns:
        Float32 tensor of shape (6,).
    """
    padded = (naca_params + [0.0] * 4)[:4]
    return torch.tensor([velocity, aoa] + padded, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Per-simulation conversion
# ---------------------------------------------------------------------------

def convert_simulation(
    sim_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> str:
    """Convert one AirFRANS simulation folder to .pt tensors.

    Args:
        sim_dir: Source directory containing the raw VTK files.
        output_dir: Destination directory (created if needed).
        overwrite: If False, skip already-converted simulations.

    Returns:
        Status string for logging (``"done"``, ``"skip"``, or ``"error: ...``).
    """
    name = sim_dir.name

    sentinel = output_dir / "volume_position.pt"
    if sentinel.exists() and not overwrite:
        return "skip"

    try:
        bc = parse_sim_name(name)
    except ValueError as e:
        return f"error: {e}"

    try:
        internal = pv.read(sim_dir / f"{name}_internal.vtu")
        aerofoil = pv.read(sim_dir / f"{name}_aerofoil.vtp")
    except Exception as e:
        return f"error reading VTK: {e}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Volume ---
    vol_pts = np.array(internal.points, dtype=np.float32)[:, :2]       # (N, 2)
    vol_U   = np.array(internal.point_data["U"], dtype=np.float32)[:, :2]  # (N, 2)
    vol_p   = np.array(internal.point_data["p"], dtype=np.float32)     # (N,)

    torch.save(torch.from_numpy(vol_pts),          output_dir / "volume_position.pt")
    torch.save(torch.from_numpy(vol_U),            output_dir / "volume_velocity.pt")
    torch.save(torch.from_numpy(vol_p), output_dir / "volume_pressure.pt")

    # --- Surface ---
    srf_pts = np.array(aerofoil.points, dtype=np.float32)[:, :2]       # (M, 2)
    srf_p   = np.array(aerofoil.point_data["p"], dtype=np.float32)     # (M,)
    srf_N   = np.array(aerofoil.point_data["Normals"], dtype=np.float32)[:, :2]  # (M, 2)

    torch.save(torch.from_numpy(srf_pts),          output_dir / "surface_position.pt")
    torch.save(torch.from_numpy(srf_p), output_dir / "surface_pressure.pt")
    torch.save(torch.from_numpy(srf_N),            output_dir / "surface_normals.pt")

    # --- Design parameters ---
    dp = design_parameters_tensor(bc["velocity"], bc["aoa"], bc["naca_params"])
    torch.save(dp, output_dir / "design_parameters.pt")

    return "done"


def _worker(args: tuple) -> tuple[str, str]:
    sim_dir, output_dir, overwrite = args
    status = convert_simulation(sim_dir, output_dir, overwrite)
    return sim_dir.name, status


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert AirFRANS VTK dataset to Noether .pt format."
    )
    parser.add_argument("--dataset-root", required=True, type=Path,
                        help="Root directory with airFoil2D_* simulation folders.")
    parser.add_argument("--output-root", required=True, type=Path,
                        help="Destination root. Mirrors the source folder structure.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-convert simulations that already have .pt files.")
    args = parser.parse_args()

    src_root: Path = args.dataset_root
    out_root: Path = args.output_root

    sim_dirs = sorted(p for p in src_root.iterdir() if p.is_dir() and p.name.startswith("airFoil2D"))
    if not sim_dirs:
        print(f"No airFoil2D_* directories found in {src_root}")
        return

    # Copy manifest to output root
    manifest_src = src_root / "manifest.json"
    if manifest_src.exists():
        out_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_src, out_root / "manifest.json")
        print(f"Copied manifest.json -> {out_root / 'manifest.json'}")

    tasks = [
        (sim_dir, out_root / sim_dir.name, args.overwrite)
        for sim_dir in sim_dirs
    ]

    done = skip = errors = 0
    t0 = time.perf_counter()

    with Pool(processes=args.workers) as pool:
        for name, status in tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks), desc="Converting"):
            if status == "done":
                done += 1
            elif status == "skip":
                skip += 1
            else:
                errors += 1
                tqdm.write(f"  [ERROR] {name}: {status}")

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s — converted: {done}, skipped: {skip}, errors: {errors}")


if __name__ == "__main__":
    main()
