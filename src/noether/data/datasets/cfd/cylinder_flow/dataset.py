#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""DeepMind MeshGraphNets cylinder_flow dataset."""

import json
import logging
from pathlib import Path
from typing import Literal

import torch

from noether.core.utils.common import validate_path

logger = logging.getLogger(__name__)

CylinderFlowSplit = Literal["train", "test", "valid"]


class CylinderFlowDataset(torch.utils.data.Dataset):
    """Dataset for the DeepMind MeshGraphNets cylinder_flow benchmark.

    Loads trajectories of 2D unsteady flow around a cylinder converted to
    ``.pt`` tensors by ``convert_cylinder_flow``.  Each sample is one full
    trajectory — all *T* timesteps — so the model or training loop decides
    how to sample individual steps or windows.

    Fields returned per sample (all tensors):

    +--------------+-------------------+---------+-------------------------------------------+
    | Key          | Shape             | dtype   | Description                               |
    +==============+===================+=========+===========================================+
    | mesh_pos     | (N, 2)            | float32 | x/y mesh-node positions (fixed per traj)  |
    +--------------+-------------------+---------+-------------------------------------------+
    | velocity     | (T, N, 2)         | float32 | U_x / U_y at each timestep                |
    +--------------+-------------------+---------+-------------------------------------------+
    | pressure     | (T, N, 1)         | float32 | static pressure at each timestep          |
    +--------------+-------------------+---------+-------------------------------------------+
    | node_type    | (N,)              | int32   | 0=fluid, 4=wall, 5=inflow, 6=outflow      |
    +--------------+-------------------+---------+-------------------------------------------+
    | cells        | (E, 3)            | int32   | triangular mesh connectivity              |
    +--------------+-------------------+---------+-------------------------------------------+

    Memory note:
        A typical trajectory (T=600, N≈1 885) occupies ~13 MB on disk and in
        RAM.  With a DataLoader of ``num_workers=4`` a single batch of four
        trajectories needs ≈52 MB of worker memory.

    Args:
        root: Root directory produced by ``convert_cylinder_flow``, containing
            ``manifest.json`` and one sub-directory per trajectory.
        split: Which manifest split to load.
        manifest: Filename of the manifest inside *root* (default:
            ``manifest.json``).

    Example:

        .. testcode::

            from noether.data.datasets.cfd.cylinder_flow import CylinderFlowDataset
            ds = CylinderFlowDataset(root="/data/cylinder_flow_pt", split="train")
            sample = ds[0]
            # sample["velocity"].shape -> (600, N, 2)
    """

    def __init__(
        self,
        root: str | Path,
        split: CylinderFlowSplit,
        manifest: str = "manifest.json",
    ) -> None:
        """
        Args:
            root: Converted dataset root directory.
            split: Split to load (``"train"``, ``"test"``, or ``"valid"``).
            manifest: Manifest filename inside *root*.

        Raises:
            FileNotFoundError: If *root* or the manifest does not exist.
            ValueError: If *split* is not in the manifest.
        """
        self.root = validate_path(root)
        self.split = split

        manifest_path = self.root / manifest
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}. "
                "Run convert_cylinder_flow first."
            )

        with open(manifest_path) as f:
            full_manifest = json.load(f)

        if split not in full_manifest:
            raise ValueError(
                f"Split '{split}' not in manifest. "
                f"Available: {list(full_manifest.keys())}"
            )

        self.trajectory_ids: list[str] = full_manifest[split]
        logger.info(
            "CylinderFlowDataset split='%s' — %d trajectories from %s",
            split,
            len(self.trajectory_ids),
            self.root,
        )

    def __len__(self) -> int:
        return len(self.trajectory_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load and return one full trajectory.

        Args:
            idx: Trajectory index in ``[0, len(dataset))``.

        Returns:
            Dict with keys ``mesh_pos``, ``velocity``, ``pressure``,
            ``node_type``, ``cells``.

        Raises:
            RuntimeError: If any expected tensor file cannot be loaded.
        """
        traj_dir = self.root / self.trajectory_ids[idx]
        fields = ["mesh_pos", "velocity", "pressure", "node_type", "cells"]
        sample: dict[str, torch.Tensor] = {}
        for field in fields:
            path = traj_dir / f"{field}.pt"
            try:
                sample[field] = torch.load(path, weights_only=True)
            except Exception as e:
                raise RuntimeError(f"Failed to load {path}: {e}") from e
        return sample

    def sample_info(self, idx: int) -> dict:
        """Return metadata about trajectory *idx*.

        Args:
            idx: Trajectory index.

        Returns:
            Dict with ``trajectory_id``, ``sample_uri``, and ``split``.
        """
        traj_id = self.trajectory_ids[idx]
        return {
            "trajectory_id": traj_id,
            "sample_uri": self.root / traj_id,
            "split": self.split,
        }

    def num_timesteps(self, idx: int = 0) -> int:
        """Return the number of timesteps in trajectory *idx*.

        Args:
            idx: Trajectory index (default 0).

        Returns:
            T — the temporal dimension of the velocity tensor.
        """
        path = self.root / self.trajectory_ids[idx] / "velocity.pt"
        t = torch.load(path, weights_only=True)
        return t.shape[0]
