#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch

from noether.core.schemas.dataset import DatasetBaseConfig
from noether.data import Dataset, with_normalizers
from noether.core.schemas.filemap import FileMap


class AeroDataset(Dataset):
    """Dataset implementation for aerodynamic datasets with volume and surface fields.
    This unified dataset class provides an interface for aerodynamics dataset with volume and surface fields.
    The dataset behavior such as the dataset choice, train/val/test split IDs, etc.
    is configured through constructor parameters, allowing for easy extension to new datasets.

    """

    def __init__(self, dataset_config: DatasetBaseConfig, filemap: FileMap) -> None:
        """

        Args:
            dataset_config: Configuration for the dataset. See :class:`~noether.core.schemas.dataset.DatasetBaseConfig` for available options.
            filemap: FileMap object defining the mapping of data properties to filenames. See :class:`~noether.data.datasets.cfd.caeml.filemap.FileMap` for details."""
        super().__init__(dataset_config=dataset_config)
        self.filemap = filemap

    def __len__(self):
        raise NotImplementedError

    def _load_from_disk(self, idx: int, filename: str) -> torch.Tensor:
        """
        Method to load data from disk. Must be implemented by subclasses (i.e., specific datasets).
        """
        raise NotImplementedError

    def _load(self, idx: int, filename: str) -> torch.Tensor:
        return self._load_from_disk(idx=idx, filename=filename)

    @with_normalizers
    def getitem_surface_position(self, idx: int) -> torch.Tensor:
        """Retrieves surface positions (num_surface_points, 3)"""
        return self._load(idx=idx, filename=self.filemap.surface_position)  # type: ignore[arg-type]

    @with_normalizers
    def getitem_surface_pressure(self, idx: int) -> torch.Tensor:
        """Retrieves surface pressures (num_surface_points, 1)"""
        return self._load(idx=idx, filename=self.filemap.surface_pressure).unsqueeze(1)  # type: ignore[arg-type]

    @with_normalizers
    def getitem_surface_friction(self, idx: int) -> torch.Tensor:
        """Retrieves surface friction (=wallshearstress) (num_surface_points, 3)"""
        return self._load(idx=idx, filename=self.filemap.surface_friction)  # type: ignore[arg-type]

    @with_normalizers
    def getitem_volume_position(self, idx: int) -> torch.Tensor:
        """Retrieves volume position (num_volume_points, 3)"""
        return self._load(idx=idx, filename=self.filemap.volume_position)  # type: ignore[arg-type]

    @with_normalizers
    def getitem_volume_pressure(self, idx: int) -> torch.Tensor:
        """Retrieves volume pressures (num_volume_points, 1)"""
        return self._load(idx=idx, filename=self.filemap.volume_pressure).unsqueeze(1)  # type: ignore[arg-type]

    @with_normalizers
    def getitem_volume_velocity(self, idx: int) -> torch.Tensor:
        """Retrieves volume velocity (num_volume_points, 3)"""
        return self._load(idx=idx, filename=self.filemap.volume_velocity)  # type: ignore[arg-type]

    @with_normalizers
    def getitem_volume_vorticity(self, idx: int) -> torch.Tensor:
        """Retrieves volume vorticity (num_volume_points, 3)"""
        return self._load(idx=idx, filename=self.filemap.volume_vorticity)  # type: ignore[arg-type]

    @with_normalizers
    def getitem_volume_sdf(self, idx: int) -> torch.Tensor:
        """Retrieve signed distance field at volume points."""
        return self._load(idx=idx, filename=self.filemap.volume_distance_to_surface).unsqueeze(1)  # type: ignore[arg-type]

    @with_normalizers("volume_sdf")
    def getitem_surface_sdf(self, idx: int) -> torch.Tensor:
        """Retrieve signed distance field at surface points. This is always 0.0, but we still create a sample processor for it to be able to easily concatenate it with the surface normals."""
        return torch.zeros(self.getitem_surface_normals(idx).shape[0], 1)

    def getitem_volume_normals(self, idx: int) -> torch.Tensor:
        """Retrieve normal vectors at volume points."""
        return self._load(idx=idx, filename=self.filemap.volume_normals)  # type: ignore[arg-type]

    def getitem_surface_normals(self, idx: int) -> torch.Tensor:
        """Retrieve surface normal vectors."""
        return self._load(idx=idx, filename=self.filemap.surface_normals)  # type: ignore[arg-type]

    def getitem_volume_importance_weights(self, idx: int) -> torch.Tensor:
        """Retrieve pre-computed gradient importance weights for volume points (N,)."""
        return self._load(idx=idx, filename=self.filemap.volume_importance_weights)  # type: ignore[arg-type]

    def getitem_surface_importance_weights(self, idx: int) -> torch.Tensor:
        """Retrieve pre-computed gradient importance weights for surface points (N,)."""
        return self._load(idx=idx, filename=self.filemap.surface_importance_weights)  # type: ignore[arg-type]

    def getitem_design_parameters(self, idx: int) -> torch.Tensor:
        """Retrieve design / boundary-condition parameters tensor."""
        return self._load(idx=idx, filename=self.filemap.design_parameters)  # type: ignore[arg-type]

    def get_all_getitem_names(self) -> list[str]:
        """Returns getitem method names, excluding loaders whose filemap entry is None."""
        names = super().get_all_getitem_names()
        fm = self.filemap
        guards = {
            "getitem_surface_position":            fm.surface_position,
            "getitem_surface_pressure":            fm.surface_pressure,
            "getitem_surface_friction":            fm.surface_friction,
            "getitem_surface_normals":             fm.surface_normals,
            "getitem_volume_position":             fm.volume_position,
            "getitem_volume_pressure":             fm.volume_pressure,
            "getitem_volume_velocity":             fm.volume_velocity,
            "getitem_volume_vorticity":            fm.volume_vorticity,
            "getitem_volume_normals":              fm.volume_normals,
            "getitem_volume_sdf":                  fm.volume_distance_to_surface,
            "getitem_surface_sdf":                 fm.volume_distance_to_surface,
            "getitem_design_parameters":           fm.design_parameters,
            "getitem_volume_importance_weights":   fm.volume_importance_weights,
            "getitem_surface_importance_weights":  fm.surface_importance_weights,
        }
        return [n for n in names if guards.get(n, True) is not None]
