#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from .airfrans import AirFRANSDataset, AirFRANSDatasetConfig, AIRFRANS_FILEMAP
from .caeml.ahmedml import AhmedMLDataset, AhmedMLDefaultSplitIDs
from .caeml.drivaerml import DrivAerMLDataset, DrivAerMLDefaultSplitIDs
from .cylinder_flow import CylinderFlowDataset, CylinderFlowSplit
from .drivaernet.dataset import DrivAerNetDataset
from .emmi_wing import EmmiWingDataset, EmmiWingHFDataset
from .shapenet_car import ShapeNetCarDataset, ShapeNetCarDefaultSplitIDs
from .simshift_heatsink import SimshiftHeatsinkDataset

__all__ = [
    "AirFRANSDataset",
    "AirFRANSDatasetConfig",
    "AIRFRANS_FILEMAP",
    "AhmedMLDataset",
    "AhmedMLDefaultSplitIDs",
    "CylinderFlowDataset",
    "CylinderFlowSplit",
    "DrivAerMLDataset",
    "DrivAerMLDefaultSplitIDs",
    "DrivAerNetDataset",
    "EmmiWingDataset",
    "EmmiWingHFDataset",
    "ShapeNetCarDataset",
    "ShapeNetCarDefaultSplitIDs",
    "SimshiftHeatsinkDataset",
]
