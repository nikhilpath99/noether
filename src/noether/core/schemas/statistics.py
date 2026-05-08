#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections.abc import Sequence

from pydantic import BaseModel


class AeroStatsSchema(BaseModel):
    """Unified statistics dataclass for aerodynamics datasets such as AhmedML, and DrivAerML, DrivAerNet++,
    ShapeNet-Car, Wing, and AirFRANS.

    Position and velocity fields accept variable-length sequences so that both
    2-D datasets (e.g. AirFRANS) and 3-D datasets share the same schema.
    """

    # Surface statistics
    surface_domain_min: Sequence[float] | None = None
    surface_domain_max: Sequence[float] | None = None
    surface_pos_mean: Sequence[float] | None = None
    surface_pos_std: Sequence[float] | None = None
    surface_pressure_mean: Sequence[float] | None = None
    surface_pressure_std: Sequence[float] | None = None
    surface_friction_mean: Sequence[float] | None = None
    surface_friction_std: Sequence[float] | None = None

    # Volume statistics
    volume_pos_mean: Sequence[float] | None = None
    volume_pos_std: Sequence[float] | None = None
    volume_pressure_mean: Sequence[float] | None = None
    volume_pressure_std: Sequence[float] | None = None
    volume_velocity_mean: Sequence[float] | None = None
    volume_velocity_std: Sequence[float] | None = None
    volume_vorticity_mean: Sequence[float] | None = None
    volume_vorticity_std: Sequence[float] | None = None
    volume_vorticity_logscale_mean: Sequence[float] | None = None
    volume_vorticity_logscale_std: Sequence[float] | None = None
    volume_vorticity_magnitude_mean: float | None = None
    volume_vorticity_magnitude_std: float | None = None
    volume_domain_min: tuple[float, float, float] | None = None
    volume_domain_max: tuple[float, float, float] | None = None
    volume_sdf_mean: tuple[float] | None = None
    volume_sdf_std: tuple[float] | None = None

    # Inflow design parameter statistics
    inflow_design_parameters_min: Sequence[float] | None = None
    inflow_design_parameters_max: Sequence[float] | None = None
    inflow_design_parameters_mean: Sequence[float] | None = None
    inflow_design_parameters_std: Sequence[float] | None = None

    # Geometry design parameter statistics
    geometry_design_parameters_min: Sequence[float] | None = None
    geometry_design_parameters_max: Sequence[float] | None = None
    geometry_design_parameters_mean: Sequence[float] | None = None
    geometry_design_parameters_std: Sequence[float] | None = None

    # raw position statistics
    raw_pos_min: Sequence[float] | None = None
    raw_pos_max: Sequence[float] | None = None
