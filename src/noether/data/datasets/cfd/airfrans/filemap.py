#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.schemas.filemap import FileMap

AIRFRANS_FILEMAP = FileMap(
    surface_position="surface_position.pt",
    surface_pressure="surface_pressure.pt",
    surface_normals="surface_normals.pt",
    volume_position="volume_position.pt",
    volume_pressure="volume_pressure.pt",
    volume_velocity="volume_velocity.pt",
    design_parameters="design_parameters.pt",
)
