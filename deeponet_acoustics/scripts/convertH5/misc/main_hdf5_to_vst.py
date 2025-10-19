# ==============================================================================
# Copyright 2025 Technical University of Denmark
# Author: Nikolas Borrel-Jensen
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import meshio

import deeponet_acoustics.datahandlers.data_rw as rw
import deeponet_acoustics.setup.parsers as parsers
from deeponet_acoustics.setup.settings import SimulationSettings

settings_path = "scripts/deeponet_2D_local.json"

settings_dict = parsers.parseSettings(settings_path)
settings = SimulationSettings(settings_dict)

# load training data
grid, t_range, p, v, conn, *_ = rw.loadDataFromH5(settings.dirs.test_data_path)

cells = [("triangle", conn[:, 0:3] - 1)]

for i, t in enumerate(t_range):
    # write ptk
    mesh = meshio.Mesh(
        grid,
        cells,
        point_data={"p": p[0, i, :]},
    )

    filename = f"test1/test_{i}.vtu"
    mesh.write(f"{settings.dirs.data_dir}/{filename}")
