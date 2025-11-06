# ==============================================================================
# Copyright 2025 Technical University of Denmark
# Author: Nikolas Borrel-Jensen
#
# All Rights Reserved.
#
# Licensed under the MIT License.
#
# Convert 2D data (generated with Matlab) to 3D data format (separate H5 files per source)

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np


def create_metadata_json(
    src_folder: Path, pressure_attrs: dict, src_pos: np.ndarray | None
) -> None:
    """
    Create simulation parameters JSON file for a source.

    Args:
        src_folder: Path to source folder
        pressure_attrs: Pressure attributes from H5 file
        src_pos: Source position array
    """
    simulation_params = {
        "SimulationParameters": {
            "SourcePosition": src_pos.tolist() if src_pos is not None else [],
            "c": float(pressure_attrs["c_phys"]),
            "dt": float(pressure_attrs["dt"]),
            "fmax": float(pressure_attrs["fmax"]),
            "rho": float(pressure_attrs["rho"]),
            "sigma": float(pressure_attrs["sigma0"]),
        }
    }

    metadata_file = src_folder / "simulation_parameters.json"
    with open(metadata_file, "w") as f_meta:
        json.dump(simulation_params, f_meta, indent=4)


def split_2d_by_source_position(input_file: Path, output_path: Path):
    """
    Convert 2D H5 file format to 3D format by creating separate files for each source position.

    Args:
        input_file: Path to input 2D H5 file
        output_dir: Directory to save output 3D format files
    """
    output_path = output_path / input_file.stem
    output_path.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_file, "r") as f_in:
        # Read common data
        mesh = f_in["mesh"][:]
        t = f_in["t"][:]
        umesh = f_in["umesh"][:]
        umesh_shape = f_in["umesh"].attrs["umesh_shape"]

        # Read pressure data and source positions
        pressures = f_in["pressures"][
            :
        ]  # Shape: (num_sources, time_steps, spatial_points)
        upressures = f_in["upressures"][:]  # Shape: (num_sources, uniform_grid_points)
        x0_srcs = (
            f_in["x0_srcs"][:] if "x0_srcs" in f_in else None
        )  # Shape: (num_sources * spatial_dims)

        x0_srcs = x0_srcs.reshape(x0_srcs.shape[0], -1)

        # Get pressure attributes
        pressure_attrs = dict(f_in["pressures"].attrs)
        upressure_attrs = dict(f_in["upressures"].attrs)
        mesh_attrs = dict(f_in["mesh"].attrs)

        num_sources = x0_srcs.shape[0] if x0_srcs is not None else pressures.shape[0]

        if num_sources != pressures.shape[0]:
            print(
                f"WARNING: There is a mismatch between number of x0_srcs ({num_sources}) and the actual sources used to simulate the pressure fields {pressures.shape[0]}. Source positions omitted."
            )
            num_sources = pressures.shape[0]
            x0_srcs = None

        print(f"Converting {num_sources} source positions from 2D to 3D format...")

        # Compute denormalized values once for all sources
        c_phys = float(pressure_attrs["c_phys"])
        fmax_normalized = float(pressure_attrs["fmax"])
        fmax_phys = round(fmax_normalized * c_phys, 1)
        dt_normalized = float(pressure_attrs["dt"])
        dt_phys = dt_normalized / c_phys

        # Update pressure_attrs with denormalized values for use in root metadata
        pressure_attrs_denormalized = pressure_attrs.copy()
        pressure_attrs_denormalized["fmax"] = fmax_phys
        pressure_attrs_denormalized["dt"] = dt_phys

        src_pos_2d = None

        for src_idx in range(num_sources):
            # Create separate folder for each source
            src_folder = output_path / f"source_{src_idx:03d}"
            src_folder.mkdir(exist_ok=True)

            # Create output filename in the source folder
            output_file = src_folder / f"source_{src_idx:03d}.h5"

            with h5py.File(output_file, "w") as f_out:
                # Keep mesh as 2D
                # Note: Adding attributes that weren't in original 3D data for consistency
                mesh_dataset = f_out.create_dataset("mesh", data=mesh)
                mesh_dataset.attrs["domain_minmax"] = mesh_attrs.get(
                    "domain_minmax", np.array([[0, 2], [0, 2]])
                )

                # Keep umesh as 2D
                umesh_dataset = f_out.create_dataset("umesh", data=umesh)
                umesh_dataset.attrs["umesh_shape"] = umesh_shape.astype(
                    np.int64
                )  # Keep as int

                # Extract pressure data for this source (transpose to match 3D format)
                src_pressures = pressures[
                    src_idx, :, :
                ]  # From (src_pos, time_steps, spatial_points) to (time_steps, spatial_points)
                pressures_dataset = f_out.create_dataset(
                    "pressures", data=src_pressures
                )

                # Denormalize time_steps by dividing by physical speed of sound
                time_steps_phys = (t / c_phys).astype(np.float32)
                pressures_dataset.attrs["time_steps"] = time_steps_phys

                # Note: Adding all pressure attributes that weren't in original 3D data
                for attr_name, attr_value in pressure_attrs_denormalized.items():
                    if attr_name != "time_steps":  # Avoid duplicate
                        pressures_dataset.attrs[attr_name] = attr_value

                if x0_srcs is not None:
                    # Extract source position and keep as 2D
                    src_pos_2d = x0_srcs[
                        src_idx, :
                    ]  # Extract (spatial_dims,) position for this source
                    f_out.create_dataset(
                        "source_position", data=src_pos_2d.astype(np.float32)
                    )

                # Extract upressures for this source
                src_upressures = upressures[
                    src_idx, :
                ]  # Extract (uniform_grid_points,) for this source
                upressures_dataset = f_out.create_dataset(
                    "upressures", data=src_upressures
                )
                # Note: Adding upressures attributes that weren't in original 3D data
                for attr_name, attr_value in upressure_attrs.items():
                    upressures_dataset.attrs[attr_name] = attr_value

                # Create metadata JSON file in source folder using corrected attributes
                create_metadata_json(
                    src_folder, dict(pressures_dataset.attrs), src_pos_2d
                )

            print(f"Created {output_file}")

        # Create metadata JSON file in root directory as well
        create_metadata_json(output_path, pressure_attrs_denormalized, None)

        print(f"Conversion complete! Created {num_sources} files in {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert 2D H5 data format to 3D format"
    )
    parser.add_argument(
        "--input_file", required=True, type=Path, help="Input 2D H5 file path"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Output directory for 3D format files",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist")
        return 1

    split_2d_by_source_position(args.input_file, args.output_dir)


if __name__ == "__main__":
    exit(main())
