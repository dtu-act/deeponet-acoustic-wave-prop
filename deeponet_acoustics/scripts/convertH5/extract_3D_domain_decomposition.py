# ==============================================================================
# Copyright 2025 Technical University of Denmark
# Author: Nikolas Borrel-Jensen
#
# All Rights Reserved.
#
# Licensed under the MIT License.
#
# 3D: used for converting the full dome to quarter dome for domain decomposition
# ==============================================================================
import argparse
import os
import shutil
from pathlib import Path

import datahandlers.io as IO
import h5py
import numpy as np


def run_domain_decomposition(input_path, output_path):
    """
    Process all H5 files in input directory for domain decomposition.

    Converts full dome acoustic simulation data to quarter dome by extracting
    a specific spatial domain. Copies simulation parameters and processes
    all H5 files found in the input directory.

    Args:
        input_path: Directory containing H5 files to process
        output_path: Directory where processed files will be saved
    """
    if Path(output_path).exists():
        shutil.rmtree(output_path)
    Path(output_path).mkdir(parents=True, exist_ok=False)

    # copy settings file
    source_settings_path = os.path.join(input_path, "simulation_parameters.json")
    if Path(source_settings_path).exists():
        shutil.copy(
            source_settings_path,
            os.path.join(output_path, "simulation_parameters.json"),
        )

    filenamesH5 = IO.pathsToFileType(input_path, ".h5")

    N = len(filenamesH5)
    print(f"Number of folders to process: {len(filenamesH5)}\n")

    extract_domain = [[0.0, 3.0], [0.0, 3.0], [0.0, 4.0]]

    for i, path_to_filename in enumerate(filenamesH5):
        print(f"Processing {i}/{N}\n")
        print(f"Input: {path_to_filename}\n")

        last_directory = os.path.basename(os.path.dirname(path_to_filename))
        new_base_dir = os.path.join(output_path, last_directory)

        # create folder for the sample
        Path(new_base_dir).mkdir(parents=True, exist_ok=True)

        path_to_new_filename = os.path.join(
            new_base_dir, os.path.basename(path_to_filename)
        )

        extract_domain_subdomain(
            path_to_filename, path_to_new_filename, extract_domain, dtype=np.float16
        )
        print(f"written file: {path_to_new_filename}\n")


def extract_domain_subdomain(
    path_data_in, path_data_out, domain_extract, dtype=np.float16
):
    """
    Extract a specific spatial domain from 3D acoustic simulation data.

    Filters mesh points and corresponding pressure data to only include
    points within the specified domain boundaries.

    Args:
        path_data_in: Input H5 file path
        path_data_out: Output H5 file path
        domain_extract: List of [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        dtype: Output data type for pressure arrays (default: np.float16)
    """
    with h5py.File(path_data_in, "r") as f:
        mesh = f["mesh"]
        pressures = f["pressures"]
        source_position = f["source_position"]
        umesh = f["umesh"]
        upressures = f["upressures"]

        time_steps = pressures.attrs["time_steps"]
        ushape = f["umesh"].attrs["umesh_shape"]

        mask = (mesh > np.array(domain_extract)[:, 0]) & (
            mesh < np.array(domain_extract)[:, 1]
        )
        mask = np.logical_and.reduce(mask, axis=1)
        mesh = mesh[mask]
        pressures = pressures[:, mask]

        with h5py.File(path_data_out, "w") as fw:
            fw.create_dataset("mesh", data=mesh, dtype=np.float32)
            pressures_new = fw.create_dataset("pressures", data=pressures, dtype=dtype)
            fw.create_dataset("source_position", data=source_position, dtype=np.float32)
            umesh_new = fw.create_dataset("umesh", data=umesh, dtype=np.float32)
            fw.create_dataset("upressures", data=upressures, dtype=dtype)

            pressures_new.attrs.create("time_steps", time_steps, dtype=np.float32)
            umesh_new.attrs.create("umesh_shape", ushape, dtype=np.float32)


if __name__ == "__main__":
    """
    Example usage:
    input_path = "/work3/nibor/1TB/libP/bilbao_1000hz_p4_5ppw_srcpos5_val/"
    output_path = "/work3/nibor/1TB/libP/bilbao_1000hz_p4_5ppw_srcpos5_1stquad_val/"
    """
    parser = argparse.ArgumentParser(
        description="Extract specific spatial domain from 3D H5 files for domain decomposition"
    )
    parser.add_argument(
        "--input_dir", required=True, help="Directory containing H5 files to process"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where processed files will be saved",
    )

    args = parser.parse_args()

    run_domain_decomposition(args.input_dir, args.output_dir)
