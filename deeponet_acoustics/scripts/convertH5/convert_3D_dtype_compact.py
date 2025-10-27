# ==============================================================================
# Copyright 2025 Technical University of Denmark
# Author: Nikolas Borrel-Jensen
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import argparse
import os
import pathlib

import h5py
import numpy as np

import deeponet_acoustics.datahandlers.io as IO


def convert_to_dtype_compact(
    path_data_in, path_data_out, temporal_prune_skip=1, dtype=np.float16
):
    """
    Convert 3D H5 file data type and optionally downsample temporal resolution.

    Converts pressure and upressure data to a more compact data type (typically float16)
    to reduce file size. Can also downsample temporal resolution by skipping time steps.

    Args:
        path_data_in: Input H5 file path
        path_data_out: Output H5 file path
        temporal_prune_skip: Skip factor for temporal downsampling (default: 1, no skipping)
        dtype: Target data type for pressure arrays (default: np.float16)
    """
    with h5py.File(path_data_in, "r") as f:
        mesh = f["mesh"]
        pressures = f["pressures"]
        source_position = f["source_position"]
        umesh = f["umesh"]
        upressures = f["upressures"]

        time_steps = pressures.attrs["time_steps"]
        ushape = f["umesh"].attrs["umesh_shape"]

        pressures = pressures[::temporal_prune_skip, :]
        time_steps = time_steps[::temporal_prune_skip]

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
    base_path = "/work3/nibor/1TB/libP/bilbao_1000hz_p4_5ppw_srcpos5_val"
    """
    parser = argparse.ArgumentParser(
        description="Convert 3D H5 files to compact data type (float16)"
    )
    parser.add_argument(
        "--base_path", required=True, help="Directory containing H5 files to process"
    )

    args = parser.parse_args()

    filenamesH5 = IO.pathsToFileType(args.base_path, ".h5")

    N = len(filenamesH5)
    print(f"Number of files to process: {N}\n")

    for i, filename in enumerate(filenamesH5):
        print(f"Processing {i + 1}/{N}")
        print(f"Input: {filename}")

        filename_out = pathlib.Path(filename).with_suffix("")
        filename_out = str(filename_out) + "_float16.h5"
        convert_to_dtype_compact(filename, filename_out, dtype=np.float16)
        print(f"Written file: {filename_out}")
        os.remove(filename)
        print(f"Deleted original file: {filename}")
        print()
