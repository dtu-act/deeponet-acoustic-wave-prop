# ==============================================================================
# Copyright 2025 Technical University of Denmark
# Author: Nikolas Borrel-Jensen
#
# All Rights Reserved.
#
# Licensed under the MIT License.
#
# Used for 2D data only: converting from high spatial/temporal resolutions to lower resolutions (data generated with MATLAB)
# ==============================================================================
import argparse

import h5py
import numpy as np


def downsample_2d_ppw(
    path_data_in,
    path_data_out,
    uprune_factor,
    p_ppw,
    t_ppw,
    src_density_fact=1,
    dtype=np.float32,
    indices_p=[],
    indices_t=[],
):
    """
    Convert 2D H5 data from high spatial/temporal resolutions to lower resolutions.
    
    This function downsamples MATLAB-generated 2D acoustic simulation data by reducing
    the points per wavelength (PPW) in spatial, temporal, and uniform grid dimensions.
    
    Args:
        path_data_in: Input H5 file path with high resolution data
        path_data_out: Output H5 file path for downsampled data
        uprune_factor: Factor to downsample uniform grid (must be >= 1)
        p_ppw: Target points per wavelength for spatial dimension
        t_ppw: Target points per wavelength for temporal dimension
        src_density_fact: Factor to downsample source position density (must be <= 1)
        dtype: Output data type (default: np.float32)
        indices_p: Pre-defined spatial indices to use (optional)
        indices_t: Pre-defined temporal indices to use (optional)
        
    Returns:
        np.ndarray: Downsampled time steps array
        
    Raises:
        AssertionError: If upsampling is attempted or invalid parameters provided
    """
    assert src_density_fact <= 1, (
        "cannot upsample src pos density"
    )  # TODO: newer data includes the dx_src, so we won't need src_density_fact

    dim = 2

    with h5py.File(path_data_in, "r") as f:
        mesh = f["mesh"]
        pressures = f["pressures"]
        time_steps = f["t"][()]

        source_position = f["x0_srcs"][()] if "x0_srcs" in f else []
        umesh = f["umesh"]
        upressures = f["upressures"][()]
        minmax = mesh.attrs["domain_minmax"]
        ushape = np.asarray(umesh.attrs["umesh_shape"], dtype=int)
        dx_u = umesh.attrs[
            "dx"
        ]  # only available for newly written data. dx_u_ppw4 = 0.08575 was used for FA data.

        tag_attr = "mesh"

        fmax = f[tag_attr].attrs["fmax"][0]
        c = f[tag_attr].attrs["c"][0]
        # c_phys = f[tag_attr].attrs['c_phys'][0]
        dt = f[tag_attr].attrs["dt"][0]
        dx = f[tag_attr].attrs["dx"][0]

        assert c == 1  # normalized

        dx_p_out = c / (fmax * p_ppw)
        dt_out = 1 / (fmax * t_ppw)

        p_factor = np.round(dx / dx_p_out, decimals=4)
        t_factor = np.round(dt / dt_out, decimals=4)

        dx_p_out = dx / p_factor
        dt_out = dt / t_factor

        assert p_factor <= 1, "cannot upsample dx"
        assert uprune_factor >= 1, "cannot upsample dx_u"
        assert t_factor <= 1, "cannot upsample dt"

        print("--------------------")
        print(f"pprune: {p_factor}")
        print(f"uprune: {uprune_factor}")
        print(f"tprune: {t_factor}")
        print(f"dx_out: {dx_p_out}")
        print(f"dt: {dt}")
        print(f"dt_out: {dt_out}")
        print("--------------------")

        # P prune
        print(f"mesh.shape in: {mesh.shape}")
        print(f"pressures.shape in: {pressures.shape}")

        N_srcs = pressures.shape[0]
        N_t = pressures.shape[1]
        N_p = pressures.shape[2]
        # N_up = upressures.shape[1]

        if src_density_fact < 1:
            indices_srcs = np.random.choice(
                N_srcs,
                size=np.round(N_srcs * src_density_fact).astype(int),
                replace=False,
            )
            indices_srcs = np.sort(indices_srcs)
        else:
            indices_srcs = np.arange(N_srcs)
        if p_factor < 1 and len(indices_p) == 0:
            indices_p = np.random.choice(
                N_p, size=np.round(N_p * p_factor**dim).astype(int), replace=False
            )
            indices_p = np.sort(indices_p)
        elif len(indices_p) == 0:
            indices_p = np.arange(N_p)

        if t_factor < 1 and len(indices_t) == 0:
            # we prune to ensure the same dt (we don't need interpolation in time)
            indices_t = np.arange(N_t)
            tprune = int(np.round(1 / t_factor))
            indices_t = indices_t[::tprune]
            # indices_t = np.random.choice(N_t, size=np.round(N_t*t_factor).astype(int), replace=False)
            # indices_t = np.sort(indices_t)
            # indices_t[0] = 0 # include start time
            # indices_t[-1] = N_t-1 # include end time
        elif len(indices_t) == 0:
            indices_t = np.arange(N_t)

        # sub-sample pressures
        mesh = mesh[indices_p, :]
        pressures = pressures[indices_srcs, :, :]
        pressures = pressures[:, indices_t, :]
        pressures = pressures[:, :, indices_p]

        print(f"mesh.shape out: {mesh.shape}")
        print(f"pressures.shape out: {pressures.shape}")

        # sub-sample time
        print(f"t.shape in: {time_steps.shape}")
        time_steps = time_steps[indices_t]
        dt = time_steps[2] - time_steps[1]
        tmax = time_steps[-1]
        print(f"t.shape out: {time_steps.shape}")

        # sub-sample upressures (IC)
        print(f"upressures.shape in: {upressures.shape}")
        print(f"umesh.shape in: {umesh[()].shape}")

        umesh0 = umesh[:, 0].reshape(ushape[0], ushape[1])[
            0::uprune_factor, 0::uprune_factor
        ]
        umesh1 = umesh[:, 1].reshape(ushape[0], ushape[1])[
            0::uprune_factor, 0::uprune_factor
        ]
        umesh = np.hstack((umesh0.reshape(-1, 1), umesh1.reshape(-1, 1)))
        upressures = upressures.reshape(N_srcs, ushape[0], ushape[1])
        upressures = upressures[:, 0::uprune_factor, 0::uprune_factor]
        ushape = upressures.shape[1::]
        upressures = upressures.reshape(N_srcs, -1)
        upressures = upressures[indices_srcs, :]  # prune source pos density

        source_position = source_position[indices_srcs, :]

        print(f"umesh.shape out: {umesh.shape}")
        print(f"upressures.shape in: {upressures.shape}")
        print(f"upressures.shape out: {upressures.shape}")
        print(f"ushape out: {ushape}")

        pressures.shape[2] == mesh.shape[0]
        pressures.shape[0] == upressures.shape[0]
        pressures.shape[1] == upressures.shape[1]

        with h5py.File(path_data_out, "w") as fw:
            mesh_new = fw.create_dataset("mesh", data=mesh, dtype=dtype)
            mesh_new.attrs.create("domain_minmax", minmax, dtype=dtype)

            umesh_new = fw.create_dataset("umesh", data=umesh, dtype=dtype)
            umesh_new.attrs.create("umesh_shape", ushape, dtype=int)

            pressures_new = fw.create_dataset("pressures", data=pressures, dtype=dtype)
            upressures_new = fw.create_dataset(
                "upressures", data=upressures, dtype=dtype
            )

            fw.create_dataset("t", data=time_steps, dtype=dtype)
            fw.create_dataset("x0_srcs", data=source_position, dtype=dtype)

            # pressure attributes
            upressures_new.attrs.create("dx", [dx_u], dtype=np.float32)

            pressures_new.attrs.create("dt", [dt_out], dtype=np.float32)
            pressures_new.attrs.create("dx", [dx_p_out], dtype=np.float32)
            pressures_new.attrs.create("c", f[tag_attr].attrs["c"], dtype=np.float32)
            pressures_new.attrs.create(
                "c_phys", f[tag_attr].attrs["c_phys"], dtype=np.float32
            )
            pressures_new.attrs.create(
                "rho", f[tag_attr].attrs["rho"], dtype=np.float32
            )
            pressures_new.attrs.create("tmax", [tmax], dtype=np.float32)
            pressures_new.attrs.create(
                "sigma0", f[tag_attr].attrs["sigma0"], dtype=np.float32
            )
            pressures_new.attrs.create(
                "fmax", f[tag_attr].attrs["fmax"], dtype=np.float32
            )

    print(f"File saved to: {path_data_out}")

    return time_steps

if __name__ == "__main__":
    """
    Example usage:
    uprune_factor = 1  # prune factor
    p_ppw = 6  # out: ppw for spatial dimension
    t_ppw = 2  # out: ppw for temporal dimension
    src_ppw = 5  # out: ppw for source position density
    src_ppw_in = 5  # in: ppw for uniform grid
    src_density_fact = src_ppw / src_ppw_in
    u_ppw_in = 2  # in: ppw for uniform grid (only used for filename)
    base_path = "/work3/nibor/1TB/deeponet/input_1D_2D/"
    filepath_in = "rect2x2_freq_indep_ppw265_train_orig.h5"    
    """
    parser = argparse.ArgumentParser(description='Downsample 2D H5 data by reducing points per wavelength')
    parser.add_argument('--base_path', required=True, help='Base path to data')
    parser.add_argument('--input_file', required=True, help='Input H5 file path with high resolution data')
    parser.add_argument('--uprune_factor', type=int, default=1, help='Factor to downsample uniform grid (default: 1)')
    parser.add_argument('--p_ppw', type=int, default=6, help='Target points per wavelength for spatial dimension (default: 6)')
    parser.add_argument('--t_ppw', type=int, default=2, help='Target points per wavelength for temporal dimension (default: 2)')    
    parser.add_argument('--dtype', default='float32', choices=['float16', 'float32'], help='Output data type (default: float32)')
    
    args = parser.parse_args()

    src_density_fact = (
        args.src_ppw / args.src_ppw_in
    )  # TODO: newer data includes the dx_src, so we can use src_ppw as argument instead of this.

    input_file = f"{args.base_path}/{args.input_file}"
    output_file = f"{args.base_path}/rect2x2_freq_indep_ppw_{int(args.u_ppw_in / args.uprune_factor)}_{args.p_ppw}_{args.t_ppw}_{args.src_ppw}_train.h5"

    time_steps = downsample_2d_ppw(
        input_file,
        output_file,
        uprune_factor=args.uprune_factor,
        p_ppw=args.p_ppw,
        t_ppw=args.t_ppw,
        src_density_fact=args.src_density_fact,
        dtype=np.float16,
    )

    print(f"Output file: {args.output_file}")
    print(f"Time steps: {len(time_steps)} steps")
    print("Downsampling completed successfully!")