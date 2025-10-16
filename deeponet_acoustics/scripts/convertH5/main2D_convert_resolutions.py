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
import deeponet_acoustics.scripts.convertH5.convert2D_resolutions as convertH5
import numpy as np

uprune_factor = 1 # prune factor
p_ppw = 6 # out: ppw for spatial dimension
t_ppw = 2 # out: ppw for temporal dimension

src_ppw = 5 # out: ppw for source position density
src_ppw_in = 5 # in: ppw for uniform grid
src_density_fact = src_ppw/src_ppw_in # TODO: newer data includes the dx_src, so we can use src_ppw as argument instead of this.

u_ppw_in = 2 # in: ppw for uniform grid (only used for filename)

filepath_in = "/work3/nibor/1TB/deeponet/input_1D_2D/rect2x2_freq_indep_ppw265_train_orig.h5"
filepath_out = f"/work3/nibor/1TB/deeponet/input_1D_2D/rect2x2_freq_indep_ppw_{int(u_ppw_in/uprune_factor)}_{p_ppw}_{t_ppw}_{src_ppw}_train.h5"

time_steps_train = convertH5.prunePPW2DH5(filepath_in, filepath_out, 
                                          uprune_factor=uprune_factor, p_ppw=p_ppw, t_ppw=t_ppw, src_density_fact=src_density_fact, 
                                          dtype=np.float16)

print(f'Filename train out: {filepath_out}')

assert p_ppw != 4, "ensure validation resolution differs from training data"
p_ppw = 4

filepath_in = "/work3/nibor/1TB/deeponet/input_1D_2D/Lshape2_5x2_5_freq_indep_ppw2410_30srcpos_val_orig.h5"
filepath_out = f"/work3/nibor/1TB/deeponet/input_1D_2D/Lshape2_5x2_5_freq_indep_ppw_{int(u_ppw_in/uprune_factor)}_{p_ppw}_{t_ppw}_30srcpos_val.h5"
time_steps_val = convertH5.prunePPW2DH5(filepath_in, filepath_out, 
                                        uprune_factor=uprune_factor, p_ppw=p_ppw, t_ppw=t_ppw, 
                                        dtype=np.float16)

print(f'Filename val out: {filepath_out}')

print(f"time_steps_train: {time_steps_train}")
print(f"time_steps_val: {time_steps_val}")
# assert (time_steps_train == time_steps_val).all(), "train and validation time steps differ"
