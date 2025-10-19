# ==============================================================================
# Copyright 2025 Technical University of Denmark
# Author: Nikolas Borrel-Jensen
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import deeponet_acoustics.scripts.convertH5.assembly2D as convertH5

data_dir = (
    "/work3/nibor/1TB/deeponet/input_1D_2D/Lshape3x3_freq_indep_ppw_2_4_2_5srcpos_val"
)
header_filepath_in = "/work3/nibor/1TB/deeponet/input_1D_2D/Lshape3x3_freq_indep_ppw_2_4_2_5srcpos_val/Lshape3x3_freq_indep_ppw_2_4_2_5srcpos_val_header.h5"
filepath_out = "/work3/nibor/1TB/deeponet/input_1D_2D/Lshape3x3_freq_indep_ppw_2_4_2_5srcpos_val.h5"

convertH5.assembleH5(data_dir, header_filepath_in, filepath_out)
