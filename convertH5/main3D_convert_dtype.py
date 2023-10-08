# ==============================================================================
# Copyright 2023 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import convertH5.convert2D_resolutions as convertH5
import datahandlers.io as IO
import pathlib
import os
import numpy as np

# 2D: converting from high spatial/temporal resolutions to lower resolutions (data generated with MATLAB)

base_path = "/work3/nibor/1TB/libP/bilbao_1000hz_p4_5ppw_srcpos5_val"

filenamesH5 = IO.pathsToFileType(base_path, '.h5')

N = len(filenamesH5)
print(f"Number of folders to process: {len(filenamesH5)}\n")

for i,filename in enumerate(filenamesH5):
    print(f"Processing {i}/{N}\n")
    print(f"Input: {filename}\n")

    filename_out = pathlib.Path(filename).with_suffix('')
    filename_out = str(filename_out) + ('_float16.h5')    
    convertH5.convertToDtypeCompactH5(filename, filename_out, dtype=np.float16)
    print(f'written file: {filename_out}\n')
    os.remove(filename)
    print(f'deleted file: {filename}\n')