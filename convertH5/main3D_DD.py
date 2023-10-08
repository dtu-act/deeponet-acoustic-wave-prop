# ==============================================================================
# Copyright 2023 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
#
# 3D: used for converting the full dome to quarter dome for domain decomposition
# ==============================================================================
import convertH5.convert2D_resolutions as convertH5
import datahandlers.io as IO
from pathlib import Path
import shutil
import os
import numpy as np

input_path  = "/work3/nibor/1TB/libP/bilbao_1000hz_p4_5ppw_srcpos5_val/"
output_path = "/work3/nibor/1TB/libP/bilbao_1000hz_p4_5ppw_srcpos5_1stquad_val/"

if Path(output_path).exists():
    shutil.rmtree(output_path)
Path(output_path).mkdir(parents=True, exist_ok=False)

# copy settings file
source_settings_path = os.path.join(input_path, 'simulation_parameters.json')
if Path(source_settings_path).exists():
    shutil.copy(source_settings_path, os.path.join(output_path, 'simulation_parameters.json'))

filenamesH5 = IO.pathsToFileType(input_path, '.h5')

N = len(filenamesH5)
print(f"Number of folders to process: {len(filenamesH5)}\n")

extract_domain = [[0.0,3.0],[0.0,3.0],[0.0,4.0]]

for i,path_to_filename in enumerate(filenamesH5):
    print(f"Processing {i}/{N}\n")
    print(f"Input: {path_to_filename}\n")

    last_directory = os.path.basename(os.path.dirname(path_to_filename))
    new_base_dir = os.path.join(output_path, last_directory)

    # create folder for the sample
    Path(new_base_dir).mkdir(parents=True, exist_ok=True)

    path_to_new_filename = os.path.join(new_base_dir, os.path.basename(path_to_filename))

    convertH5.splitDomain(path_to_filename, path_to_new_filename, extract_domain, dtype=np.float16)
    print(f'written file: {path_to_new_filename}\n')    