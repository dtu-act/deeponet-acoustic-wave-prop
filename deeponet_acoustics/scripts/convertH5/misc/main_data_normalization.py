# ==============================================================================
# Copyright 2025 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import deeponet_acoustics.datahandlers.data_rw as rw
from deeponet_acoustics.setup.settings import SimulationSettings
import deeponet_acoustics.setup.parsers as parsers
import deeponet_acoustics.datahandlers.io as IO
from deeponet_acoustics.datahandlers.io import XdmfReader

id = "deeponet_3D_bn5_512_tn5_512_out100_bs64_coord500_cnn_2xch64_fnn80_80_ampl2"

output_dir = "/users/nborrelj/data/nborrelj/deeponet/output"
settings_path = os.path.join(output_dir, f"{id}/settings.json")

settings_dict = parsers.parseSettings(settings_path)
settings = SimulationSettings(settings_dict, output_dir=output_dir)

training = settings.training_settings

tmax = settings.tmax

sim_params_path = os.path.join(settings.dirs.training_data_path, "simulation_parameters.json")
phys_params = rw.loadSimulationParametersJson(sim_params_path)
c_phys = phys_params.c_phys

### Initialize model ###
filenames_xdmf = IO.pathsToXdmf(settings.dirs.training_data_path)

xdmf = XdmfReader(filenames_xdmf[0], tmax=tmax/c_phys)
tags_field = xdmf.tags_field

num_tsteps = len(xdmf.tsteps)

datasets = []
for filename in filenames_xdmf[0:100]:
    xdmf = XdmfReader(filename, tmax=tmax/c_phys) 
    datasets.append(h5py.File(xdmf.filenameH5, 'r')) # add file handles and keeps open

p_means = []
p_max = -float('inf')
p_min = float('inf')

fig0 = plt.figure()
ax0 = fig0.add_subplot(111)

for i, dataset in enumerate(datasets):
    for j in range(num_tsteps):
        data = dataset[tags_field[j]][:]
        p_means.append(np.mean(data))
        p_max = max(max(data),p_max)
        p_min = min(min(data),p_min)
        # if (i == 0):
        #     ax0.hist(data, bins=50, color = "skyblue")
        #     plt.savefig(output_dir+f"/histograms/hist_all.png",bbox_inches='tight',pad_inches=0)
            
        #     fig1 = plt.figure()
        #     ax1 = fig1.add_subplot(111)
        #     ax1.hist(data, bins=20)
        #     plt.savefig(output_dir+f"/histograms/hist_{j}.png",bbox_inches='tight',pad_inches=0)
        #     plt.close()

p_tot_mean = np.mean(p_means)

print(f"p_min={p_min}, p_max={p_max}, p_tot_mean={p_tot_mean}")

fig0 = plt.figure()
ax0 = fig0.add_subplot(111)

#for i, dataset in enumerate(datasets):
for j in range(num_tsteps):
    data = datasets[0][tags_field[j]][:]
    data_norm = (data - p_min)/(p_max-p_min)
    data_std = (data - p_tot_mean)/(p_max-p_min)
    
    # ax0.hist(data_norm, bins=50, color = "skyblue")
    # plt.savefig(output_dir+f"/histograms/histnorm_all.png",bbox_inches='tight',pad_inches=0)
    
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # ax1.hist(data_norm, bins=20)
    # plt.savefig(output_dir+f"/histograms/histnorm_{j}.png",bbox_inches='tight',pad_inches=0)
    # plt.close()
