# ==============================================================================
# Copyright 2025 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import h5py
import numpy as np
from matplotlib import pyplot as plt
# import matplotlib.tri as mtri

dir_out = "/work3/nibor/data/deeponet/output_1D_2D/"

tag_attr = 'mesh'
tag_attr = 'pressures'

path_data_train = "/work3/nibor/1TB/deeponet/input_1D_2D/rect_freq_indep_ppw_2_6_2_6_train.h5"
path_data_val = "/work3/nibor/1TB/deeponet/input_1D_2D/rect_freq_indep_ppw_2_4_2_from_ppw_dx5_srcs5_val.h5"
# path_data_val = "/work3/nibor/1TB/deeponet/input_1D_2D/rect_freq_indep_ppw_2_4_2_from_ppw_dx5_srcs5_val.h5"

figscatter = plt.figure(figsize=(10,10))
ax_scatter = figscatter.add_subplot(111)

fig3d = plt.figure(figsize=(10,10))
ax3d = fig3d.add_subplot(111, projection='3d')

with h5py.File(path_data_train, 'r') as f:        
	time_steps = f['t'][()]
	dt = f[tag_attr].attrs['dt'][0]        

	umesh = f['mesh']
	mesh = f['mesh']
	pressures = f['pressures']
	upressures = f['upressures']
	source_position = f['x0_srcs'][()]
	N_srcs = pressures.shape[0]
	N_t = pressures.shape[1]
	N_p = pressures.shape[2]

	print(f"--------------------")
	print(f"train dt: {dt}")
	print(f"train time_step dt: {time_steps[100]-time_steps[99]}")
	print(f"N_srcs: {N_srcs}")
	print(f"N_t: {N_t}")
	print(f"N_p: {N_p}")
	print(f"src pos: {source_position}")
	print(f"--------------------")

	# for i,_ in enumerate(upressures.shape[0]):
	#         triang = mtri.Triangulation(umesh[:,0], umesh[:,1])
	#         ax.plot_trisurf(triang, upressures[i,:], cmap='viridis')

	ax_scatter.scatter(mesh[:,0], mesh[:,1])

tag_attr = 'pressures'
with h5py.File(path_data_val, 'r') as f:        
	time_steps = f['t'][()]
	dt = f[tag_attr].attrs['dt'][0]        

	umesh = f['umesh']
	mesh = f['mesh']
	pressures = f['pressures']
	upressures = f['upressures']
	source_position = f['x0_srcs'][()]
	N_srcs = pressures.shape[0]
	N_t = pressures.shape[1]
	N_p = pressures.shape[2]

	print(f"--------------------")
	print(f"val dt: {dt}")
	print(f"val time_step dt: {time_steps[100]-time_steps[99]}")
	print(f"N_srcs: {N_srcs}")
	print(f"N_t: {N_t}")
	print(f"N_p: {N_p}")
	# print(f"train time_steps: {time_steps}")
	print(f"src pos: {source_position}")
	print(f"--------------------")

	print(umesh.shape)
	print(upressures.shape)
	#for i in range(upressures.shape[0]):
	i = 2
	# triang = mtri.Triangulation(umesh[:,0], umesh[:,1])
	ax3d.plot_trisurf(umesh[:,0], umesh[:,1], upressures[i,:])

	ax_scatter.scatter(mesh[:,0], mesh[:,1])

ax_scatter.legend(['Training', 'Refererence'])
figscatter.savefig(dir_out + "train_val_mesh.png",bbox_inches='tight',pad_inches=0)
fig3d.savefig(dir_out + "srcs_mesh.png",bbox_inches='tight',pad_inches=0)
