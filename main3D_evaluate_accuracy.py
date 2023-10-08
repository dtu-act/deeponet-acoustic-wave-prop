# ==============================================================================
# Copyright 2023 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import os
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from models.datastructures import NetworkArchitectureType, TransferLearning
import utils.utils as utils

from datahandlers.datagenerators import DataH5Compact, DatasetStreamer
from models.deeponet import DeepONet
from models.networks_flax import modified_MLP
from utils.feat_expansion import fourierFeatureExpansion_f0
import datahandlers.data_rw as rw
import plotting.visualizing as plotting
from setup.configurations import setupPlotParams
from setup.settings import SimulationSettings
import setup.parsers as parsers
import datahandlers.io as IO

output_dir = "/work3/nibor/data/deeponet/output"
input_dir = "/work3/nibor/1TB/libP"
do_animate = False

prune_spatial = 1

## CUBE 2x2x2
# id = "cube_6ppw"
# testing_data_path = "/work3/nibor/1TB/libP/cube_1000hz_p4_5ppw_srcpos5_val" # overwrite validation data from settings file

## BILBAO ROOM
# FULL
id = "bilbao_6ppw" 
testing_data_path = "/work3/nibor/1TB/libP/bilbao_1000hz_p4_5ppw_srcpos5_val"

# 1st QUADRANT
# id = "bilbao_6ppw_1stquad" 
# testing_data_path = "/work3/nibor/1TB/libP/bilbao_1000hz_p4_5ppw_srcpos5_1stquad_val/"

recv_pos = np.array([
    [0.8, 1.5, 0.75],
    [0.8, 0.6, 0.75],
    [2.4, 0.4, 0.75],
    [0.8, 2.0, 0.75],
    [2.2, 0.4, 0.75]
])

## RECEIVERS FURNISHED ROOM
# id = "furnished_6ppw"
# testing_data_path = "/work3/nibor/1TB/libP/furnished_1000hz_p4_5ppw_srcpos5_val"
# recv_pos = np.array([
#     [2.25, 0.8, 1.2],
#     [1.8,  0.8, 1.2],
#     [1.5,  2.4, 1.2],
#     [1.2,  0.8, 1.2],    
#     [0.75, 0.8, 1.2]
# ])

## RECEIVERS LSHAPED
# id = "Lshape_6ppw"
# testing_data_path = "/work3/nibor/1TB/libP/Lshape_1000hz_p4_5ppw_srcpos5_val"
# recv_pos = np.array([
#     [1.0, 2.00, 1.0],
#     [1.0, 1.75, 1.0],
#     [1.0, 1.50, 1.0],
#     [1.0, 1.75, 1.0],
#     [1.0, 2.00, 1.0]
# ])

settings_path = os.path.join(output_dir, f"{id}/settings.json")

settings_dict = parsers.parseSettings(settings_path)
settings = SimulationSettings(settings_dict, input_dir=input_dir, output_dir=output_dir)
settings.dirs.createDirs()

do_fnn = settings.branch_net.architecture == NetworkArchitectureType.MLP
training = settings.training_settings
branch_net = settings.branch_net
trunk_net = settings.trunk_net

tmax = settings.tmax

sim_params_path = os.path.join(settings.dirs.training_data_path, "simulation_parameters.json")
phys_params = rw.loadSimulationParametersJson(sim_params_path)
c_phys = phys_params.c_phys

### Initialize model ###
f = settings.f0_feat
y_feat = fourierFeatureExpansion_f0(f)

metadata_compare = DataH5Compact(settings.dirs.training_data_path, tmax=tmax, t_norm=c_phys, 
    flatten_ic=do_fnn, data_prune=prune_spatial, norm_data=settings.normalize_data)
metadata = DataH5Compact(testing_data_path, tmax=tmax, t_norm=c_phys, 
    flatten_ic=do_fnn, data_prune=prune_spatial, norm_data=settings.normalize_data)
dataset = DatasetStreamer(metadata, y_feat_extractor=y_feat)

if not np.allclose(metadata.tsteps, metadata_compare.tsteps):
    raise Exception(f"Time steps differs between training and validation data: \nN_train={len(metadata.tsteps)}, N_val={len(metadata_compare.tsteps)}, dt_train={metadata.tsteps[1]-metadata.tsteps[0]} and dt_val={metadata_compare.tsteps[1]-metadata_compare.tsteps[0]}.\n The network is not supposed to learn temporal interpolation. Exiting.")

# setup network
in_bn = metadata.u_shape[0]
in_tn = y_feat(np.array([[0.0,0.0,0.0,0.0]])).shape[1]
branch_layers = branch_net.num_hidden_layers*[branch_net.num_hidden_neurons] + [branch_net.num_output_neurons]
trunk_layers  = trunk_net.num_hidden_layers*[trunk_net.num_hidden_neurons]  + [trunk_net.num_output_neurons]

bn_fnn = modified_MLP(layers=branch_layers, tag="bn")
tn_fnn = modified_MLP(layers=trunk_layers, tag="tn")
print(bn_fnn.tabulate(jax.random.PRNGKey(1234), jnp.ones((1, in_bn))))
print(tn_fnn.tabulate(jax.random.PRNGKey(1234), jnp.ones((1, in_tn))))    

lr = settings.training_settings.learning_rate
decay_steps=settings.training_settings.decay_steps
decay_rate=settings.training_settings.decay_rate
transfer_learning = TransferLearning({'transfer_learning': {'resume_learning': True}}, 
                                     settings.dirs.models_dir)
model = DeepONet(lr, bn_fnn, in_bn, tn_fnn, in_tn, settings.dirs.models_dir, 
                 decay_steps, decay_rate, do_fnn=do_fnn, 
                 transfer_learning=transfer_learning)

params = model.params

model.plotLosses(settings.dirs.figs_dir)

## Plot solution using test data ###
# sample a subset of the source positions
# num_srcs = min(dataset.N,5)
# rng = np.random.default_rng()
# indxs_src = rng.integers(0, dataset.N, size=num_srcs)

num_srcs = dataset.N
indxs_src = range(0,num_srcs)
#indxs_src = range(4,5)

tdim = metadata.num_tsteps
S_pred_srcs = np.empty((num_srcs,tdim,dataset.Pmesh), dtype=float)
S_test_srcs = np.empty((num_srcs,tdim,dataset.Pmesh), dtype=float)

path_receivers = os.path.join(settings.dirs.figs_dir , "receivers")
Path(path_receivers).mkdir(parents=True, exist_ok=True)

xxyyzztt = metadata.xxyyzztt
y_in = y_feat(xxyyzztt)

xxyyzz_phys = metadata.denormalizeSpatial(xxyyzztt[:,0:3])
mesh_phys = metadata.denormalizeSpatial(metadata.mesh)
tsteps_phys = metadata.denormalizeTemporal(metadata.tsteps/c_phys)

x0_srcs = []
for i_src,indx_src in enumerate(indxs_src):
    (u_test_i,*_),s_test_i,_,x0 = dataset[indx_src]
    x0_denorm = metadata.denormalizeSpatial(x0)

    s_pred_i = np.zeros(s_test_i.reshape(tdim,-1).shape)
    # Predict
    for i,_ in enumerate(metadata.tsteps):
        yi = y_in.reshape(tdim,s_pred_i.shape[1],-1)[i,:]
        s_pred_i[i,:] = model.predict_s(params, u_test_i, yi)

    S_pred_srcs[i_src,:,:] = np.asarray(s_pred_i)
    S_test_srcs[i_src,:,:] = np.asarray(s_test_i).reshape(tdim,-1)

    x0 = metadata.denormalizeSpatial(x0)
    x0_srcs.append(x0)

    IO.writeTetraXdmf(mesh_phys, metadata.conn,
                 tsteps_phys, S_pred_srcs[i_src], 
                 os.path.join(path_receivers, f"wavefield_pred{x0}.xdmf"))
    IO.writeTetraXdmf(mesh_phys, metadata.conn,
                 tsteps_phys, S_test_srcs[i_src], 
                 os.path.join(path_receivers, f"wavefield_ref{x0}.xdmf"))

setupPlotParams(True)

r0_list, r0_indxs = utils.getNearestFromCoordinates(xxyyzz_phys, recv_pos)
#r0_list, r0_indxs = utils.calcReceiverPositionsSimpleDomain(xxyyzz_phys, x0_srcs)

plotting.plotAtReceiverPosition(x0_srcs,r0_list, r0_indxs,
    tsteps_phys,S_pred_srcs,S_test_srcs,tmax/c_phys,figs_dir=path_receivers,animate=do_animate)