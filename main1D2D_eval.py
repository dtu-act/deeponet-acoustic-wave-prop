# ==============================================================================
# Copyright 2023 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import os, shutil
import jax.numpy as jnp
import jax
import numpy as np
from models.datastructures import NetworkArchitectureType, TransferLearning

from models.networks_flax import setupNetwork
from datahandlers.datagenerators import normalizeDomain, normalizeData
from setup.data import setupData
from setup.configurations import setupPlotParams
import utils.utils as utils
from models.deeponet import DeepONet
import utils.feat_expansion as featexp
import datahandlers.data_rw as rw
import plotting.visualizing as plotting
from setup.settings import SimulationSettings
import setup.parsers as parsers
from pathlib import Path

id = "spectral_sine_1D"
input_dir = "/work3/nibor/data/input1D"
output_path = "/work3/nibor/data/deeponet/output1D"

### overwrite validation data from settings file ###
# testing_data_path = os.path.join(input_dir, "rect3x3_freq_indep_ppw_2_4_2_from_ppw_dx5_srcs33_val.h5")

do_animate = False
tmax = 16.9

settings_path = os.path.join(output_path, f"{id}/settings.json")

settings_dict = parsers.parseSettings(settings_path)
settings = SimulationSettings(settings_dict)
### uncomment this line if custom testing data should be used ###
testing_data_path = settings.dirs.testing_data_path

do_fnn = settings.branch_net.architecture != NetworkArchitectureType.RESNET
training = settings.training_settings
branch_net = settings.branch_net
trunk_net = settings.trunk_net

tmax = settings.tmax

# load testing data
data_test = rw.loadDataFromH5(testing_data_path, tmax=tmax)
simulation_settings = rw.loadAttrFromH5(settings.dirs.training_data_path)

c_phys = 343
c = simulation_settings['c']
fmax = simulation_settings['fmax']

prune_spatial = 2

conn_test = data_test.conn[::prune_spatial,:]
mesh_test = data_test.mesh[::prune_spatial,:]
t_test = data_test.t
p_test = data_test.pressures[:,:,::prune_spatial]
up_test = data_test.upressures
x0_srcs = data_test.x0_srcs
ushape_test = data_test.ushape # TODO for CNNs

u_test,s_test,t1d_test,grid1d_test = setupData(mesh_test,p_test,up_test,t_test,ushape_test,do_fnn)

y_test = jnp.hstack([grid1d_test, t1d_test])

if settings.normalize_data:
        from_zero = settings.trunk_net.activation == "relu"
        yminmax = simulation_settings['domain_minmax']
        if data_test.dim == 1:
            ymin, ymax = yminmax[0], yminmax[1]
        else: # 2D
            ymin, ymax = min(yminmax[:,0]), max(yminmax[:,1])    
        y_test = normalizeDomain(y_test, ymin, ymax, from_zero=from_zero)

y_feat = featexp.fourierFeatureExpansion_f0(settings.f0_feat)
#y_feat = featexp.fourierFeatureExpansion_gaussian((10,3), mean=fmax/2, std_dev=fmax/2)
#y_feat = featexp.fourierFeatureExpansion_exact_sol([fmax, fmax/2, fmax/6], c, 3.0, 3.0)

y_test = y_feat(y_test)

# setup network
in_tn = y_test.shape[1]
trunk_nn = setupNetwork(trunk_net, in_tn, 'tn')
in_bn = u_test.shape[1]
branch_nn = setupNetwork(branch_net, in_bn, 'bn')

lr = settings.training_settings.learning_rate    
bs = settings.training_settings.batch_size_branch * settings.training_settings.batch_size_coord,
adaptive_weights_shape = bs if settings.training_settings.use_adaptive_weights else []
transfer_learning = TransferLearning({'transfer_learning': {'resume_learning': True}}, settings.dirs.models_dir)

model = DeepONet(lr, branch_nn, trunk_nn, 
                 settings.dirs.models_dir,
                 decay_steps=settings.training_settings.decay_steps,
                 decay_rate=settings.training_settings.decay_rate,
                 transfer_learning=transfer_learning,
                 adaptive_weights_shape=adaptive_weights_shape)

model.plotLosses(settings.dirs.figs_dir)

tdim = t_test.shape[0]
S_pred_srcs = np.empty((x0_srcs.shape[0],tdim,mesh_test.shape[0]), dtype=float)
S_test_srcs = np.empty((x0_srcs.shape[0],tdim,mesh_test.shape[0]), dtype=float)

figs_dir = settings.dirs.figs_dir
path_receivers = os.path.join(figs_dir, "receivers")
Path(path_receivers).mkdir(parents=True, exist_ok=True)

for i_src in range(x0_srcs.shape[0]):
    x0 = x0_srcs[i_src]    

    u_test_i = u_test[i_src,:]
    s_test_i = s_test[i_src,:]

    # Predict
    s_pred_i = model.predict_s(model.params, u_test_i, y_test)

    S_pred_srcs[i_src,:,:] = np.array(s_pred_i).reshape(tdim,-1)
    S_test_srcs[i_src,:,:] = np.array(s_test_i).reshape(tdim,-1)

    # if data_test.dim == 2:
    #     IO.writeTriangleXdmf(grid_test, conn_test-1, t_test, S_pred_srcs[i_src], os.path.join(path_receivers, f"{i_src}_wavefield_pred{x0}.xdmf"))
    #     IO.writeTriangleXdmf(grid_test, conn_test-1, t_test, S_test_srcs[i_src], os.path.join(path_receivers, f"{i_src}_wavefield_test{x0}.xdmf"))

path_receivers = os.path.join(figs_dir, "receivers")
Path(path_receivers).mkdir(parents=True, exist_ok=True)

r0_list, r0_indxs = utils.calcReceiverPositionsSimpleDomain(grid1d_test, x0_srcs)

if settings.normalize_data:
    from_zero = settings.trunk_net.activation == "relu"
    yminmax = simulation_settings['domain_minmax']
    if data_test.dim == 1:
        ymin, ymax = yminmax[0], yminmax[1]
    else: # 2D
        ymin, ymax = min(yminmax[:,0]), max(yminmax[:,1])    
    r0_list = normalizeData(r0_list, ymin, ymax, from_zero=from_zero)

setupPlotParams(True)

N_srcs = len(r0_indxs)

ir_ref_srcs = np.empty(N_srcs, dtype=object)
ir_pred_srcs = np.empty(N_srcs, dtype=object)

for i_src in range(N_srcs):
    ir_ref_srcs[i_src] = np.expand_dims(S_test_srcs[i_src,:,r0_indxs[i_src]], 1)
    ir_pred_srcs[i_src] = np.expand_dims(S_pred_srcs[i_src,:,r0_indxs[i_src]], 1)

plotting.writeIRPlotsWithReference(x0_srcs,np.expand_dims(r0_list,1),t_test/c_phys,
    ir_pred_srcs,ir_ref_srcs,tmax/c_phys,
    figs_dir=path_receivers,animate=do_animate)

if data_test.dim == 1:
    plotting.plotWaveFields1D(grid1d_test,t1d_test,S_pred_srcs,S_test_srcs,x0_srcs,c_phys,path_receivers)