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
import numpy
from models.datastructures import NetworkArchitectureType, TransferLearning

from models.networks_flax import ResNet, setupFNN
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

mod_fnn = True
do_animate = False
tmax = 16.9

settings_path = os.path.join(output_path, f"{id}/settings.json")

settings_dict = parsers.parseSettings(settings_path)
settings = SimulationSettings(settings_dict)
### uncomment this line if custom testing data should be used ###
testing_data_path = settings.dirs.testing_data_path

do_fnn = settings.branch_net.architecture == NetworkArchitectureType.MLP
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
in_tn = y_test.shape[1],
tn_fnn = setupFNN(trunk_net, "tn", mod_fnn=mod_fnn)
print(tn_fnn.tabulate(jax.random.PRNGKey(1234), numpy.expand_dims(jnp.ones(in_tn), [0])))

if do_fnn:    
    in_bn = u_test.shape[1],
    bn_fnn = setupFNN(branch_net, "bn", mod_fnn=mod_fnn)
    print(bn_fnn.tabulate(jax.random.PRNGKey(1234), numpy.expand_dims(jnp.ones(in_bn), [0])))
else:
    num_blocks : tuple = (3, 3, 3, 3)
    c_hidden : tuple = (16, 32, 64, 128)
    in_bn = u_test.shape[1:] 
    branch_layers = 0*[branch_net.num_hidden_neurons] + [branch_net.num_output_neurons]
    bn_fnn = ResNet(layers_fnn=branch_layers, num_blocks=num_blocks, c_hidden=c_hidden, act_fn=jax.nn.relu)
    print(bn_fnn.tabulate(jax.random.PRNGKey(1234), numpy.expand_dims(jnp.ones(in_bn), [0,3])))

lr = settings.training_settings.learning_rate
decay_steps=settings.training_settings.decay_steps
decay_rate=settings.training_settings.decay_rate
transfer_learning = TransferLearning({'transfer_learning': {'resume_learning': True}}, settings.dirs.models_dir)
model = DeepONet(lr, bn_fnn, in_bn, tn_fnn, in_tn, settings.dirs.models_dir, 
                 decay_steps, decay_rate, do_fnn=do_fnn, 
                 transfer_learning=transfer_learning)

model.plotLosses(settings.dirs.figs_dir)

tdim = t_test.shape[0]
S_pred_srcs = numpy.empty((x0_srcs.shape[0],tdim,mesh_test.shape[0]), dtype=float)
S_test_srcs = numpy.empty((x0_srcs.shape[0],tdim,mesh_test.shape[0]), dtype=float)

figs_dir = settings.dirs.figs_dir
path_receivers = os.path.join(figs_dir, "receivers")
Path(path_receivers).mkdir(parents=True, exist_ok=True)

for i_src in range(x0_srcs.shape[0]):
    x0 = x0_srcs[i_src]    

    u_test_i = u_test[i_src,:]
    s_test_i = s_test[i_src,:]

    # Predict
    s_pred_i = model.predict_s(model.params, u_test_i, y_test)

    S_pred_srcs[i_src,:,:] = numpy.array(s_pred_i).reshape(tdim,-1)
    S_test_srcs[i_src,:,:] = numpy.array(s_test_i).reshape(tdim,-1)

    # if isTwoD:
    #     IO.writeTriangleXdmf(grid_test, conn_test-1, t_test, S_pred_srcs[i_src], os.path.join(path_receivers, f"wavefield_pred{x0}.xdmf"))
    #     IO.writeTriangleXdmf(grid_test, conn_test-1, t_test, S_test_srcs[i_src], os.path.join(path_receivers, f"wavefield_test{x0}.xdmf"))

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

plotting.plotAtReceiverPosition(x0_srcs,r0_list, r0_indxs,t_test/c_phys,
    S_pred_srcs,S_test_srcs,tmax/c_phys,figs_dir=path_receivers,animate=do_animate)
if data_test.dim == 1:
    plotting.plotWaveFields1D(grid1d_test,t1d_test,S_pred_srcs,S_test_srcs,x0_srcs,tmax,c_phys,figs_dir=path_receivers)