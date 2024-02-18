# ==============================================================================
# Copyright 2024 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import os, shutil
import jax.numpy as jnp
import numpy
from models.datastructures import NetworkArchitectureType

from models.networks_flax import setupNetwork
from datahandlers.datagenerators import DataGenerator, normalizeDomain, normalizeData, normalizeFourierDataExpansionZero
from setup.data import setupData, setupTransferLearningData
from models.deeponet import DeepONet
import utils.feat_expansion as featexp
import datahandlers.data_rw as rw
from setup.settings import SimulationSettings
import setup.parsers as parsers

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

def train(settings_path):
    settings_dict = parsers.parseSettings(settings_path)
    settings = SimulationSettings(settings_dict)
    if settings.transfer_learning == None or not settings.transfer_learning.resume_learning:
        settings.dirs.createDirs(delete_existing=True)
        
    shutil.copyfile(settings_path, os.path.join(settings.dirs.id_dir, 'settings.json')) # copy settings

    training = settings.training_settings
    branch_net = settings.branch_net
    trunk_net = settings.trunk_net

    tmax = settings.tmax
    nIter = training.iterations

    # load training data
    data = rw.loadDataFromH5(settings.dirs.training_data_path, tmax=tmax)
    data_val = rw.loadDataFromH5(settings.dirs.testing_data_path, tmax=tmax)    
    simulation_settings = rw.loadAttrFromH5(settings.dirs.training_data_path)

    if not numpy.allclose(data.t, data_val.t):
        raise Exception(f"Time steps differs between training and validation data: \nN_train={len(data.t)}, N_val={len(data_val.t)}, dt_train={data.t[1]-data.t[0]} and dt_val={data_val.t[1]-data_val.t[0]}.\n The network is not supposed to learn temporal interpolation. Exiting.")
    
    flatten_data = branch_net.architecture != NetworkArchitectureType.RESNET
    u_train,s_train,t1d,grid1d = setupData(data.mesh,data.pressures,data.upressures,data.t,data.ushape,flatten_data)
    u_val,s_val,t1d_val,grid1d_val = setupData(data_val.mesh,data_val.pressures,data_val.upressures,data_val.t,data_val.ushape,flatten_data)

    y_train = jnp.hstack([grid1d, t1d])
    y_val = jnp.hstack([grid1d_val, t1d_val])
    data_nonfeat_dim = y_train.shape[1]

    if settings.normalize_data:
        from_zero = settings.trunk_net.activation == "relu"
        domain_minmax = simulation_settings['domain_minmax']
        if data.dim == 1:
            domain_min, domain_max = domain_minmax[0], domain_minmax[1]
        else: # 2D
            domain_min, domain_max = min(domain_minmax[:,0]), max(domain_minmax[:,1])    
        y_train = normalizeDomain(y_train, domain_min, domain_max, from_zero=from_zero)
        y_val = normalizeDomain(y_val, domain_min, domain_max, from_zero=from_zero)

    y_feat = featexp.fourierFeatureExpansion_f0(settings.f0_feat)
    # y_feat = featexp.fourierFeatureExpansion_gaussian((10,3), mean=fmax/2, std_dev=fmax/2)
    # domain_minmax_norm = normalizeData(domain_minmax, domain_min, domain_max, from_zero=from_zero)
    # L_dom = domain_minmax_norm[:,1] - domain_minmax_norm[:,0]
    # y_feat = featexp.fourierFeatureExpansion_exact_sol([fmax, fmax/2, fmax/4], c, L_dom[0], L_dom[1]) # only defined in 2D

    y_train = y_feat(y_train)
    y_val = y_feat(y_val)

    if settings.normalize_data and from_zero:
        # only used for relu activation function normalizing cos/sin domain from [-1,1] to [0,1]
        y_train = normalizeFourierDataExpansionZero(y_train, data_nonfeat_dim=data_nonfeat_dim)
        y_val = normalizeFourierDataExpansionZero(y_val, data_nonfeat_dim=data_nonfeat_dim)

    # setup network
    in_tn = y_train.shape[1]
    trunk_nn = setupNetwork(trunk_net, in_tn, 'tn')
    branch_nn = setupNetwork(branch_net, u_train.shape[1], 'bn')

    lr = settings.training_settings.learning_rate    
    bs = settings.training_settings.batch_size_branch * settings.training_settings.batch_size_coord,
    adaptive_weights_shape = bs if settings.training_settings.use_adaptive_weights else []
                     
    model = DeepONet(lr, branch_nn, trunk_nn, 
                     settings.dirs.models_dir,
                     decay_steps=settings.training_settings.decay_steps,
                     decay_rate=settings.training_settings.decay_rate,
                     transfer_learning=settings.transfer_learning,
                     adaptive_weights_shape=adaptive_weights_shape)

    if settings.transfer_learning == None:
        dataset = DataGenerator(u_train, y_train, s_train, training.batch_size_branch, training.batch_size_coord)
        dataset_val = DataGenerator(u_val, y_val, s_val, training.batch_size_branch, training.batch_size_coord) 
    else:
        u_src_train, u_src_val = setupTransferLearningData(settings.transfer_learning, flatten_data=branch_nn.network_type != NetworkArchitectureType.RESNET)
        dataset = DataGenerator(u_train, y_train, s_train, training.batch_size_branch, training.batch_size_coord, u_src_train)
        dataset_val = DataGenerator(u_val, y_val, s_val, training.batch_size_branch, training.batch_size_coord, u_src_val)
    
    ### Train ###
    model.trainFromMemory(dataset, dataset_val, nIter=nIter, save_every=100)
    model.plotLosses(settings.dirs.figs_dir)

# settings_path = "scripts/twoD/settings.json"
# train(settings_path)