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
from models.datastructures import NetworkArchitectureType

import utils.utils as utils
import datahandlers.data_rw as rw
import setup.parsers as parsers
from datahandlers.datagenerators import DataH5Compact, DatasetStreamer, NumpyLoader
from models.networks_flax import ResNet, setupFNN
from models.deeponet import DeepONet
from utils.feat_expansion import fourierFeatureExpansion_f0
from setup.settings import SimulationSettings

def train(settings_path):
    mod_fnn_bn = True
    mod_fnn_tn = True

    settings_dict = parsers.parseSettings(settings_path)
    settings = SimulationSettings(settings_dict)
    if settings.transfer_learning == None or not settings.transfer_learning.resume_learning:
        settings.dirs.createDirs(delete_existing=True)
    
    shutil.copyfile(settings_path, os.path.join(settings.dirs.id_dir, 'settings.json')) # copy settings

    do_fnn = settings.branch_net.architecture == NetworkArchitectureType.MLP
    training = settings.training_settings
    branch_net = settings.branch_net
    trunk_net = settings.trunk_net

    tmax = settings.tmax
    nIter = training.iterations

    # load training data
    sim_params_path = os.path.join(settings.dirs.training_data_path, "simulation_parameters.json")
    phys_params = rw.loadSimulationParametersJson(sim_params_path)
    c_phys = phys_params.c_phys
    
    f = settings.f0_feat
    y_feat = fourierFeatureExpansion_f0(f)

    metadata = DataH5Compact(settings.dirs.training_data_path, tmax=tmax, t_norm=c_phys, 
        norm_data=settings.normalize_data)
    dataset = DatasetStreamer(metadata, training.batch_size_coord, y_feat_extractor=y_feat)
    metadata_val = DataH5Compact(settings.dirs.testing_data_path, tmax=tmax, t_norm=c_phys, 
        norm_data=settings.normalize_data)
    dataset_val = DatasetStreamer(metadata_val, training.batch_size_coord, y_feat_extractor=y_feat)
    
    dataloader = NumpyLoader(dataset, batch_size=training.batch_size_branch, shuffle=True, drop_last=True)
    dataloader_val = NumpyLoader(dataset_val, batch_size=training.batch_size_branch, shuffle=True, num_workers=0) # do not drop last, validation set has few samples

    if not np.allclose(metadata.tsteps, metadata_val.tsteps):
        raise Exception(f"Time steps differs between training and validation data: \nN_train={len(metadata.tsteps)}, N_val={len(metadata_val.tsteps)}, dt_train={metadata.tsteps[1]-metadata.tsteps[0]} and dt_val={metadata_val.tsteps[1]-metadata_val.tsteps[0]}.\n The network is not supposed to learn temporal interpolation. Exiting.")
    
    utils.printInfo(metadata, metadata_val, training.batch_size_coord, training.batch_size_branch)

    # setup network
    in_tn = y_feat(np.array([[0.0,0.0,0.0,0.0]])).shape[1]
    tn_fnn = setupFNN(trunk_net, "tn", mod_fnn=mod_fnn_tn)
    print(tn_fnn.tabulate(jax.random.PRNGKey(1234), np.expand_dims(jnp.ones(in_tn), [0])))

    if do_fnn:    
        in_bn = metadata.u_shape
        bn_fnn = setupFNN(branch_net, "bn", mod_fnn=mod_fnn_bn)
        print(bn_fnn.tabulate(jax.random.PRNGKey(1234), np.expand_dims(jnp.ones(in_bn), [0])))
    else:
        num_blocks : tuple = (3, 3, 3, 3)
        c_hidden : tuple = (16, 32, 64, 128)
        in_bn = metadata.u_shape
        branch_layers = 0*[branch_net.num_hidden_neurons] + [branch_net.num_output_neurons]
        bn_fnn = ResNet(layers_fnn=branch_layers, num_blocks=num_blocks, c_hidden=c_hidden, act_fn=jax.nn.relu) #jnp.sin #jax.nn.relu
        print(bn_fnn.tabulate(jax.random.PRNGKey(1234), np.expand_dims(jnp.ones(in_bn), [0,3])))        

    lr = settings.training_settings.learning_rate
    
    bs = settings.training_settings.batch_size_branch * settings.training_settings.batch_size_coord,
    adaptive_weights_shape = bs if settings.training_settings.use_adaptive_weights else -1
    
    model = DeepONet(lr, bn_fnn, in_bn, tn_fnn, in_tn, 
                     settings.dirs.models_dir,
                     decay_steps=settings.training_settings.decay_steps,
                     decay_rate=settings.training_settings.decay_rate, 
                     do_fnn=do_fnn,
                     transfer_learning=settings.transfer_learning,
                     adaptive_weights_shape=adaptive_weights_shape)

    ### Train ###
    model.train(dataloader, dataloader_val, nIter, save_every=200)
    model.plotLosses(settings.dirs.figs_dir)

# settings_path = "scripts/threeD/setups/settings.json"
# train(settings_path)