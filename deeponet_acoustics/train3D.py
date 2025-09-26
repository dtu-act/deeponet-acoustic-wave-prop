# ==============================================================================
# Copyright 2023 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import os, shutil
import numpy as np

import deeponet_acoustics.datahandlers.data_rw as rw
from deeponet_acoustics.models.datastructures import NetworkArchitectureType
import deeponet_acoustics.setup.parsers as parsers
from deeponet_acoustics.datahandlers.datagenerators import DataH5Compact, DatasetStreamer, NumpyLoader, printInfo
from deeponet_acoustics.models.networks_flax import setupNetwork
from deeponet_acoustics.models.deeponet import DeepONet
from deeponet_acoustics.utils.feat_expansion import fourierFeatureExpansion_f0
from deeponet_acoustics.setup.settings import SimulationSettings

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
    sim_params_path = os.path.join(settings.dirs.training_data_path, "simulation_parameters.json")
    phys_params = rw.loadSimulationParametersJson(sim_params_path)
    c_phys = phys_params.c_phys
    
    f = settings.f0_feat
    y_feat = fourierFeatureExpansion_f0(f)

    flatten_ic = branch_net.architecture != NetworkArchitectureType.RESNET

    # setup dataloaders
    metadata = DataH5Compact(settings.dirs.training_data_path, tmax=tmax, t_norm=c_phys, 
        norm_data=settings.normalize_data, flatten_ic=flatten_ic)
    dataset = DatasetStreamer(metadata, training.batch_size_coord, y_feat_extractor=y_feat)
    metadata_val = DataH5Compact(settings.dirs.testing_data_path, tmax=tmax, t_norm=c_phys, 
        norm_data=settings.normalize_data, flatten_ic=flatten_ic)
    dataset_val = DatasetStreamer(metadata_val, training.batch_size_coord, y_feat_extractor=y_feat)
    
    dataloader = NumpyLoader(dataset, batch_size=training.batch_size_branch, shuffle=True, drop_last=len(dataset) > 1)
    dataloader_val = NumpyLoader(dataset_val, batch_size=training.batch_size_branch, shuffle=True, num_workers=0) # do not drop last, validation set has few samples

    if not np.allclose(metadata.tsteps, metadata_val.tsteps):
        raise Exception(f"Time steps differs between training and validation data: \nN_train={len(metadata.tsteps)}, N_val={len(metadata_val.tsteps)}, dt_train={metadata.tsteps[1]-metadata.tsteps[0]} and dt_val={metadata_val.tsteps[1]-metadata_val.tsteps[0]}.\n The network is not supposed to learn temporal interpolation. Exiting.")
    
    printInfo(metadata, metadata_val, training.batch_size_coord, training.batch_size_branch)

    # setup network
    in_tn = y_feat(np.array([[0.0,0.0,0.0,0.0]])).shape[1]
    tn_fnn = setupNetwork(trunk_net, in_tn, 'tn')
    in_bn = metadata.u_shape
    bn_fnn = setupNetwork(branch_net, in_bn, 'bn')

    lr = settings.training_settings.learning_rate    
    bs = settings.training_settings.batch_size_branch * settings.training_settings.batch_size_coord,
    adaptive_weights_shape = bs if settings.training_settings.use_adaptive_weights else []
    
    model = DeepONet(lr, bn_fnn, tn_fnn, 
                     settings.dirs.models_dir,
                     decay_steps=settings.training_settings.decay_steps,
                     decay_rate=settings.training_settings.decay_rate,
                     transfer_learning=settings.transfer_learning,
                     adaptive_weights_shape=adaptive_weights_shape)

    ### Train ###
    model.train(dataloader, dataloader_val, nIter, save_every=200)
    model.plotLosses(settings.dirs.figs_dir)

# settings_path = "scripts/threeD/setups/settings.json"
# train(settings_path)