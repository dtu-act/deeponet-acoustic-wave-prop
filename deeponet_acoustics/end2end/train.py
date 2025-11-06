# ==============================================================================
# Copyright 2025 Technical University of Denmark
# Author: Nikolas Borrel-Jensen
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import json
import os
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

import deeponet_acoustics.datahandlers.data_rw as rw
from deeponet_acoustics.datahandlers.datagenerators import (
    DataH5Compact,
    DatasetStreamer,
    numpy_collate,
)
from deeponet_acoustics.models.datastructures import NetworkArchitectureType
from deeponet_acoustics.models.deeponet import DeepONet
from deeponet_acoustics.models.networks_flax import setupNetwork
from deeponet_acoustics.setup.settings import SimulationSettings
from deeponet_acoustics.utils.feat_expansion import fourierFeatureExpansion_f0
from deeponet_acoustics.visualization.info_printing import datasetInfo, networkInfo


def train(settings_dict: dict[str, Any]):
    settings = SimulationSettings(settings_dict)
    if (
        settings.transfer_learning is None
        or not settings.transfer_learning.resume_learning
    ):
        settings.dirs.createDirs(delete_existing=True)

    # copy settings
    with open(os.path.join(settings.dirs.id_dir, "settings.json"), "w") as json_file:
        json_file.write(json.dumps(settings_dict, indent=4))

    training = settings.training_settings
    branch_net = settings.branch_net
    trunk_net = settings.trunk_net

    tmax = settings.tmax
    nIter = training.iterations

    # load training data
    sim_params_path = os.path.join(
        settings.dirs.training_data_path, "simulation_parameters.json"
    )
    phys_params = rw.loadSimulationParametersJson(sim_params_path)
    c_phys = phys_params.c_phys

    f = settings.f0_feat
    y_feat_fn = fourierFeatureExpansion_f0(f)

    flatten_ic = branch_net.architecture != NetworkArchitectureType.RESNET

    # setup dataloaders
    metadata = DataH5Compact(
        settings.dirs.training_data_path,
        tmax=tmax,
        t_norm=c_phys,
        norm_data=settings.normalize_data,
        flatten_ic=flatten_ic,
        u_p_range=(-2.0, 2.0),
    )
    dataset = DatasetStreamer(
        metadata, training.batch_size_coord, y_feat_extract_fn=y_feat_fn
    )
    metadata_val = DataH5Compact(
        settings.dirs.testing_data_path,
        tmax=tmax,
        t_norm=c_phys,
        norm_data=settings.normalize_data,
        flatten_ic=flatten_ic,
        u_p_range=(-2.0, 2.0),
    )
    dataset_val = DatasetStreamer(
        metadata_val, training.batch_size_coord, y_feat_extract_fn=y_feat_fn
    )

    dataloader = DataLoader(
        dataset,
        batch_size=training.batch_size_branch,
        shuffle=True,
        collate_fn=numpy_collate,
        drop_last=len(dataset) > 1,
    )
    # do not drop last, validation set has few samples
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=training.batch_size_branch,
        shuffle=True,
        collate_fn=numpy_collate,
        drop_last=False,
    )

    if not np.allclose(metadata.tsteps, metadata_val.tsteps):
        raise Exception(
            f"Time steps differs between training and validation data: \nN_train={len(metadata.tsteps)}, N_val={len(metadata_val.tsteps)}, dt_train={metadata.tsteps[1] - metadata.tsteps[0]} and dt_val={metadata_val.tsteps[1] - metadata_val.tsteps[0]}.\n The network is not supposed to learn temporal interpolation. Exiting."
        )

    datasetInfo(
        metadata, metadata_val, training.batch_size_coord, training.batch_size_branch
    )

    # setup network
    input_example = next(iter(dataloader))[0][1]
    in_tn = input_example.shape[-1]
    tn_fnn = setupNetwork(trunk_net, "tn")
    networkInfo(tn_fnn, in_tn)
    in_bn = metadata.u_shape
    bn_fnn = setupNetwork(branch_net, "bn", len(metadata.u_shape))
    networkInfo(bn_fnn, in_bn)

    model = DeepONet(
        settings.training_settings,
        metadata,
        (bn_fnn, in_bn),
        (tn_fnn, in_tn),
        settings.dirs.models_dir,
        transfer_learning=settings.transfer_learning,
    )

    ### Train ###
    model.train(dataloader, dataloader_val, nIter, save_every=200)
    model.plotLosses(settings.dirs.figs_dir)
