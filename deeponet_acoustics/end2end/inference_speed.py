# ==============================================================================
# Copyright 2025 Technical University of Denmark
# Author: Nikolas Borrel-Jensen
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import os
import time
from pathlib import Path

import numpy as np
from utils.feat_expansion import fourierFeatureExpansion_f0

import datahandlers.data_rw as rw
from datahandlers.datagenerators import DataH5Compact, DatasetStreamer
from deeponet_acoustics.setup.settings import SimulationSettings
from deeponet_acoustics.visualization.info_printing import networkInfo
from models.datastructures import NetworkArchitectureType, TransferLearning
from models.deeponet import DeepONet
from models.networks_flax import setupNetwork


def evaluate_inference_speed3D(settings: SimulationSettings, tmax_eval=0.05):
    # id = "bilbao_4ppw_bs96_1500"
    # id = "dome_6ppw_1stquad_resnet"
    # output_dir = "/work3/nibor/data/deeponet/output3D"
    # input_dir = "/work3/nibor/1TB/input3D/"
    # settings_path = os.path.join(output_dir, f"{id}/settings.json")

    # some random receivers
    receivers = np.array(
        [
            [0.8, 0.8, 0.8],
            [0.9, 0.9, 0.9],
            [1.0, 1.0, 1.0],
            [1.1, 1.1, 1.1],
            [1.2, 1.2, 1.2],
        ]
    )

    settings.dirs.createDirs()

    do_fnn = settings.branch_net.architecture != NetworkArchitectureType.RESNET
    branch_net = settings.branch_net
    trunk_net = settings.trunk_net

    tmax = settings.tmax

    sim_params_path = os.path.join(
        settings.dirs.training_data_path, "simulation_parameters.json"
    )
    phys_params = rw.loadSimulationParametersJson(sim_params_path)
    c_phys = phys_params.c_phys

    ### Initialize model ###
    f = settings.f0_feat
    y_feat_fn = fourierFeatureExpansion_f0(f, c_phys)

    prune_spatial = 2
    metadata = DataH5Compact(
        settings.dirs.testing_data_path,
        tmax=tmax,
        t_norm=c_phys,
        flatten_ic=do_fnn,
        data_prune=prune_spatial,
        norm_data=settings.normalize_data,
    )
    dataset = DatasetStreamer(metadata, y_feat_extract_fn=y_feat_fn)

    # setup network
    in_bn = metadata.u_shape[0]
    in_tn = y_feat_fn(np.array([[0.0, 0.0, 0.0, 0.0]])).shape[1]

    # setup network
    in_tn = y_feat_fn(np.array([[0.0, 0.0, 0.0, 0.0]])).shape[1]
    tn_fnn = setupNetwork(trunk_net, "tn")
    networkInfo(tn_fnn, in_tn)
    in_bn = metadata.u_shape
    bn_fnn = setupNetwork(branch_net, "bn", len(metadata.u_shape))
    networkInfo(bn_fnn, in_bn)

    transfer_learning = TransferLearning(
        {"transfer_learning": {"resume_learning": True}}, settings.dirs.models_dir
    )

    model = DeepONet(
        settings.training_settings,
        metadata,
        (bn_fnn, in_bn),
        (tn_fnn, in_tn),
        settings.dirs.models_dir,
        transfer_learning=transfer_learning,
    )

    path_receivers = os.path.join(settings.dirs.figs_dir, "receivers")
    Path(path_receivers).mkdir(parents=True, exist_ok=True)

    # CONSTRUCT DATA
    # for number of timesteps and receivers (for a fixed source position u)
    # use sample rate from simulation (could be chosen arbitrarily if needed)
    assert phys_params.fmax == 1000
    # scaled timesteps
    tsteps_eval = np.linspace(
        0, tmax_eval / phys_params.c, int(tmax_eval / (1 / (phys_params.fmax * 2)))
    )

    (u, *_), *_ = dataset[0]
    y_rcvs = np.repeat(np.array(receivers), len(tsteps_eval), axis=0)
    t_all = np.tile(tsteps_eval, receivers.shape[0])
    y_rcvs_feat = y_feat_fn(np.hstack((y_rcvs, t_all.reshape(-1, 1))))

    _ = model.predict_s(model.params, u, y_rcvs_feat)  # warmup - compiling
    _ = model.predict_s(model.params, u, y_rcvs_feat)  # warmup - just to make sure...

    total_time_ns = 0.0
    N = 100
    i = 0
    while i < N:
        start_time = time.perf_counter_ns()
        _ = model.predict_s(model.params, u, y_rcvs_feat)
        end_time = time.perf_counter_ns()
        total_time_ns += end_time - start_time
        i += 1

    ns_to_ms_factor = 1e6
    evaluation_time_ms = total_time_ns / N / ns_to_ms_factor

    out_path = os.path.join(settings.dirs.id_dir, "inferance_timings.txt")
    with open(out_path, "w") as f:
        f.write("----------------------\n")
        f.write("Runtime measurements of the surrogate model:\n")
        f.write("----------------------\n")
        f.write(f"TRUNK NET: {'MLP' if trunk_net.architecture == 1 else 'MOD-MLP'}\n")
        f.write(f"   #trunk layers (pde) = {trunk_net.num_hidden_layers}\n")
        f.write(f"   num_output_neurons = {trunk_net.num_output_neurons}\n")
        if branch_net.architecture == NetworkArchitectureType.RESNET:
            f.write("BRANCH NET: RESNET\n")
            f.write(f"   num_hidden_neurons = {branch_net.num_hidden_layers}\n")
            f.write(f"   num_output_neurons = {branch_net.num_output_neurons}\n")
            f.write(f"   num_output_neurons = {branch_net.num_group_blocks}\n")
            f.write(f"   cnn_hidden_layers = {branch_net.cnn_hidden_layers}\n")
        else:
            f.write(
                f"BRANCH NET: {'MLP' if trunk_net.architecture == NetworkArchitectureType.MLP else 'MOD-MLP'}\n"
            )
            f.write(f"   #neurons (pde) = {branch_net.num_hidden_neurons}\n")
            f.write(f"   num_output_neurons = {branch_net.num_output_neurons}\n")
        f.write(f"tmax time = {tmax_eval}\n")
        f.write(f"fmax = {phys_params.fmax}\n")
        f.write(f"Inferance performance (ms): {evaluation_time_ms}\n")
        f.write("----------------------")
