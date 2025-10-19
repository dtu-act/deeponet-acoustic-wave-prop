# ==============================================================================
# Copyright 2025 Technical University of Denmark
# Author: Nikolas Borrel-Jensen
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import deeponet_acoustics.datahandlers.data_rw as rw
import deeponet_acoustics.utils.feat_expansion as featexp
import deeponet_acoustics.utils.utils as utils
import deeponet_acoustics.visualization.visualizing as plotting
from deeponet_acoustics.datahandlers.datagenerators import (
    normalizeData,
    normalizeDomain,
)
from deeponet_acoustics.models.datastructures import (
    NetworkArchitectureType,
    TransferLearning,
)
from deeponet_acoustics.models.deeponet import DeepONet
from deeponet_acoustics.models.networks_flax import setupNetwork
from deeponet_acoustics.setup.configurations import setupPlotParams
from deeponet_acoustics.setup.data import setupData
from deeponet_acoustics.setup.settings import SimulationSettings


def inference(settings_dict, custom_data_path=None, tmax=None, do_animate=False):
    settings = SimulationSettings(settings_dict)

    if custom_data_path:
        testing_data_path = custom_data_path
    else:
        testing_data_path = settings.dirs.testing_data_path

    branch_net = settings.branch_net
    trunk_net = settings.trunk_net

    if not tmax:
        tmax = settings.tmax

    # load testing data
    dataset_test = rw.loadDataFromH5(testing_data_path, tmax=tmax)
    simulation_settings = rw.loadAttrFromH5(settings.dirs.training_data_path)

    c_phys = 343
    prune_spatial = 2

    mesh_test = dataset_test.mesh[::prune_spatial, :]
    t_test = dataset_test.t
    p_test = dataset_test.pressures[:, :, ::prune_spatial]
    up_test = dataset_test.upressures
    x0_srcs = dataset_test.x0_srcs
    ushape_test = dataset_test.ushape  # TODO for CNNs

    do_fnn = settings.branch_net.architecture != NetworkArchitectureType.RESNET
    u_test, s_test, t1d_test, grid1d_test = setupData(
        mesh_test, p_test, up_test, t_test, ushape_test, do_fnn
    )

    y_test = jnp.hstack([grid1d_test, t1d_test])

    if settings.normalize_data:
        from_zero = settings.trunk_net.activation == "relu"
        yminmax = simulation_settings["domain_minmax"]
        if dataset_test.dim == 1:
            ymin, ymax = yminmax[0], yminmax[1]
        else:  # 2D
            ymin, ymax = min(yminmax[:, 0]), max(yminmax[:, 1])
        y_test = normalizeDomain(y_test, ymin, ymax, from_zero=from_zero)

    y_feat_fn = featexp.fourierFeatureExpansion_f0(settings.f0_feat)
    # c = simulation_settings['c']
    # fmax = simulation_settings['fmax']
    # y_feat_fn = featexp.fourierFeatureExpansion_gaussian((10,3), mean=fmax/2, std_dev=fmax/2)
    # y_feat_fn = featexp.fourierFeatureExpansion_exact_sol([fmax, fmax/2, fmax/6], c, 3.0, 3.0)

    y_test = y_feat_fn(y_test)

    # setup network
    in_bn = u_test.shape[1::]
    in_tn = y_test.shape

    trunk_nn = setupNetwork(trunk_net, "tn")
    branch_nn = setupNetwork(branch_net, "bn", len(in_bn))

    transfer_learning = TransferLearning(
        {"transfer_learning": {"resume_learning": True}}, settings.dirs.models_dir
    )

    model = DeepONet(
        settings.training_settings,
        dataset_test,
        (branch_nn, in_bn),
        (trunk_nn, in_tn),
        settings.dirs.models_dir,
        transfer_learning=transfer_learning,
    )

    model.plotLosses(settings.dirs.figs_dir)

    tdim = t_test.shape[0]
    S_pred_srcs = np.empty((x0_srcs.shape[0], tdim, mesh_test.shape[0]), dtype=float)
    S_test_srcs = np.empty((x0_srcs.shape[0], tdim, mesh_test.shape[0]), dtype=float)

    figs_dir = settings.dirs.figs_dir
    path_receivers = os.path.join(figs_dir, "receivers")
    Path(path_receivers).mkdir(parents=True, exist_ok=True)

    for i_src in range(x0_srcs.shape[0]):
        u_test_i = u_test[i_src, :]
        s_test_i = s_test[i_src, :]

        # Predict
        s_pred_i = model.predict_s(model.params, u_test_i, y_test)

        S_pred_srcs[i_src, :, :] = np.array(s_pred_i).reshape(tdim, -1)
        S_test_srcs[i_src, :, :] = np.array(s_test_i).reshape(tdim, -1)

        # x0 = x0_srcs[i_src]
        # if dataset_test.dim == 2:
        #     IO.writeTriangleXdmf(grid_test, conn_test-1, t_test, S_pred_srcs[i_src], os.path.join(path_receivers, f"{i_src}_wavefield_pred{x0}.xdmf"))
        #     IO.writeTriangleXdmf(grid_test, conn_test-1, t_test, S_test_srcs[i_src], os.path.join(path_receivers, f"{i_src}_wavefield_test{x0}.xdmf"))

    path_receivers = os.path.join(figs_dir, "receivers")
    Path(path_receivers).mkdir(parents=True, exist_ok=True)

    r0_list, r0_indxs = utils.calcReceiverPositionsSimpleDomain(grid1d_test, x0_srcs)

    if settings.normalize_data:
        from_zero = settings.trunk_net.activation == "relu"
        yminmax = simulation_settings["domain_minmax"]
        if dataset_test.dim == 1:
            ymin, ymax = yminmax[0], yminmax[1]
        else:  # 2D
            ymin, ymax = min(yminmax[:, 0]), max(yminmax[:, 1])
        r0_list = normalizeData(r0_list, ymin, ymax, from_zero=from_zero)

    setupPlotParams(True)

    N_srcs = len(r0_indxs)

    ir_ref_srcs = np.empty(N_srcs, dtype=object)
    ir_pred_srcs = np.empty(N_srcs, dtype=object)

    for i_src in range(N_srcs):
        ir_ref_srcs[i_src] = np.expand_dims(S_test_srcs[i_src, :, r0_indxs[i_src]], 1)
        ir_pred_srcs[i_src] = np.expand_dims(S_pred_srcs[i_src, :, r0_indxs[i_src]], 1)

    plotting.writeIRPlotsWithReference(
        x0_srcs,
        np.expand_dims(r0_list, 1),
        t_test / c_phys,
        ir_pred_srcs,
        ir_ref_srcs,
        tmax / c_phys,
        figs_dir=path_receivers,
        animate=do_animate,
    )

    if dataset_test.dim == 1:
        plotting.plotWaveFields1D(
            grid1d_test,
            t1d_test,
            S_pred_srcs,
            S_test_srcs,
            x0_srcs,
            c_phys,
            path_receivers,
        )
