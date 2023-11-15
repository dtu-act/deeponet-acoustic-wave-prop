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
from models.datastructures import EvaluationSettings, NetworkArchitectureType, TransferLearning
import utils.utils as utils

from datahandlers.datagenerators import DataH5Compact, DatasetStreamer
from models.deeponet import DeepONet
from models.networks_flax import setupFNN
from utils.feat_expansion import fourierFeatureExpansion_f0
import datahandlers.data_rw as rw
import plotting.visualizing as plotting
from setup.configurations import setupPlotParams
from setup.settings import SimulationSettings
import setup.parsers as parsers
import datahandlers.io as IO

def evaluate(settings_path, settings_eval_path):
    prune_spatial = 1
    mod_fnn_bn, mod_fnn_tn = True, True # manually set (should be defined and read from JSON)

    settings_dict = parsers.parseSettings(settings_path)
    settings = SimulationSettings(settings_dict)
    settings.dirs.createDirs()

    path_receivers = os.path.join(settings.dirs.figs_dir , "receivers")
    Path(path_receivers).mkdir(parents=True, exist_ok=True)

    settings_eval_dict = parsers.parseSettings(settings_eval_path)
    settings_eval = EvaluationSettings(settings_eval_dict)

    tmax = settings_eval.tmax

    do_fnn = settings.branch_net.architecture == NetworkArchitectureType.MLP
    if not do_fnn:
        raise Exception("CNN/ResNet not supported in 3D")
    
    branch_net = settings.branch_net
    trunk_net = settings.trunk_net

    sim_params_path = os.path.join(settings.dirs.training_data_path, "simulation_parameters.json")
    phys_params = rw.loadSimulationParametersJson(sim_params_path)
    c_phys = phys_params.c_phys

    ### Initialize model ###
    f = settings.f0_feat
    y_feat = fourierFeatureExpansion_f0(f)
    
    metadata = DataH5Compact(settings_eval.data_path, tmax=tmax, t_norm=c_phys,
        flatten_ic=do_fnn, data_prune=prune_spatial, norm_data=settings.normalize_data)
    dataset = DatasetStreamer(metadata, y_feat_extractor=y_feat)

    # assert that the time step resolution of the test data is the same as the resolution of the trained model, 
    # since we do not interpolate in time (not needed)
    metadata_model = DataH5Compact(settings.dirs.training_data_path, tmax=tmax, t_norm=c_phys, 
        flatten_ic=do_fnn, data_prune=prune_spatial, norm_data=settings.normalize_data, MAXNUM_DATASETS=1)
    if not np.allclose(metadata.tsteps, metadata_model.tsteps):
        raise Exception(f"Time steps differs between training and validation data: \nN_train={len(metadata.tsteps)} N_val={len(metadata_model.tsteps)}, dt_train={metadata.tsteps[1]-metadata.tsteps[0]} and dt_val={metadata_model.tsteps[1]-metadata_model.tsteps[0]}.\n The network is not supposed to learn temporal interpolation. Exiting.")

    ############## SETUP NETWORK ##############
    in_tn = y_feat(np.array([[0.0,0.0,0.0,0.0]])).shape[1]
    tn_fnn = setupFNN(trunk_net, "tn", mod_fnn=mod_fnn_tn)
    print(tn_fnn.tabulate(jax.random.PRNGKey(1234), np.expand_dims(jnp.ones(in_tn), [0])))
    
    in_bn = metadata.u_shape
    bn_fnn = setupFNN(branch_net, "bn", mod_fnn=mod_fnn_bn)
    print(bn_fnn.tabulate(jax.random.PRNGKey(1234), np.expand_dims(jnp.ones(in_bn), [0])))

    lr = settings.training_settings.learning_rate
    decay_steps=settings.training_settings.decay_steps
    decay_rate=settings.training_settings.decay_rate
    transfer_learning = TransferLearning({'transfer_learning': {'resume_learning': True}}, 
                                        settings.dirs.models_dir)
    
    model = DeepONet(lr, bn_fnn, in_bn, tn_fnn, in_tn, settings.dirs.models_dir, 
                    decay_steps, decay_rate, do_fnn=do_fnn, 
                    transfer_learning=transfer_learning)

    model.plotLosses(settings.dirs.figs_dir)

    tdim = metadata.num_tsteps
    xxyyzztt = metadata.xxyyzztt
    y_in = y_feat(xxyyzztt)

    xxyyzz_phys = metadata.denormalizeSpatial(xxyyzztt[:,0:3])
    mesh_phys = metadata.denormalizeSpatial(metadata.mesh)
    tsteps_phys = metadata.denormalizeTemporal(metadata.tsteps/c_phys)

    num_srcs = dataset.N    

    ############## WRITE FULL WAVE FIELD ##############
    if settings_eval.write_full_wave_field:
        S_pred_srcs = np.empty((num_srcs,tdim,dataset.Pmesh), dtype=float)
        S_test_srcs = np.empty((num_srcs,tdim,dataset.Pmesh), dtype=float)        

        for i_src in range(num_srcs):
            (u_test_i,*_),s_test_i,_,x0 = dataset[i_src]

            s_pred_i = np.zeros(s_test_i.reshape(tdim,-1).shape)
            # Predict
            for i,_ in enumerate(metadata.tsteps):
                yi = y_in.reshape(tdim,s_pred_i.shape[1],-1)[i,:]
                s_pred_i[i,:] = model.predict_s(model.params, u_test_i, yi)

            S_pred_srcs[i_src,:,:] = np.asarray(s_pred_i)
            S_test_srcs[i_src,:,:] = np.asarray(s_test_i).reshape(tdim,-1)

            x0 = metadata.denormalizeSpatial(x0)

            IO.writeTetraXdmf(mesh_phys, metadata.conn,
                        tsteps_phys, S_pred_srcs[i_src], 
                        os.path.join(path_receivers, f"wavefield_pred{x0}.xdmf"))
            IO.writeTetraXdmf(mesh_phys, metadata.conn,
                        tsteps_phys, S_test_srcs[i_src], 
                        os.path.join(path_receivers, f"wavefield_ref{x0}.xdmf"))

    ############## PREDICT IRs ##############
    ir_pred_srcs = np.empty((num_srcs), dtype=object)    
    x0_srcs = []

    if settings_eval.snap_to_grid:
        r0_list, r0_indxs = utils.getNearestFromCoordinates(xxyyzz_phys, settings_eval.receiver_pos)
        ir_ref_srcs = np.empty((num_srcs), dtype=object)
    else:
        r0_list = settings_eval.receiver_pos
        ir_ref_srcs = []

    for i_src in range(num_srcs):        
        r0_list_norm = metadata.normalizeSpatial(r0_list[i_src])
        
        (u_test_i,*_),s_test_i,_,x0 = dataset[i_src]

        x0 = metadata.denormalizeSpatial(x0)
        x0_srcs.append(x0)

        y_rcvs = np.repeat(np.array(r0_list_norm), len(metadata.tsteps), axis=0)
        tsteps_rcvs = np.tile(metadata.tsteps, len(r0_list_norm))
        yi = y_feat(np.concatenate((y_rcvs, np.expand_dims(tsteps_rcvs, 1)), axis=1))

        # predict using the DeepONet models
        ir_predict = model.predict_s(model.params, u_test_i, yi)

        ir_pred_srcs[i_src] = np.asarray(ir_predict).reshape(tdim, -1, order='F') # 'F': split reading from beginning of array
        if settings_eval.snap_to_grid:
            ir_ref_srcs[i_src] = np.asarray(s_test_i).reshape(tdim,-1)[:,r0_indxs[i_src]]
    
    setupPlotParams(True)
    
    ############## WRITE RESULTS ##############
    if settings_eval.snap_to_grid:
        if settings_eval.write_ir_plots:
            plotting.writeIRPlotsWithReference(x0_srcs,r0_list,
                tsteps_phys,ir_pred_srcs,ir_ref_srcs,tmax/c_phys,path_receivers,
                animate=settings_eval.write_ir_animations)
    
        if settings_eval.write_ir_wav:
            plotting.writeWav(x0_srcs,r0_list,
                tsteps_phys,ir_pred_srcs,tmax/c_phys,path_receivers,'pred')
            plotting.writeWav(x0_srcs,r0_list,
                tsteps_phys,ir_ref_srcs,tmax/c_phys,path_receivers,'ref')
    else:
        if settings_eval.write_ir_plots:
            plotting.writeIRPlots(x0_srcs,r0_list,
                tsteps_phys,ir_pred_srcs,tmax/c_phys,path_receivers,
                animate=settings_eval.write_ir_animations)
    
        if settings_eval.write_ir_wav:
            plotting.writeWav(x0_srcs,r0_list,
                tsteps_phys,ir_pred_srcs,tmax/c_phys,path_receivers,'pred')


# settings_path = "scripts/threeD/setups/cube.json"
# settings_eval_path = "scripts/threeD/setups/cube_src_dir_eval.json"

# evaluate(settings_path, settings_eval_path)