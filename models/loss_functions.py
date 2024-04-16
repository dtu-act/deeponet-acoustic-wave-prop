# ==============================================================================
# Copyright 2023 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import jax.numpy as jnp
from jax import vmap
from models.CEOD import calcCEOD

def lossCEOD(params: dict, batch_target: list[float], 
             branch_net: callable, operator_net: callable, ceod_indx: int = -1):
    '''
        u: #batch x #sample_size
        y: #batch x #batch_coord x #inputs
    '''

    # operator_net calculates inner product from branch latent output and the output coordinates y.
    # inner vmap for operator_net maps along the coordinate dimension for a single batch
    calc_inner_single_batch_fn = vmap(operator_net, (None, None, 0))
    # outer vmap along batch dimensions (src pos) w.r.t. branch and coordinates pairs
    s_pred_fn = vmap(calc_inner_single_batch_fn, (None, 0, 0))

    # extract data        
    inputs_tar, outputs_tar, *_ = batch_target
    u_tar,y_tar,u_src = inputs_tar # make sure to modify the data loader to return u_src

    assert len(u_src) > 0

    # RESIDUAL LOSS - forward pass 
    branch_latent_batch = vmap(branch_net, (None, 0))(params, u_tar) # vmaps along branch batch dimension     
    s_pred_tar = s_pred_fn(params, branch_latent_batch, y_tar) # calculate predictions on coordinate locations

    loss_residual = jnp.mean((outputs_tar.flatten() - s_pred_tar.flatten())**2)

    # DISCREPENCY LOSS - forward pass
    branch_latent_batch = vmap(branch_net, (None, 0))(params, u_src) # vmaps along branch batch dimension  
    s_pred_src = s_pred_fn(params, branch_latent_batch, y_tar) # calculate predictions on coordinate locations

    # get output from layer 'ceod_indx' for both source and target data
    branch_discrepancy_layer_tar = vmap(branch_net, (None, 0, None))(params, u_tar, ceod_indx)
    branch_discrepancy_layer_src = vmap(branch_net, (None, 0, None))(params, u_src, ceod_indx)

    loss_ceod = calcCEOD(branch_discrepancy_layer_tar, outputs_tar,
                         branch_discrepancy_layer_src, s_pred_src)

    # hybrid loss
    loss = 1 * loss_residual + 10 * loss_ceod

    return loss

def loss(params: list, batch: list, branch_net: callable, operator_net: callable, apply_adaptive_weights=False):
    '''
        u: #batch x #sample_size
        y: #batch x #batch_coord x #inputs
    '''

    # operator_net calculates inner product from branch latent output and the output coordinates y.
    # inner vmap for operator_net maps along the coordinate dimension for a single batch
    calc_inner_single_batch_fn = vmap(operator_net, (None, None, 0))
    # outer vmap along batch dimensions (src pos) w.r.t. branch and coordinates pairs
    s_pred_fn = vmap(calc_inner_single_batch_fn, (None, 0, 0))

    # extract data
    inputs, outputs, idx_coord, *_ = batch
    u,y = inputs

    # RESIDUAL LOSS - forward pass 
    branch_latent_batch = vmap(branch_net, (None, 0))(params, u) # vmaps along branch batch dimension     
    s_pred = s_pred_fn(params, branch_latent_batch, y) # calculate predictions on coordinate locations

    if apply_adaptive_weights:
        adaptive_weights = params['adaptive_weights'][idx_coord.flatten()]

        # Compute loss
        return jnp.mean((outputs.flatten() - s_pred.flatten() * adaptive_weights )**2)
    else:
        return jnp.mean((outputs.flatten() - s_pred.flatten())**2)