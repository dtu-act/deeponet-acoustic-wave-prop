# ==============================================================================
# Copyright 2025 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import jax.numpy as np
from jax import random, vmap
import matplotlib.pyplot as plt
import deeponet_acoustics.utils.utils as utils
from deeponet_acoustics.setup.settings import TransferLearning
import deeponet_acoustics.datahandlers.data_rw as rw

def setupDataResample(p,t,grid,dx_u,xminmax,yminmax,do_plot=False):
    # Resample functions to match num of sensors    
    srcs, XX, YY = utils.resampleICs2D(grid,p,dx_u,xminmax,yminmax)
    assert(not np.isnan(srcs).any())
    N = p.shape[0]
    u = np.array(srcs)
    s = np.array(p).reshape(N,-1,)

    t1d = np.repeat(t, grid.shape[0]).reshape(-1,1)
    grid1d = np.tile(grid, (t.shape[0],1))

    if do_plot:
        plt.figure()
        ax = plt.axes(projection ='3d')
        ax.plot_trisurf(XX.flatten(), YY.flatten(), srcs[0,:])
        plt.show(block=True)

        t_indx_i = 10
        Npoints = p.shape[-1]
        plt.figure(); ax = plt.axes(projection ='3d'); ax.plot_trisurf(grid1d[0:(Npoints-1),0], grid1d[0:(Npoints-1),1], s[0,t_indx_i*Npoints:((t_indx_i+1)*Npoints)])
        plt.show(block=True)

    return u,s,t1d,grid1d

def setupDataResample1D(p,t,grid,dx_u,xminmax):
    u_train = utils.resampleICs1D(grid,p,dx_u,xminmax)
    assert(not np.isnan(u_train).any())
    N = p.shape[0]
    u = np.array(u_train)
    s = np.array(p).reshape(N,-1,)

    t1d = np.repeat(t, grid.shape[0]).reshape(-1,1)
    grid1d = np.tile(grid, (t.shape[0],1))

    return u,s,t1d,grid1d

def setupData(grid,p,up,t,ushape,flatten_input=True):
    N = p.shape[0]
    s = np.array(p).reshape(N,-1,)

    if flatten_input:
        u = np.array(up).reshape(N,-1,)
    else:
        Nu = up.shape[0]
        reshape_to = np.concatenate((np.array([Nu]),ushape), dtype=int)
        u = np.array(up).reshape(reshape_to)

    t1d = np.repeat(t, grid.shape[0]).reshape(-1,1)
    grid1d = np.tile(grid, (t.shape[0],1))

    return u,s,t1d,grid1d

def setupTransferLearningFromMat(transfer_learning: TransferLearning, flatten_data=True):        
    if transfer_learning.ceod == None:
        u_src_train = []
        u_src_val = []
    else:
        #r = 1541 # source square
        #r = 1200 # target right triangle
        u_src_train, u_src_val, *_ = rw.loadDataFromMat(transfer_learning.ceod.training_data_src_path,flatten_data=flatten_data, r=1541)

    return u_src_train, u_src_val

def setupTransferLearningData(transfer_learning: TransferLearning, flatten_data):
    if transfer_learning.ceod == None:
        u_src_train = []
        u_src_val = []
    else:
        data_train = rw.loadDataFromH5(transfer_learning.ceod.training_data_src_path)
        u_src_train,*_ = setupData(data_train.mesh,
                                   data_train.pressures,
                                   data_train.upressures,
                                   data_train.t,
                                   data_train.ushape,
                                   flatten_data)

        data_val = rw.loadDataFromH5(transfer_learning.ceod.testing_data_src_path)
        u_src_val,*_ = setupData(data_val.mesh,
                                 data_val.pressures,
                                 data_val.upressures,
                                 data_val.t,
                                 data_val.ushape,
                                 flatten_data)

    return u_src_train, u_src_val


def setupTrainingData(key, samples_srcs, p_srcs, xx, tt, P, N):
    """ Matrix dimension        
        N: number of samples (initial conditions in our case)
        m: number of sensor for each initial conditions sample
        Px: number of domain locations
        Pt: number of temporal steps

        Input:
            srcs:    (N, m)
            xx_srcs: (Px, Pt)
            tt_srcs: (Px, Pt)
            p_srcs:  (N, Px, Pt)

        Output:
            P: random select P data points in time/space

            u_train: (N X P, m)
            y_train: (N X P, 2)
            s_train: (N X P, 1)

        NOTE: the samples u for the branch net are repeated 
              P times for each 
    """    
    keys = random.split(key, N)
    u_train, y_train, s_train = vmap(setupOneTrainingData, (0,0,0,None,None,None))(keys, samples_srcs, p_srcs, 
                                                                                   np.array(xx), np.array(tt), P)

    N = samples_srcs.shape[0] # number of samples (initial conditions in our case)
    m = samples_srcs.shape[1] # number of sensors

    u_train = u_train.reshape(N*P,-1)
    y_train = y_train.reshape(N*P,-1)
    s_train = s_train.reshape(N*P,-1)

    return np.float32(u_train), np.float32(y_train), np.float32(s_train)

def setupOneTrainingData(key, sample, p, xx, tt, P):
    """ Matrix dimension        
        Output:
            P: random select P data points in time/space

            u_train: (P, m)
            y_train: (P, 2)
            s_train: (P, 1)

        NOTE: the samples u for the branch net are repeated 
              P times for each 
    """
    indxs = random.randint(key, (P,), minval=0, maxval=xx.size)

    u_train = np.tile(sample, (P,1))
    y_train = np.hstack([xx.reshape(-1,1)[indxs], tt.reshape(-1,1)[indxs]])
    s_train = p.reshape(-1,1)[indxs]

    return np.float32(u_train), np.float32(y_train), np.float32(s_train)