# ==============================================================================
# Copyright 2023 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import numpy as np
from datahandlers.datagenerators import IData

def getNearestFromCoordinates(grid,coords):
    r0 = np.empty(coords.shape, dtype=float)
    r0_indxs = np.empty(coords.shape[0], dtype=int)

    for i,X in enumerate(coords):
        indx_x = np.sum(np.abs(grid-coords[i,:]),1).argmin()
        r0[i,:] = grid[indx_x]
        r0_indxs[i] = int(indx_x)
    
    return r0,r0_indxs

def calcReceiverPositionsSimpleDomain(grid, x0_srcs):
    x0_srcs = np.array(x0_srcs)
    
    def srcPos1D():
        r0 = np.empty(x0_srcs.shape)
        r0_indxs = np.empty(x0_srcs.shape[0], dtype=int)

        xmin = min(grid)
        xmax = max(grid)

        for i,x0 in enumerate(x0_srcs):
            if x0[0] <= (xmin + xmax)/2:
                rx = xmax - np.abs(xmin - x0[0])/2
            else:
                rx = xmin + (xmax - x0[0])/2

            indx_x = (np.abs(grid-np.array([rx]))).argmin()
            r0[i] = grid[indx_x]
            r0_indxs[i] = int(indx_x)
        
        return r0, r0_indxs

    def srcPos2D():
        r0 = np.empty(x0_srcs.shape, dtype=float)
        r0_indxs = np.empty(x0_srcs.shape[0], dtype=int)

        xmin = min(grid[:,0])
        xmax = max(grid[:,0])
        ymin = min(grid[:,1])
        ymax = max(grid[:,1])

        for i,x0 in enumerate(x0_srcs):

            if x0[0] <= (xmin + xmax)/2:
                rx = xmax - np.abs(xmin - x0[0])/2
            else:
                rx = xmin + (xmax - x0[0])/2
            
            if x0[1] <= (xmin + xmax)/2:
                ry = ymax - np.abs(ymin - x0[1])/2
            else:
                ry = ymin + (ymax - x0[1])/2

            indx_x = np.sum(np.abs(grid-np.array([rx,ry])),1).argmin()
            r0[i,:] = grid[indx_x]
            r0_indxs[i] = int(indx_x)
        
        return r0, r0_indxs

    def srcPos3D():
        r0 = np.empty(x0_srcs.shape, dtype=float)
        r0_indxs = np.empty(x0_srcs.shape[0], dtype=int)

        xmin = min(grid[:,0])
        xmax = max(grid[:,0])
        ymin = min(grid[:,1])
        ymax = max(grid[:,1])
        zmin = min(grid[:,2])
        zmax = max(grid[:,2])

        for i,x0 in enumerate(x0_srcs):
            if x0[0] <= (xmin + xmax)/2:
                rx = xmax - np.abs(xmin - x0[0])/2
            else:
                rx = xmin + (xmax - x0[0])/2
            
            if x0[1] <= (xmin + xmax)/2:
                ry = ymax - np.abs(ymin - x0[1])/2
            else:
                ry = ymin + (ymax - x0[1])/2

            if x0[2] <= (xmin + xmax)/2:
                rz = zmax - np.abs(zmin - x0[2])/2
            else:
                rz = zmin + (zmax - x0[2])/2

            indx_x = np.sum(np.abs(grid-[rx,ry,rz]),1).argmin()
            r0[i,:] = grid[indx_x]
            r0_indxs[i] = int(indx_x)
        
        return r0, r0_indxs

    if grid.shape[1] == 1:
        return srcPos1D()
    elif grid.shape[1] == 2:
        return srcPos2D()
    elif grid.shape[1] == 3:
        return srcPos3D()
    else:
        raise NotImplemented()

def extractSignal(r0,grid,p):
    tol = 10e-5
    result = np.where(np.abs(grid - r0) <= tol)
    indxs = result[0]
    assert(indxs.shape[0] >= 1)

    return p[:,indxs],grid[indxs,:]

def calcErrors(p_pred, p_ref, x0, r0, f):
    # find indexes for value diff. greater than -60dB
    threshold_dB = 60
    indxs_ref = np.where(20*np.log(np.abs(p_ref/np.max(abs(p_ref)))) > -threshold_dB)[0]
    indxs_pred = np.where(20*np.log(np.abs(p_pred/np.max(abs(p_ref)))) > -threshold_dB)[0]
    indxs = np.union1d(indxs_ref, indxs_pred)

    err_L1 = np.abs(p_pred - p_ref)
    err_rel = abs(err_L1[indxs]/p_ref[indxs])
    mean_err = np.round(np.mean(err_L1),4)
    mean_err_rel = np.round(np.mean(err_rel),4)
            
    p0 = 2e-5
    Lpred_eq = 10*np.log(1/len(p_pred) * sum(p_pred**2) / p0**2)
    Lref_eq = 10*np.log(1/len(p_ref) * sum(p_ref**2) / p0**2)
    Leqerr = abs(Lpred_eq - Lref_eq)
    
    pmax = np.abs(np.max(abs(p_ref)))
    Lrmse = np.sqrt(1/len(p_pred) * sum((p_pred/pmax - p_ref/pmax)**2))
    Lrmse_rel = Lrmse/np.sqrt(1/len(p_pred) * sum((p_ref/pmax)**2))

    error_l1 = np.linalg.norm(p_pred.flatten() - p_ref)
    
    f.write('---------------')
    f.write(f'(src,rec) = ({x0},{np.round(r0,3)})\n\n')
    
    f.write(f'l1 error: {error_l1}\n')
    f.write(f'Relative l1 error: {error_l1 / np.linalg.norm(p_ref)}\n')    
    f.write(f'Mean/max err: {mean_err} / {np.round(max(err_L1),3)}\n')
    f.write(f'Relative err: {np.round(mean_err_rel*100,1)}% / {np.round(20*np.log10(1 - mean_err_rel), 1)} dB\n')
    f.write(f'Equivalent SPL err: {np.round(Leqerr,3)} dB\n')
    f.write(f'RMSE: = {np.round(Lrmse,3)}\n')
    f.write(f'Relative RMSE: {np.round(Lrmse_rel,3)}\n')
    f.write('---------------')

    return mean_err, mean_err_rel, err_L1, err_rel

def resampleICs1D(grid,p,dx,xminmax):
    xlen = xminmax[1] - xminmax[0]
    x_interp = np.linspace(xminmax[0],xminmax[1], int(np.ceil(xlen/dx)))

    u_train = np.empty((p.shape[0],len(x_interp)), dtype=float)
    for i in range(p.shape[0]):
        p_i = np.array(p[i,0,:]) # srcs are initiated at time t=0 
        u_train[i,:] = griddata(grid, p_i, x_interp, method='cubic')

    assert(not np.isnan(u_train).any())

    return u_train

def resampleICs2D(grid,p,dx,xminmax,yminmax):
    xlen = xminmax[1] - xminmax[0]
    ylen = yminmax[1] - yminmax[0]
    x = np.linspace(xminmax[0],xminmax[1], int(np.ceil(xlen/dx)))
    y = np.linspace(yminmax[0],yminmax[1], int(np.ceil(ylen/dx)))
    XX, YY = np.meshgrid(x,y)

    x_interp = XX.flatten()
    y_interp = YY.flatten()

    u_train = np.empty((p.shape[0],len(x_interp)), dtype=float)
    for i in range(p.shape[0]):
        p_i = np.array(p[i,0,:]) # srcs are initiated at time t=0 
        u_train[i,:] = griddata(grid, p_i, (x_interp,y_interp), method='cubic')

    assert(not np.isnan(u_train).any())    

    return u_train, XX, YY

def tf_to_jax(x_tf):
    return x_tf.numpy()
    #x_dl = tf.experimental.dlpack.to_dlpack(x_tf)
    #return jax.dlpack.from_dlpack(x_dl)

# https://github.com/google/jax/discussions/10141
def toJaxBatch(batch_tf):
    u, y = map(tf_to_jax, batch_tf[0])
    outputs, x0 = tf_to_jax(batch_tf[1]), tf_to_jax(batch_tf[2])
    return (u,y), outputs, x0

def printInfo(dataset: IData, dataset_val: IData, batch_size_coord: int, batch_size: int):
    batch_size_train = min(batch_size, dataset.N)
    batch_size_val = min(batch_size, dataset_val.N)

    print(f"Mesh shape: {dataset.mesh.shape}")
    print(f"Time steps: {len(dataset.tsteps)}")
    print(f"IC shape: {dataset.u_shape}")

    print(f"Train data size: {dataset.P}")
    print(f"Train batch size (total): {batch_size_coord*batch_size_train}")
    print(f"Train num datasets: {dataset.N}")

    print(f"Val data size: {dataset_val.P}")
    print(f"Val batch size (total): {batch_size_coord*batch_size_val}")
    print(f"Val num datasets: {dataset_val.N}")