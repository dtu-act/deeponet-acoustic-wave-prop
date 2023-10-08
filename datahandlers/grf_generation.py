# ==============================================================================
# Copyright 2023 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
from pydoc import doc
import numpy as np
import matplotlib.pyplot as plt

def generate_grf_function_mask_1d(xmin, xmax, sigma_0, l_0, num_of_elements_x, num_of_samples, sigma0_window=0.2):
    if len(sigma_0) != len(l_0):
        raise ValueError('Size of sigma_0 and l_0 mismatch')
    
    num_of_nodes = num_of_elements_x + 1
    
    x = np.linspace(xmin, xmax, num_of_nodes)
    dx = x[2]-x[1]

    x1, x2 = np.meshgrid(x, x, indexing='ij')
    distances_squared  = ((x1 - x2) ** 2)
    covariance_matrix = np.zeros((num_of_nodes, num_of_nodes))
    for mode_index, corr_length in enumerate(l_0):
        # normalization factor changed from original code -> (sigma_0[mode_index] ** 2)
        covariance_matrix += (1/(sigma_0[mode_index]*np.sqrt(2*np.pi)) *
                               np.exp(- 0.5 / (corr_length ** 2) *
                                      distances_squared))

    mu = np.zeros_like(x)
    samples = np.random.multivariate_normal(mu, covariance_matrix, num_of_samples)
    samples = samples.reshape([-1, num_of_nodes])
    
    offset = sigma0_window*3
    x_half = np.arange(0,offset+1e-10, dx)
    gauss_half = np.exp(-(x_half/sigma0_window)**2)
    mask = np.ones(len(samples[0]))
    mask[:len(x_half)] = gauss_half[::-1]
    mask[-len(x_half):] = gauss_half
    
    samples_masked = samples*mask
    normalizePerSample(samples_masked)    

    gaussian_compare = np.exp(-(x/sigma0_window)**2)
    gaussian_compare = np.expand_dims(gaussian_compare, axis=0)

    return samples, samples_masked, x, mask, gaussian_compare

def generate_grf_function_2d(sigma_0, l_0, num_of_elements_x, num_of_elements_y, num_of_samples):
    if len(sigma_0) != len(l_0):
        raise ValueError('Size of sigma_0 and l_0 mismatch')
    num_of_nodes_x = num_of_elements_x + 1
    num_of_nodes_y = num_of_elements_y + 1
    num_of_nodes = num_of_nodes_x * num_of_nodes_y
    x_1d = np.linspace(0, 1, num_of_nodes_x)
    y_1d = np.linspace(0, 1, num_of_nodes_y)
    x2d, y2d = np.meshgrid(x_1d, y_1d, indexing='ij')
    x, y = x2d.flatten(), y2d.flatten()
    x1, x2 = np.meshgrid(x, x, indexing='ij')
    y1, y2 = np.meshgrid(y, y, indexing='ij')
    distances_squared  = ((x1 - x2) ** 2 + (y1 - y2) ** 2)
    covariance_matrix = np.zeros((num_of_nodes, num_of_nodes))
    for mode_index, corr_length in enumerate(l_0):
        covariance_matrix += ((sigma_0[mode_index] ** 2) *
                               np.exp(- 0.5 / (corr_length ** 2) *
                                      distances_squared))
    mu = np.zeros_like(x)
    samples = np.random.multivariate_normal(mu, covariance_matrix, num_of_samples)
    return samples.reshape([-1, num_of_nodes_x, num_of_nodes_y]), x2d,y2d

def generate_grf_function_mask_2d(xminmax, yminmax, sigma_0, l_0, x1d, y1d, num_of_samples, sigma0_window):
    if len(sigma_0) != len(l_0):
        raise ValueError('Size of sigma_0 and l_0 mismatch')
    if len(x1d) != len(y1d):
        raise ValueError('x, y size mismatch')

    num_of_nodes = len(x1d)
    x1, x2 = np.meshgrid(x1d, x1d, indexing='ij')
    y1, y2 = np.meshgrid(y1d, y1d, indexing='ij')
    distances_squared  = ((x1 - x2) ** 2 + (y1 - y2) ** 2)
    covariance_matrix = np.zeros((num_of_nodes, num_of_nodes))
    for mode_index, corr_length in enumerate(l_0):
        covariance_matrix += ((sigma_0[mode_index] ** 2) *
                               np.exp(- 0.5 / (corr_length ** 2) *
                                      distances_squared))
    mu = np.zeros_like(x1d)
    samples = np.random.multivariate_normal(mu, covariance_matrix, num_of_samples)
    
    offset = sigma0_window*3
    x0 = xminmax[0] + offset
    y0 = yminmax[0] + offset
    xhat0 = xminmax[1] - offset
    yhat0 = yminmax[1] - offset

    gauss_fn = lambda x,y,x0,y0: np.exp(-(((x-x0)**2 + (y-y0)**2)/sigma0_window**2))

    def mask_fun(x,y):
        # edges
        if x <= x0 and (y >= y0 and y <= yhat0):
            return gauss_fn(x,0,x0,0)
        elif x >= xhat0 and (y >= y0 and y <= yhat0):
            return gauss_fn(x,0,xhat0,0)        
        elif y <= y0 and (x >= x0 and x <= xhat0):
            return gauss_fn(0,y,0,y0)        
        elif y >= yhat0 and (x >= x0 and x <= xhat0):
            return gauss_fn(0,y,0,yhat0)
        # corners
        elif x <= x0 and y <= y0:
            return gauss_fn(x,y,x0,y0)
        elif x >= xhat0 and y <= y0:
            return gauss_fn(x,y,xhat0,y0)
        elif x <= x0 and y >= yhat0:
            return gauss_fn(x,y,x0,yhat0)
        elif x >= xhat0 and y >= yhat0:
            return gauss_fn(x,y,xhat0,yhat0)
        else:
            return 1 # inside
    
    mask = np.empty(num_of_nodes)

    for i in range(len(x1d)):
        mask[i] = mask_fun(x1d[i],y1d[i])

    samples_masked = samples*mask
    normalizePerSample(samples_masked)

    return samples, samples_masked, mask

def generate_grf_function_mask_3d(xminmax, yminmax, zminmax, sigma_0, l_0, x1d, y1d, z1d, num_of_samples, sigma0_window=0.2):
    num_of_nodes = len(x1d)

    x1, x2 = np.meshgrid(x1d, x1d, indexing='ij')
    y1, y2 = np.meshgrid(y1d, y1d, indexing='ij')
    z1, z2 = np.meshgrid(z1d, z1d, indexing='ij')
    distances_squared  = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    covariance_matrix = np.zeros((num_of_nodes, num_of_nodes))

    # normalization factor changed from original code: 
    # From (sigma_0[i] ** 2) to (1/(sigma_0[mode_index]*np.sqrt(2*np.pi))
    covariance_matrix += (1/(sigma_0*np.sqrt(2*np.pi)) *
                            np.exp(- 0.5 / (l_0 ** 2) *
                                    distances_squared))
    mu = np.zeros_like(x1d)
    samples = np.random.multivariate_normal(mu, covariance_matrix, num_of_samples)
    
    offset = sigma0_window*3
    x0 = xminmax[0] + offset
    y0 = yminmax[0] + offset
    z0 = zminmax[0] + offset
    xhat0 = xminmax[1] - offset
    yhat0 = yminmax[1] - offset
    zhat0 = zminmax[1] - offset

    gauss_fn = lambda x,y,z,x0,y0,z0: np.exp(-(((x-x0)**2 + (y-y0)**2 + (z-z0)**2)/sigma0_window**2))

    def mask_fun(x,y,z):
        # 6 squares
        # x
        if x <= x0 and (y >= y0 and y <= yhat0) and (z >= z0 and z <= zhat0):
            return gauss_fn(x,0,0,x0,0,0)               
        elif x >= xhat0 and (y >= y0 and y <= yhat0) and (z >= z0 and z <= zhat0):
            return gauss_fn(x,0,0,xhat0,0,0)
        # y
        elif y <= y0 and (x >= x0 and x <= xhat0) and (z >= z0 and z <= zhat0):
            return gauss_fn(0,y,0,0,y0,0)
        elif y >= yhat0 and (x >= x0 and x <= xhat0) and (z >= z0 and z <= zhat0):
            return gauss_fn(0,y,0,0,yhat0,0)
        # z
        elif z <= z0 and (x >= x0 and x <= xhat0) and (y >= y0 and y <= yhat0):
            return gauss_fn(0,0,z,0,0,z0)
        elif z >= zhat0 and (x >= x0 and x <= xhat0) and (y >= y0 and y <= yhat0):
            return gauss_fn(0,0,z,0,0,zhat0)
        
        # 12 edges
        # along y-axis lower z-plane
        elif x <= x0 and (y >= y0 and y <= yhat0) and z <= z0:
            return gauss_fn(x,0,z,x0,0,z0)
        elif x >= xhat0 and (y >= y0 and y <= yhat0) and z <= z0:
            return gauss_fn(x,0,z,xhat0,0,z0)
        # along y-axis upper z-plane
        elif x <= x0 and (y >= y0 and y <= yhat0) and z >= zhat0:
            return gauss_fn(x,0,z,x0,0,zhat0)
        elif x >= xhat0 and (y >= y0 and y <= yhat0) and z >= zhat0:
            return gauss_fn(x,0,z,xhat0,0,zhat0)
        # along x-axis lower z-plane
        elif y <= y0 and (x >= x0 and x <= xhat0) and z <= z0:
            return gauss_fn(0,y,z,0,y0,z0)
        elif y >= yhat0 and (x >= x0 and x <= xhat0) and z <= z0:
            return gauss_fn(0,y,z,0,yhat0,z0)
        # along x-axis upper z-plane
        elif y <= y0 and (x >= x0 and x <= xhat0) and z >= zhat0:
            return gauss_fn(0,y,z,0,y0,zhat0)
        elif y >= yhat0 and (x >= x0 and x <= xhat0) and z >= zhat0:
            return gauss_fn(0,y,z,0,yhat0,zhat0)
        # along z-axis nearest y-plane
        elif x <= x0 and (z >= z0 and z <= zhat0) and y <= y0:
            return gauss_fn(x,y,0,x0,y0,0)
        elif x >= xhat0 and (z >= z0 and z <= xhat0) and y <= y0:
            return gauss_fn(x,y,0,xhat0,y0,0)
        # along z-axis furthest y-plane
        elif x <= x0 and (z >= z0 and z <= zhat0) and y >= yhat0:
            return gauss_fn(x,y,0,x0,yhat0,0)
        elif x >= xhat0 and (z >= z0 and z <= xhat0) and y >= yhat0:
            return gauss_fn(x,y,0,xhat0,yhat0,0)

        # 8 corners
        # lower z-plane
        elif x <= x0 and y <= y0 and z <= z0:
            return gauss_fn(x,y,z,x0,y0,z0)
        elif x >= xhat0 and y <= y0 and z <= z0:
            return gauss_fn(x,y,z,xhat0,y0,z0)
        elif x <= x0 and y >= yhat0 and z <= z0:
            return gauss_fn(x,y,z,x0,yhat0,z0)
        elif x >= xhat0 and y >= yhat0 and z <= z0:
            return gauss_fn(x,y,z,xhat0,yhat0,z0)
        # upper z-plane
        elif x <= x0 and y <= y0 and z >= z0:
            return gauss_fn(x,y,z,x0,y0,zhat0)
        elif x >= xhat0 and y <= y0 and z >= z0:
            return gauss_fn(x,y,z,xhat0,y0,zhat0)
        elif x <= x0 and y >= yhat0 and z >= z0:
            return gauss_fn(x,y,z,x0,yhat0,zhat0)
        elif x >= xhat0 and y >= yhat0 and z >= z0:
            return gauss_fn(x,y,z,xhat0,yhat0,zhat0)        
        else:
            return 1 # inside
    
    mask = np.empty(num_of_nodes)

    for i in range(len(x1d)):
        mask[i] = mask_fun(x1d[i],y1d[i], z1d[i])

    samples_masked = samples*mask
    normalizePerSample(samples_masked)

    return samples, samples_masked, mask

def generate_grf_function_3d(sigma_0, l_0, num_of_elements_x,
                             num_of_elements_y, num_of_elements_z, num_of_samples):
    if len(sigma_0) != len(l_0):
        raise ValueError('Size of sigma_0 and l_0 mismatch')
    num_of_nodes_x = num_of_elements_x + 1
    num_of_nodes_y = num_of_elements_y + 1
    num_of_nodes_z = num_of_elements_z + 1
    num_of_nodes = num_of_nodes_x * num_of_nodes_y * num_of_nodes_z
    x_1d = np.linspace(0, 1, num_of_nodes_x)
    y_1d = np.linspace(0, 1, num_of_nodes_y)
    z_1d = np.linspace(0, 1, num_of_nodes_z)
    x, y, z = np.meshgrid(x_1d, y_1d, z_1d, indexing='ij')
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    x1, x2 = np.meshgrid(x, x, indexing='ij')
    y1, y2 = np.meshgrid(y, y, indexing='ij')
    z1, z2 = np.meshgrid(z, z, indexing='ij')
    distances_squared  = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    covariance_matrix = np.zeros((num_of_nodes, num_of_nodes))
    
    for mode_index, corr_length in enumerate(l_0):
        covariance_matrix += ((sigma_0[mode_index] ** 2) *
                               np.exp(- 0.5 / (corr_length ** 2) *
                                      distances_squared))
    mu = np.zeros_like(x)
    samples = np.random.multivariate_normal(mu, covariance_matrix, num_of_samples)

    return samples.reshape([-1, num_of_nodes_x, num_of_nodes_y, num_of_nodes_z])

def normalizePerSample(samples):
    for i in range(len(samples)):
        if np.min(samples[i,...]) < -1:
            samples[i,...] = samples[i,...]/np.abs(np.min(samples[i,...]))
        if np.max(samples[i,...]) > 1:
            samples[i,...] = samples[i,...]/np.abs(np.max(samples[i,...]))