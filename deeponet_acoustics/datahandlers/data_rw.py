# ==============================================================================
# Copyright 2025 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import os
import h5py
from pathlib import Path
import numpy as np
from deeponet_acoustics.models.datastructures import Domain, Physics, SimulationData
import deeponet_acoustics.setup.parsers as parsers

def loadSimulationParametersJson(path_data):
    param_dict = parsers.parseSettings(path_data)
    fmax = param_dict['SimulationParameters']['fmax']
    c = param_dict['SimulationParameters']['c']
    dt = param_dict['SimulationParameters']['dt']
    rho = param_dict['SimulationParameters']['rho']
    
    return Physics(fmax, c, c, rho, dt)

def loadAttrFromH5(path_data):
        """ Load attributes from simulation data
            https://www.pythonforthelab.com/blog/how-to-use-hdf5-files-in-python/
        """

        with h5py.File(path_data, 'r') as f:
            settings_dict = {}
            
            settings_dict['dt'] = f['/pressures'].attrs['dt'][0]
            settings_dict['dx'] = f['/pressures'].attrs['dx'][0]
            settings_dict['c'] = f['/pressures'].attrs['c'][0]
            settings_dict['c_phys'] = f['/pressures'].attrs['c_phys'][0]
            settings_dict['rho'] = f['/pressures'].attrs['rho'][0]
            settings_dict['tmax'] = f['/pressures'].attrs['tmax'][0]
            settings_dict['sigma0'] = f['/pressures'].attrs['sigma0'][0]
            settings_dict['fmax'] = f['/pressures'].attrs['fmax'][0]
            settings_dict['tmax'] = f['/pressures'].attrs['tmax'][0]

            settings_dict['domain_minmax'] = f['/mesh'].attrs['domain_minmax']
            settings_dict['boundary_type'] = f['/mesh'].attrs['boundary_type'][0] if 'boundary_type' in f else ''

        return settings_dict
     
def loadDataFromH5(path_data, tmax=None):
    """ input
            tmax: normalized max time 
        output
            grid: is a x X y x t dimensional array
    """

    with h5py.File(path_data, 'r') as f:
        mesh = np.asarray(f['mesh'][()])
        umesh = np.asarray(f['umesh'][()])
        ushape = f['umesh'].attrs['umesh_shape']

        p = np.asarray(f['pressures'][()])
        up = np.asarray(f['upressures'][()])

        conn = np.asarray(f['conn'][()]) if 'conn' in f else np.asarray([])
        if 'x0_srcs' in f:
            x0_srcs = np.asarray(f['x0_srcs'][()])
        else:
            print("WARNING: x0_srcs not found in dataset")
            x0_srcs = np.asarray([])
            
        if len(x0_srcs) > 0 and len(x0_srcs.shape) == 3:
            # 3D data was written wrongly at some point, remove this when all data is correct
            x0_srcs = x0_srcs[:,0,:] if x0_srcs.shape[1] == 1 else x0_srcs
            x0_srcs = x0_srcs.T if x0_srcs.shape[0] != up.shape[0] else x0_srcs

        dt = f['pressures'].attrs['dt'][0]
        t = np.asarray(f['t'][()])

        if tmax == None:
            ilast = len(t) - 1
        else:
            ilist = [i for i, n in enumerate(t) if abs(n - tmax) <= dt/2]
            if not ilist:
                raise Exception(f'tmax {tmax} exceeds simulation data running time {t[-1]}')
            ilast = ilist[0]            
            # crop w.r.t. time
            t = np.array(t[:ilast+1])

        if len(p.shape) == 2: # data is written differently in Matlab and C++/Python
            p = np.array([p[:ilast+1,:]])
        else:
            p = np.array(p[:,:ilast+1,:])

        assert p.shape[2] == mesh.shape[0], f"p.shape: {p.shape}, mesh.shape {mesh.shape}"

        data = SimulationData(mesh, umesh, ushape, p, up, t, conn, x0_srcs, mesh.shape[1])

    return data

def writeDataToHDF5(grid, p, domain: Domain, physics: Physics, path_file: str, v=None):
    """ Write data as HDF5 format
    """
    """ Ex: writeHDF5([1,2,3],[0.1,0.2,0.3],data,[-0.2,0.0,0.2],0.1,1.0,343,0.1,2000,'test1.h5')
    """
    x0_sources = domain.x0_sources
    dt = domain.dt
    dx = domain.dx
    tmax = domain.tmax

    c = physics.c
    rho = physics.rho
    sigma0 = physics.sigma0
    fmax = physics.fmax    

    assert(len(p) == len(x0_sources))
    assert(len(p) == len(v))
    
    if Path(path_file).exists():
        os.remove(path_file)

    with h5py.File(path_file, 'w') as f:
        f.create_dataset('srcs', data=x0_sources)
        f.create_dataset('xx', data=grid[0])
        f.create_dataset('tt', data=grid[1])
        f.create_dataset('p', data=p)
        if v != None:
            f.create_dataset('v', data=v)
        
        # wrap in array to be consistent with data generated from Matlab
        f.attrs['dt'] = [dt]
        f.attrs['dx'] = [dx]
        f.attrs['c'] = [c]
        f.attrs['rho'] = [rho]
        f.attrs['sigma0'] = [sigma0]
        f.attrs['fmax'] = [fmax]
        f.attrs['tmax'] = [tmax]
        
        print(list(f.keys()))
        print(list(f.attrs))