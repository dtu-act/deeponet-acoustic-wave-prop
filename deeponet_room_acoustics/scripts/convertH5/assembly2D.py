# ==============================================================================
# Copyright 2025 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
#
# 2D: When running MATLAB in parallel with multiple threads, each source position is written to separate files.
#     We need to assemble the data into one file for the 2D Python code to process the data. Call this script 
#     before converting resolutions
# ==============================================================================
import h5py
import numpy as np
import deeponet_room_acoustics.datahandlers.io as IO
from pathlib import Path

def assembleH5(data_dir, header_filepath_in, filepath_out):
    filenames_h5 = IO.pathsToFileType(data_dir, '.h5', exclude='header')
    N_srcs = len(filenames_h5)

    if N_srcs == 0:
        raise Exception('No files found')

    # get data sizes
    with h5py.File(filenames_h5[0], 'r')  as f:
        shape_p = (N_srcs, f['pressures'][()].shape[0], f['pressures'][()].shape[1])
        shape_up = (N_srcs,f['upressures'][()].shape[0])
        shape_srcs = (N_srcs,f['x0_srcs'][()].shape[1])

    with h5py.File(filepath_out, 'w') as fw:
        # TODO: https://stackoverflow.com/questions/47072859/how-to-append-data-to-one-specific-dataset-in-a-hdf5-file-with-h5py
        pressure_new = fw.create_dataset('pressures', chunks=True, shape=shape_p, dtype=np.float32)
        upressure_new = fw.create_dataset('upressures', chunks=True, shape=shape_up, dtype=np.float32)
        x0_srcs_new = fw.create_dataset('x0_srcs', chunks=True, shape=shape_srcs, dtype=np.float32)

        with h5py.File(header_filepath_in, 'r')  as f:                        
            conn = f['conn']            
            time_steps = f['t']            

            # umesh
            umesh = f['umesh']
            # umesh attributes
            dx_u = f['umesh'].attrs['dx']
            ushape = f['umesh'].attrs['umesh_shape']
            dx_u = f['umesh'].attrs('dx')

            # mesh
            mesh = f['mesh']
            # mesh attributes
            dt = mesh.attrs['dt']
            dx = mesh.attrs['dx']
            dx_src = mesh.attrs['dx_src'] if 'dx_src' in mesh.attrs else [] # only available in newer datasets 
            c = mesh.attrs['c']
            c_phys = mesh.attrs['c_phys']
            rho = mesh.attrs['rho']
            sigma0 = mesh.attrs['sigma0']
            fmax = mesh.attrs['fmax']
            tmax = mesh.attrs['tmax']
            domain_minmax = mesh.attrs['domain_minmax']
            #boundary_type = mesh.attrs['boundary_type'][0]       
            
            # create dataset            
            fw.create_dataset('conn', data=conn, dtype=np.float32)
            fw.create_dataset('t', data=time_steps, dtype=np.float32)     

            mesh_new = fw.create_dataset('mesh', data=mesh, dtype=np.float32)
            mesh_new.attrs.create('dt', dt, dtype=np.float32)
            mesh_new.attrs.create('dx', dx, dtype=np.float32)
            mesh_new.attrs.create('dx_src', dx_src, dtype=np.float32)
            mesh_new.attrs.create('c', c, dtype=np.float32)
            mesh_new.attrs.create('c_phys', c_phys, dtype=np.float32)
            mesh_new.attrs.create('rho', rho, dtype=np.float32)
            mesh_new.attrs.create('sigma0', sigma0, dtype=np.float32)
            mesh_new.attrs.create('fmax', fmax, dtype=np.float32)
            mesh_new.attrs.create('tmax', tmax, dtype=np.float32)
            mesh_new.attrs.create('domain_minmax', domain_minmax, dtype=np.float32)
            #mesh_new.attrs.create('boundary_type', boundary_type, dtype=str)

            umesh_new = fw.create_dataset('umesh', data=umesh, dtype=np.float32)
            umesh_new.attrs.create('dx', dx_u, dtype=np.float32)
            umesh_new.attrs.create('umesh_shape', ushape, dtype=np.float32)                   
        
        for i,filename in enumerate(filenames_h5):
            if not Path(filename).exists():
                raise Exception(f"File could not be found: {filename}")
            
            with h5py.File(filename, 'r')  as f:
                pressures = f['pressures']
                upressures = f['upressures']
                x0_srcs = f['x0_srcs']

                # append to preallocate dset
                pressure_new[i,:,:] = pressures[()]
                upressure_new[i,:] = upressures[()].flatten()
                x0_srcs_new[i,:] = x0_srcs[()] 
                
    
