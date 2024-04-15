# ==============================================================================
# Copyright 2023 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import sys
import jax.numpy as jnp
from jax import random, vmap, jit
from functools import partial
from torch.utils import data
import h5py
import numpy as np
from typing import Callable, Dict
from pathlib import Path
import itertools
import time
from datahandlers.io import XdmfReader
from models.datastructures import SimulationDataType
import datahandlers.io as IO

IC_NORM = True
AMPLITUDE = 2 # HACK: pressure min/max hardcoded

def normalizeFourierDataExpansionZero(data, data_nonfeat_dim, ymin=-1, ymax=1):
    # used for normalizing cos/sin domain from [-1,1] to [0,1] (for relu activation function)    
    data_nonfeat = data[:, 0:data_nonfeat_dim]

    data_feat_norm = normalizeData(data[:, data_nonfeat_dim::], ymin, ymax, from_zero=True)
    
    return np.hstack((data_nonfeat,data_feat_norm))
    
def normalizeData(data, ymin, ymax, from_zero=False):
    if from_zero:
        return (data - ymin)/(ymax - ymin)
    else:
        return 2*(data - ymin)/(ymax - ymin) - 1

def normalizeDomain(data, ymin, ymax, from_zero=False):
    spatial = normalizeData(data[..., 0:-1], ymin, ymax, from_zero)
    if from_zero:
        temp = np.expand_dims(data[..., -1]/(ymax - ymin), [1])
    else:
        temp = np.expand_dims(data[..., -1]/((ymax - ymin)/2), [1])
        
    return np.hstack((spatial,temp))
    
# def denormalizeDomain(data, ymin, ymax, from_zero=False):
#     spatial = (data[:, 0:-1] + 1)/2*(ymax - ymin) + ymin
#     temp = data[:, -1]*2*(ymax - ymin)
#     return np.hstack((spatial,temp))

# Data generator
class DataGenerator(data.Dataset):
    def __init__(self, u, y, s,                  
                 batch_size_branch, 
                 batch_size_coord,
                 u_src = [],
                 rng_key=random.PRNGKey(1234)):
        ''' 
            Batching is done along samples N (e.g. initial conditions)

            Input dimensions:

            u: (N, m)
            y: (P, dim)
            s: (N, P)
        '''
        self.u = jnp.asarray(u)
        self.y = jnp.asarray(y)
        self.s = jnp.asarray(s)

        self.u_src = jnp.asarray(u_src)
        
        self.N = self.u.shape[0]
        self.N_src = self.u_src.shape[0]
        self.P = self.y.shape[0]
        
        N_min = min(self.N_src, self.N) if self.N_src > 0 else self.N
        self.batch_size_branch = batch_size_branch if batch_size_branch <= N_min else N_min        
        self.batch_size_coord = batch_size_coord if batch_size_coord > 0 and batch_size_coord <= self.P else self.P
        self.key = rng_key
    
    def __getitem__(self, index):
        'Generate one batch of data'        

        self.key, key1, key2, key3 = random.split(self.key, 4)
        inputs, outputs, idx_coord = self.__data_generation(key1,key2,key3)
        return inputs, outputs, idx_coord

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key1, key2, key3):
        'Generates data containing batch_size samples'

        # sample functions randomly for each batch
        idx_funcs = random.choice(key1, self.N, (self.batch_size_branch,), replace=False)        

        # sample coordinates randomly for each function
        assert self.batch_size_coord > 0, "batch size for coordinate should be larger than 0"
        idx_coord = random.choice(key2, self.P, (self.batch_size_branch,self.batch_size_coord), replace=False)
                
        y = vmap(lambda idx: self.y[idx,:], (1))(idx_coord.T) # random samples for each sampled function batch
        s = self.s[idx_funcs[:,None],idx_coord]               # random samples for each sampled function batch
        u = self.u[idx_funcs,:]                               # sampled function batch        

        if len(self.u_src) > 0:  # source function batch (if set)
            idx_funcs_src = random.choice(key3, self.N_src, (self.batch_size_branch,), replace=False)
            u_src = self.u_src[idx_funcs_src,:]
        else:
            u_src = self.u_src

        # Construct batch
        inputs = (u, y, u_src)
        outputs = s
        return inputs, outputs, idx_coord

class IData():
    simulationDataType: SimulationDataType
    datasets: list[h5py.File]
    mesh: list
    tags_field: list[str]
    tag_ufield: str
    tt: list[float]
    data_prune: int

class DataXdmf(IData):
    simulationDataType: SimulationDataType = SimulationDataType.XDMF

    datasets: list[h5py.File]
    mesh: list
    tags_field: list[str]
    tag_ufield: str
    tt: list[float]
    data_prune: int
    N: int
    P: int
    Pmesh: int
    
    xmin: float
    xmax: float
    normalize_data: bool

    # MAXNUM_DATASETS: SET TO E.G: 500 WHEN DEBUGGING ON MACHINES WITH LESS RESOURCES
    def __init__(self, data_path, tmax=float('inf'), t_norm=1, flatten_ic=True, data_prune=1, norm_data=False, MAXNUM_DATASETS=sys.maxsize):
        filenames_xdmf = IO.pathsToFileType(data_path, '.xdmf', exclude='rectilinear')
        self.normalize_data = norm_data

        # NOTE: we assume meshes, tags, etc are the same accross all xdmf datasets
        xdmf = XdmfReader(filenames_xdmf[0], tmax=tmax/t_norm)
        self.tags_field = xdmf.tags_field
        self.tag_ufield = xdmf.tag_ufield
        self.data_prune = data_prune
        self.tsteps = xdmf.tsteps*t_norm

        with h5py.File(xdmf.filenameH5) as r:
            self.mesh = np.array(r[xdmf.tag_mesh][::self.data_prune])
            self.xmin, self.xmax = np.min(self.mesh), np.max(self.mesh)            
            umesh_obj = r[xdmf.tag_umesh]
            umesh = np.array(umesh_obj[:])
            self.u_shape = [len(umesh)] if flatten_ic else jnp.array(umesh_obj.attrs[xdmf.tag_ushape][:], dtype=int).tolist()
                
        if norm_data:
            self.mesh = self.normalizeSpatial(self.mesh)
            self.tsteps = self.normalizeTemporal(self.tsteps)

        self.tt = np.repeat(self.tsteps, self.mesh.shape[0])
        self.num_tsteps = len(self.tsteps)

        self.Pmesh = self.mesh.shape[0]
        self.P = self.Pmesh * self.tsteps.shape[0] # total number of time/space points        

        self.datasets = []
        for i in range(0, min(MAXNUM_DATASETS, len(filenames_xdmf))):
            filename = filenames_xdmf[i]
            if Path(filename).exists():
                xdmf = XdmfReader(filename, tmax/t_norm)
                self.datasets.append(h5py.File(xdmf.filenameH5, 'r')) # add file handles and keeps open
            else:
                print(f"Could not be found (ignoring): {filename}")

        self.N = len(self.datasets)

    def normalizeSpatial(self, data):
        return 2*(data - self.xmin)/(self.xmax - self.xmin) - 1
    
    def normalizeTemporal(self, data):
        return data/(self.xmax - self.xmin)/2

    def __del__(self):
        for (_, dataset) in enumerate(self.datasets):
            dataset.close()

def getNumberOfSources(data_path: str):
    return len(IO.pathsToFileType(data_path, '.h5', exclude='rectilinear'))

class DataH5Compact(IData):
    simulationDataType: SimulationDataType = SimulationDataType.H5COMPACT

    tag_ufield: str    
    data_prune: int    
    N: int # number of sources
    P: int
    Pmesh: int
    
    xmin: float
    xmax: float
    normalize_data: bool
    
    datasets: list[h5py.File] = []
    mesh: list = []
    conn: list = []
    tsteps: list[float] = []
    tt: list[float] = []
    tags_field: list[str] = []

    # MAXNUM_DATASETS: SET TO E.G: 500 WHEN DEBUGGING ON MACHINES WITH LESS RESOURCES
    def __init__(self, data_path, tmax=float('inf'), t_norm=1, flatten_ic=True, data_prune=1, norm_data=False, MAXNUM_DATASETS=sys.maxsize):
        filenamesH5 = IO.pathsToFileType(data_path, '.h5', exclude='rectilinear')
        self.data_prune = data_prune
        self.normalize_data = norm_data

        # NOTE: we assume meshes, tags, etc are the same accross all xdmf datasets
        tag_mesh = "/mesh"
        tag_conn = "/conn"
        tag_umesh = "/umesh"
        tag_ushape = "umesh_shape"
        self.tags_field = ["/pressures"]
        self.tag_ufield = "/upressures"             

        with h5py.File(filenamesH5[0]) as r:
            self.mesh = np.array(r[tag_mesh][::self.data_prune])
            self.conn = np.array(r[tag_conn]) if self.data_prune == 1 and tag_conn in r else np.array([])
            self.xmin, self.xmax = np.min(self.mesh), np.max(self.mesh)
            
            umesh_obj = r[tag_umesh]
            umesh = np.array(umesh_obj[:])
            self.u_shape = jnp.array([len(umesh)], dtype=int) if flatten_ic else jnp.array(umesh_obj.attrs[tag_ushape][:], dtype=int)
            self.tsteps = r[self.tags_field[0]].attrs['time_steps']
            self.tsteps = jnp.array([t for t in self.tsteps if t <= tmax/t_norm])
            self.tsteps = self.tsteps*t_norm

            if self.normalize_data:
                self.mesh = self.normalizeSpatial(self.mesh)
                self.tsteps = self.normalizeTemporal(self.tsteps)
                    
        self.tt = np.repeat(self.tsteps, self.mesh.shape[0])
        self.num_tsteps = len(self.tsteps)
        
        self.Pmesh = self.mesh.shape[0]
        self.P = self.Pmesh * self.tsteps.shape[0] # total number of time/space points
        self.N = len(filenamesH5)

        self.datasets = []
        for i in range(0, min(MAXNUM_DATASETS, len(filenamesH5))):
            filename = filenamesH5[i]
            if Path(filename).exists():
                self.datasets.append(h5py.File(filename, 'r')) # add file handles and keeps open
            else:
                print(f"Could not be found (ignoring): {filename}")

    @property
    def xxyyzztt(self):
        xxyyzz = np.tile(self.mesh, (len(self.tsteps), 1))
        return np.hstack((xxyyzz, self.tt.reshape(-1,1)))

    def normalizeSpatial(self, data):
        return 2*(data - self.xmin)/(self.xmax - self.xmin) - 1
    
    def normalizeTemporal(self, data):
        return data/(self.xmax - self.xmin)/2
    
    def denormalizeSpatial(self, data):
        return (data + 1)/2*(self.xmax - self.xmin) + self.xmin

    def denormalizeTemporal(self, data):
        return data*2*(self.xmax - self.xmin)

    def __del__(self):
        for (_, dataset) in enumerate(self.datasets):
            dataset.close()

class DatasetH5Mock:
    dict: Dict

    def __init__(self, dict):
        self.dict = dict

    def __getitem__(self, item):
        return self.dict[item]
    
    def __contains__(self, key):
        return key in self.dict

    def close(self):
        # mocking HdF5 close method
        pass
    
class DataSourceOnly(IData):
    simulationDataType: SimulationDataType = SimulationDataType.SOURCE_ONLY
    
    N: int # number of sources
    P: int
    Pmesh: int
    
    xmin: float
    xmax: float
    normalize_data: bool
    
    datasets: list[h5py.File] = []
    mesh: list = []
    conn: list = []
    tsteps: list[float] = []
    tt: list[float] = []
    tags_field: list[str] = []

    # MAXNUM_DATASETS: SET TO E.G: 500 WHEN DEBUGGING ON MACHINES WITH LESS RESOURCES
    def __init__(self, data_path, source_pos, params, tmax=float('inf'), t_norm=1, flatten_ic=True, data_prune=1, norm_data=False):
        filenamesH5 = IO.pathsToFileType(data_path, '.h5', exclude='rectilinear')
        self.data_prune = data_prune
        self.normalize_data = norm_data        

        # NOTE: we assume meshes, tags, etc are the same accross all xdmf datasets
        tag_mesh = "/mesh"
        tag_conn = "/conn"
        tag_umesh = "/umesh"
        tag_ushape = "umesh_shape"
        self.tags_field = ["/pressures"]
        self.tag_ufield = "/upressures"             

        with h5py.File(filenamesH5[0]) as r:
            self.mesh = np.array(r[tag_mesh][::self.data_prune])
            self.conn = np.array(r[tag_conn]) if self.data_prune == 1 and tag_conn in r else np.array([])
            self.xmin, self.xmax = np.min(self.mesh), np.max(self.mesh)
            
            umesh_obj = r[tag_umesh]
            self.u_shape = jnp.array([len(umesh_obj[:])], dtype=int) if flatten_ic else jnp.array(umesh_obj.attrs[tag_ushape][:], dtype=int)
            self.tsteps = r[self.tags_field[0]].attrs['time_steps']
            self.tsteps = jnp.array([t for t in self.tsteps if t <= tmax/t_norm])
            self.tsteps = self.tsteps*t_norm

            if self.normalize_data:
                self.mesh = self.normalizeSpatial(self.mesh)
                self.tsteps = self.normalizeTemporal(self.tsteps)
            
            gaussianSrc = lambda x, y, z, xyz0, sigma, ampl: \
            ampl*np.exp(-((x - xyz0[0])**2 + (y - xyz0[1])**2 + (z - xyz0[2])**2)/sigma**2)
        
            self.datasets = [] # todo: preload self.N
            sigma0 = params.c / (np.pi * params.fmax / 2)

            self.N = len(source_pos)

            for i in range(self.N):
                x0 = source_pos[i]                        
                ic_field = gaussianSrc(umesh_obj[:,0], umesh_obj[:,1], umesh_obj[:,2], x0, sigma0, AMPLITUDE)
                # mimic H5 file handle but only for loading source
                self.datasets.append(DatasetH5Mock({self.tag_ufield: ic_field, 'source_position': x0}))
                    
        self.tt = np.repeat(self.tsteps, self.mesh.shape[0])
        self.num_tsteps = len(self.tsteps)
        
        self.Pmesh = self.mesh.shape[0]
        self.P = self.Pmesh * self.tsteps.shape[0] # total number of time/space points        

    @property
    def xxyyzztt(self):
        xxyyzz = np.tile(self.mesh, (len(self.tsteps), 1))
        return np.hstack((xxyyzz, self.tt.reshape(-1,1)))

    def normalizeSpatial(self, data):
        return 2*(data - self.xmin)/(self.xmax - self.xmin) - 1
    
    def normalizeTemporal(self, data):
        return data/(self.xmax - self.xmin)/2
    
    def denormalizeSpatial(self, data):
        return (data + 1)/2*(self.xmax - self.xmin) + self.xmin

    def denormalizeTemporal(self, data):
        return data*2*(self.xmax - self.xmin)

    def __del__(self):
        for (_, dataset) in enumerate(self.datasets):
            dataset.close()





class DatasetStreamer(IData):  
    dim_input: int
    Pmesh: int
    P: int
    batch_size_coord: int

    data: IData

    itercount: itertools.count

    pmin: float = -AMPLITUDE # HACK: pressure min/max hardcoded
    pmax: float = AMPLITUDE  # HACK: pressure min/max hardcoded
    
    __y_feat_extract_fn = Callable[[list],list]

    total_time_0 = 0
    total_time_1 = 0

    @property
    def N(self):
        return self.data.N

    def __init__(self, data, batch_size_coord=-1, y_feat_extractor=None):
        # batch_size_coord: set to -1 if full dataset should be used (e.g. for validation data)
        
        self.data = data

        self.Pmesh = data.mesh.shape[0]        
        self.P = self.Pmesh * data.tsteps.shape[0] # total number of time/space points
        
        self.batch_size_coord = batch_size_coord if batch_size_coord <= self.P else self.P
        self.__y_feat_extract_fn = (lambda y: y) if y_feat_extractor == None else y_feat_extractor

        self.dim_input = self.__y_feat_extract_fn(jnp.array([[0,0,0,0]])).shape[1]
        self.itercount = itertools.count()
        self.rng = np.random.default_rng()
    
    def __len__(self):
        return len(self.data.datasets)

    def __getitem__(self, idx):        
        dataset = self.data.datasets[idx]
        u_norm = 2*(dataset[self.data.tag_ufield][:] - self.pmin)/(self.pmax-self.pmin)-1 if IC_NORM else dataset[self.data.tag_ufield][:] # [-1,1]
        u = jnp.reshape(u_norm, self.data.u_shape)

        start_time_0 = time.perf_counter()
        if self.batch_size_coord > 0:
            indxs_coord  = self.rng.choice(self.P, (self.batch_size_coord), replace=False)
        else:
            indxs_coord = jnp.arange(0,self.P)
        end_time_0 = time.perf_counter()
        self.total_time_0 += end_time_0 - start_time_0

        xxyyzz = self.data.mesh[np.mod(indxs_coord,self.data.Pmesh),:]
        tt = self.data.tt[indxs_coord].reshape(-1,1)
        y = self.__y_feat_extract_fn(np.hstack((xxyyzz,tt)))
        
        # collect all field data for all timesteps - might be memory consuming
        # If memory load gets too heavy, consider selecting points at each timestep
        if self.data.simulationDataType == SimulationDataType.H5COMPACT:
            s = dataset[self.data.tags_field[0]][0:self.data.num_tsteps,::self.data.data_prune].flatten()[indxs_coord]
        elif self.data.simulationDataType == SimulationDataType.XDMF:
            s = np.empty((self.P), dtype=jnp.float32)
            for j in range(self.data.num_tsteps):
                s[j*self.data.Pmesh:(j+1)*self.Pmesh] = dataset[self.data.tags_field[j]][::self.data.data_prune]
            s = s[indxs_coord]
        elif self.data.simulationDataType == SimulationDataType.SOURCE_ONLY:
            s = []
        else:
            raise Exception('Data format unknown: should be H5COMPACT or XDMF')

        # normalize
        x0 = self.data.normalizeSpatial(dataset['source_position'][:]) if 'source_position' in dataset else []    
        
        inputs = jnp.asarray(u), jnp.asarray(y)
        return inputs, jnp.asarray(s), indxs_coord, x0

def numpy_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return jnp.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return np.array(batch)

class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)
    
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