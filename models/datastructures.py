# ==============================================================================
# Copyright 2023 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
from dataclasses import dataclass # https://ber2.github.io/posts/dataclasses/
from enum import Enum
import os
from pathlib import Path
import shutil
from typing import Callable, List
import jax
import numpy as np
from flax import linen as nn           # The Linen API

class NetworkArchitectureType(Enum):    
    MLP = 1
    MOD_MLP = 2
    RESNET = 3

class SimulationDataType(Enum):
    H5COMPACT = 1
    XDMF = 2    

class BoundaryType(Enum):
    DIRICHLET = 1
    NEUMANN = 2
    IMPEDANCE_FREQ_DEP = 3
    IMPEDANCE_FREQ_INDEP = 4

class SourceType(Enum):
    IC = 1
    INJECTION = 2

@dataclass
class NetworkContainer:
    in_dim: List[float]
    network_type: NetworkArchitectureType
    network: nn.Module

    def __init__(self, network: nn.Module, in_dim):
        self.network_type = network.network_type
        self.network = network
        self.in_dim = in_dim

        if network.network_type == NetworkArchitectureType.MOD_MLP:
            print(network.tabulate(jax.random.PRNGKey(1234), 
                            np.expand_dims(np.ones(in_dim), [0]))
                )
        else:
            print(network.tabulate(np.random.PRNGKey(1234), np.expand_dims(np.ones(in_dim), [0,3] 
                                                                           if network.network_type == NetworkArchitectureType.RESNET else [0])))        

@dataclass
class SourceInfo:
    type: SourceType
    mu: float = 0
    sigma0: float = None
    source: Callable = None

    def __init__(self, type, sigma0: float, spatial_dim: int, src_fn: Callable):
        self.type = type
        self.sigma0 = sigma0
        self.source = src_fn(sigma0, spatial_dim)

@dataclass
class SimulationData:
    mesh: list[np.float32]
    umesh: list[np.float32]
    ushape: list[int]
    pressures: list[np.float32]
    upressures: list[np.float32]
    t: list[np.float32]
    conn: list[int]
    x0_srcs: list[float]
    dim: int # 1D, 2D, 3D 

@dataclass
class InputOutputDirs:
    id: str
    id_dir: str
    figs_dir: str
    models_dir: str
    plot_graph_path: str
    data_dir: str
    plot_graph_path: str
    training_data_path: str
    testing_data_path: str

    def __init__(self,settings_dict,input_dir=None,output_dir=None):        
        self.id = settings_dict['id']

        self.data_dir = settings_dict['input_dir'] if input_dir == None else input_dir
        
        if output_dir == None:
            output_dir = settings_dict['output_dir']
        self.id_dir = os.path.join(output_dir, self.id)

        self.figs_dir = os.path.join(self.id_dir, "figs")
        self.models_dir = os.path.join(self.id_dir, "models")
        self.training_data_path = os.path.join(self.data_dir, settings_dict['training_data_dir'])
        self.testing_data_path = os.path.join(self.data_dir, settings_dict['testing_data_dir'])
        self.plot_graph_path = os.path.join(self.models_dir, 'deeponet', 'network.png')

    def createDirs(self, delete_existing=False):
        if delete_existing and Path(self.id_dir).exists():
            shutil.rmtree(self.id_dir)
        if Path(self.figs_dir).exists():
            # always clear figs folder
            shutil.rmtree(self.figs_dir)
        
        Path(self.figs_dir).mkdir(parents=True, exist_ok=False)
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class Physics:
    fmax: float
    c: float
    c_phys: float
    rho: float
    dt: float

@dataclass
class Domain:
    spatial_dimension: int

    Xbounds: List[List[float]]
    tmax: float

    nX: List[List[int]]
    nt: int

    def __init__(self, Xbounds, tmax, dt, dx):
        assert(len(Xbounds[0]) == len(Xbounds[1]))
        
        if len(Xbounds) > 2:
            raise NotImplementedError()

        self.spatial_dimension = np.asarray(Xbounds).shape[1]
        self.Xbounds = Xbounds
        self.tmax = tmax

        self.dt = dt
        self.dx = dx        
        self.nt = int(tmax/dt) # number of temporal steps

    @property
    def num_sources(self) -> int:
        return len(self.x0_sources)

@dataclass(frozen=True)
class CEODSettings:    
    training_data_src_path: str
    testing_data_src_path: str
    loss_layer_indx: int

@dataclass
class TransferLearning:
    transfer_model_path: str
    resume_learning: bool
    freeze_layers: set

    def __init__(self,settings_dict: dict, transfer_model_path: str):
        self.transfer_model_path = settings_dict['transfer_learning']['transfer_model_path'] if 'transfer_model_path' in settings_dict['transfer_learning'] else transfer_model_path
        self.freeze_layers = settings_dict['transfer_learning']['freeze_layers'] if 'freeze_layers' in settings_dict['transfer_learning'] else dict()
        self.resume_learning = settings_dict['transfer_learning']['resume_learning']

@dataclass(frozen=True)
class TrainingSettings:
    iterations: int
    use_adaptive_weights: bool    
    learning_rate: float
    decay_steps: float
    decay_rate: float
    optimizer: str
    batch_size_branch: int
    batch_size_coord: int
    

@dataclass(frozen=True)
class NetworkArchitecture:
    architecture: NetworkArchitectureType
    activation: str
    num_hidden_layers: int
    num_hidden_neurons: int
    num_output_neurons: int

@dataclass
class EvaluationSettings:
    model_dir: str
    data_path: str
    receiver_pos: [object]
    tmax: float
            
    snap_to_grid: bool
    write_full_wave_field: bool
    write_ir_wav: bool
    write_ir_plots: bool
    write_ir_animations: bool

    def __init__(self, settings):
        self.receiver_pos = np.empty(len(settings['receiver_positions']), dtype=object)
        for i_src in range(len(self.receiver_pos)):
            if isinstance(settings['receiver_positions'][i_src], str):
                # the receiver positions are located in another entry in the JSON file with the key inside 'recvs'
                recvs_key = settings['receiver_positions'][i_src]                
                if recvs_key not in settings:
                    raise Exception(f"The JSON key {recvs_key} for source index {i_src} was not found. Please add this key to the JSON file with corresponding receiver positions as value.")
                self.receiver_pos[i_src] = np.array(settings[recvs_key])
            else:
                # the receiver positions can be read directly from the 'receiver_positions' array
                self.receiver_pos[i_src] = np.array(settings['receiver_positions'][i_src])

        self.data_path = settings['validation_data_dir']
        self.model_dir = settings['model_dir']        
        self.tmax = settings['tmax']

        self.snap_to_grid = settings['snap_to_grid']
        self.write_full_wave_field = settings['write_full_wave_field']
        self.write_ir_wav = settings['write_ir_wav']
        self.write_ir_plots = settings['write_ir_plots']
        self.write_ir_animations = settings['write_ir_animations']