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
from typing import Callable, Dict, List, TypeAlias
import jax
import numpy as np
from flax import linen as nn           # The Linen API
from utils.utils import expandCnnData

class NetworkArchitectureType(Enum):    
    MLP = 1
    MOD_MLP = 2
    RESNET = 3

class SimulationDataType(Enum):
    H5COMPACT = 1
    XDMF = 2
    SOURCE_ONLY = 3

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
    network: nn.Module

    def __init__(self, network: nn.Module, in_dim):
        self.network = network
        self.in_dim = in_dim
        
        is_resnet = network.network_type == NetworkArchitectureType.RESNET
        print(network.tabulate(jax.random.PRNGKey(1234), expandCnnData(np.ones(in_dim)) if is_resnet else np.expand_dims(np.ones(in_dim), axis=0)))

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
class MLPArchitecture:
    architecture: NetworkArchitectureType
    activation: str
    num_hidden_layers: int
    num_hidden_neurons: int
    num_output_neurons: int

# "num_group_blocks": [3, 3, 3, 3],
# "cnn_hidden_layers": [16, 32, 64, 128],
# "num_hidden_layers": 0,
# "num_hidden_neurons": 2048

@dataclass(frozen=True)
class ResNetArchitecture: 
    architecture: NetworkArchitectureType # = NetworkArchitectureType.RESNET
    activation: str
    num_hidden_layers: int
    num_hidden_neurons: int
    num_output_neurons: int
    num_group_blocks: tuple
    cnn_hidden_layers: tuple

NetworkArchitecture: TypeAlias = MLPArchitecture | ResNetArchitecture

@dataclass
class EvaluationSettings:
    model_dir: str
    data_path: str
    receiver_pos: List[object]
    tmax: float
    num_srcs: int
            
    snap_to_grid: bool
    source_position_override: List[object]
    write_full_wave_field: bool
    write_ir_wav: bool
    write_ir_plots: bool
    write_ir_animations: bool

    def __init__(self, settings: Dict, num_srcs: int = -1):
        key_recv_pos = 'receiver_positions_all_sources'
        key_recv_groups = 'receiver_position_groups'
        key_src_pos = 'source_positions'

        if key_src_pos in settings:
            if not isinstance(settings['source_positions'], list) or \
                not len(np.array(settings['source_positions']).shape) == 2:
                raise Exception("Source positions are explicitly set: Expected non-empty two-dimensional list.")
            
            self.num_srcs = len(settings['source_positions'])
            self.source_position_override = np.empty(self.num_srcs, dtype=object)

            for i_src in range(len(self.source_position_override)):
                self.source_position_override[i_src] = settings['source_positions'][i_src]
        else:
            if num_srcs <= 0:
                raise Exception("Number of source positions cannot be determined from the settings (source not set explicitly, instead loaded from disk), please provide it as input argument to the function")
            self.num_srcs = num_srcs
            self.source_position_override = np.array([])

        if (key_recv_groups in settings):
            if not isinstance(settings[key_recv_groups], list) or \
                len(settings[key_recv_groups]) == 0 or \
                not isinstance(settings[key_recv_groups][0], str):
                raise Exception("Expected non-empty list of string for key receiver_position_groups")
            elif len(settings[key_recv_groups]) != self.num_srcs:
                raise Exception(f"Number of receiver groups (receiver_position_groups) {len(settings[key_recv_groups])} differs from number of source {self.num_srcs}")
            
            self.receiver_pos = self.parseReceiverGroups(settings[key_recv_groups], settings)                
        elif key_recv_pos in settings:            
            if not isinstance(settings[key_recv_pos], list) or \
                len(settings[key_recv_pos]) == 0 or \
                not isinstance(settings[key_recv_pos][0], list) or \
                not len(np.array(settings[key_recv_pos]).shape) == 2:
                raise Exception("Expected non-empty two-dimensional list of receiver coordinates ('receiver_positions_all_sources'). The same receivers are expected to be used for each source position and should NOT be repeated for each source (contrary to 'receiver_position_groups')")
            self.receiver_pos = np.empty(self.num_srcs, dtype=object)
            for i_src in range(self.num_srcs):
                # repeat the same receivers for each source
                self.receiver_pos[i_src] = np.array(settings[key_recv_pos])
        else:
            raise Exception("Missing receiver information: expected either of the following keys: 'receiver_position_groups', 'receiver_positions_all_sources'")
        
        self.data_path = settings['validation_data_dir']
        self.model_dir = settings['model_dir']        
        self.tmax = settings['tmax']

        self.snap_to_grid = settings['snap_to_grid']
        self.write_full_wave_field = settings['write_full_wave_field']
        self.write_ir_wav = settings['write_ir_wav']
        self.write_ir_plots = settings['write_ir_plots']
        self.write_ir_animations = settings['write_ir_animations']

    def parseReceiverGroups(self, receiver_keys: List, receivers: Dict) -> List[object]:
        receiver_pos = np.empty(len(receiver_keys), dtype=object)
        for i_src in range(len(receiver_keys)):
            # the receiver positions are located in another entry in the JSON file with the key inside 'recvs'
            recvs_key = receiver_keys[i_src]                
            if recvs_key not in receivers:
                raise Exception(f"The JSON key {recvs_key} for source index {i_src} was not found. Please add this key to the JSON file with corresponding receiver positions as value.")
            receiver_pos[i_src] = np.array(receivers[recvs_key])
        
        return receiver_pos