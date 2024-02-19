# ==============================================================================
# Copyright 2023 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
from dataclasses import dataclass
from models.datastructures import InputOutputDirs, NetworkArchitecture, NetworkArchitectureType, TrainingSettings, TransferLearning, ResNetArchitecture, MLPArchitecture

@dataclass
class SimulationSettings:
    dirs: InputOutputDirs    
    normalize_data: bool    
    
    tmax: float
    f0_feat: list[float]

    training_settings: TrainingSettings
    branch_net: NetworkArchitecture
    trunk_net: NetworkArchitecture

    transfer_learning: TransferLearning | None = None

    def __init__(self, settings, input_dir=None, output_dir=None):
        self.dirs = InputOutputDirs(settings, input_dir=input_dir, output_dir=output_dir)

        self.tmax = settings['tmax']
        self.f0_feat = settings['f0_feat'] if 'f0_feat' in settings else []        
        self.normalize_data = settings['normalize_data'] if 'normalize_data' in settings else True

        # networks
        num_output_neurons = settings['num_output_neurons'] # same for both nets        
        self.branch_net = setupNetworkArchitecture(settings['branch_net'], num_output_neurons)        
        self.trunk_net  = setupNetworkArchitecture(settings['trunk_net'], num_output_neurons)        

        if 'transfer_learning' in settings:
            self.transfer_learning = TransferLearning(settings, self.dirs.models_dir)
            
            num_frozen_layers_bn = len(self.transfer_learning.freeze_layers['bn']) if 'bn' in self.transfer_learning.freeze_layers else 0
            num_frozen_layers_tn = len(self.transfer_learning.freeze_layers['tn']) if 'tn' in self.transfer_learning.freeze_layers else 0
                                       
            if num_frozen_layers_bn > self.branch_net.num_hidden_layers + 1:
                raise Exception(f"ERROR: Number of branch layers to freeze exceeds the number of layers (hidden + out): {num_frozen_layers_bn}/{self.branch_net.num_hidden_layers + 1}")
            if num_frozen_layers_bn == self.branch_net.num_hidden_layers + 1:
                print(f"WARNING: all branch layers are frozen (hidden + out): {num_frozen_layers_bn}/{self.branch_net.num_hidden_layers + 1}")
            
            if num_frozen_layers_tn > self.trunk_net.num_hidden_layers + 1:
                raise Exception(f"ERROR: Number of trunk layers to freeze exceeds the number of layers (hidden + out): {num_frozen_layers_tn}/{self.trunk_net.num_hidden_layers + 1}")
            if num_frozen_layers_tn == self.trunk_net.num_hidden_layers + 1:
                print(f"WARNING: all trunk layers are frozen (hidden + out): {num_frozen_layers_tn}/{self.trunk_net.num_hidden_layers + 1}")

        batch_size_branch = settings['batch_size_branch']
        batch_size_coord = settings['batch_size_coord']
        iter = settings['iterations']
        use_adaptive_weights = settings['use_adaptive_weights'] if 'use_adaptive_weights' in settings else False
        learning_rate = settings['learning_rate']
        decay_steps = settings['decay_steps']
        decay_rate = settings['decay_rate']
        optimizer = settings['optimizer']
        self.training_settings = TrainingSettings(
            iter,use_adaptive_weights,learning_rate,decay_steps,decay_rate,optimizer,
            batch_size_branch,batch_size_coord)

def setupNetworkArchitecture(net : dict, num_output_neurons: int):
    architecture = net['architecture']
    activation = net['activation']    
    num_hidden_layers = net['num_hidden_layers']
    num_hidden_neurons = net['num_hidden_neurons']
    
    if architecture == "mlp":
        return MLPArchitecture(NetworkArchitectureType.MLP,activation,num_hidden_layers,num_hidden_neurons,num_output_neurons)
    elif architecture == "mod-mlp":        
        return MLPArchitecture(NetworkArchitectureType.MOD_MLP,activation,num_hidden_layers,num_hidden_neurons,num_output_neurons)
    elif architecture == "resnet":
        num_group_blocks = net['num_group_blocks']
        cnn_hidden_layers = net['cnn_hidden_layers']
        return ResNetArchitecture(NetworkArchitectureType.RESNET,activation,num_hidden_layers,num_hidden_neurons,num_output_neurons, 
                                   num_group_blocks, cnn_hidden_layers)
    else:
        raise Exception("Architecture type not supported: %s", architecture)