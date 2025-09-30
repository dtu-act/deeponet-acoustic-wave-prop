# ==============================================================================
# Copyright 2024 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import flax
import jax.numpy as jnp
from numpy import dtype
from flax import linen as nn
from typing import Any, Callable
from jax import random
from dataclasses import dataclass
from jax.typing import ArrayLike

from deeponet_acoustics.models.datastructures import NetworkArchitecture, NetworkArchitectureType

DTypeLikeInexact = Any  # DTypeLikeFloat | DTypeLikeComplex

def freezeSomdattaLayers():
    freeze_set = set()
    freeze_set.add("Conv_0")
    freeze_set.add("Conv_1")
    freeze_set.add("Conv_2")
    freeze_set.add("Conv_3")

    freeze_set.add("linear_tn_0")
    freeze_set.add("linear_tn_1")
    freeze_set.add("linear_tn_2")
    freeze_set.add("linear_tn_3")

    return freeze_set

def freezeLayersToKeys(freeze_layers: dict, freeze_b0=False):
    freeze_set = set()
    if freeze_b0:
        freeze_set.add(f"b0") 
    if 'bn' in freeze_layers:
        for i in freeze_layers['bn']:
            freeze_set.add(f"linear_bn_{i}")
    if 'tn' in freeze_layers:
        for i in freeze_layers['tn']:
            freeze_set.add(f"linear_tn_{i}")
    if 'b0' in freeze_layers:
        freeze_set.add(f"b0")
    if 'bn_transformer' in freeze_layers:
        freeze_set.add(f"transformerU_bn")
        freeze_set.add(f"transformerV_bn")
    if 'tn_transformer' in freeze_layers:
        freeze_set.add(f"transformerU_tn")
        freeze_set.add(f"transformerV_tn")

    return freeze_set

def freezeCnnLayersToKeys(params: dict):
    freeze_set = set(params['params'].keys())

    keys_remove = set()
    for elem in freeze_set:
        if 'Dense' in elem:
            keys_remove.add(elem)
    
    for key in keys_remove:
        freeze_set.remove(key)

    return freeze_set

def flattened_traversal(fn, network_keys):
  """Returns function that is called with `(path, param)` instead of pytree."""
  def mask(tree):
    masked_dict = {}
    for key in network_keys:
        if isinstance(tree[key], dict) or isinstance(tree[key], flax.core.FrozenDict):
            flat = flax.traverse_util.flatten_dict(tree[key])
            out = flax.traverse_util.unflatten_dict(
                {k: fn(k, v) for k, v in flat.items()})
            masked_dict[key] = out
        else:
            masked_dict[key] = fn(key, tree[key])
    print(masked_dict)
    return flax.core.frozen_dict.freeze(masked_dict)
  return mask

def sinusoidal_init(is_first=False):
    def init(key: ArrayLike,
             shape: ArrayLike,
             dtype: DTypeLikeInexact = dtype) -> ArrayLike:

        d_in = shape[0]
        d_out = shape[1]

        if is_first:
            minval=-1 / d_in
            maxval= 1 / d_in
        else:
            minval=-jnp.sqrt(6 / d_in) / 30
            maxval= jnp.sqrt(6 / d_in) / 30
        W = random.uniform(key, (d_in, d_out), minval=minval, maxval=maxval, dtype=dtype)
        
        return W

    return init

def setupNetwork(net: NetworkArchitecture, in_bn: ArrayLike, tag: str) -> tuple[nn.Module, tuple[float]]:
    if net.activation == "sin":
        activation = jnp.sin
        kernel_init = sinusoidal_init
        angular_freq = 30.0
    elif net.activation == "tanh":
        activation = nn.tanh
        kernel_init = lambda _=None: nn.initializers.xavier_uniform()
        angular_freq = 1.0
    elif net.activation == "relu":
        activation = nn.relu
        kernel_init = lambda _=None: nn.initializers.he_uniform()
        angular_freq = 1.0
    elif net.activation == "leaky_relu":
        activation = nn.leaky_relu
        kernel_init = lambda _=None: nn.initializers.he_uniform()
        angular_freq = 1.0
    else:
        raise Exception(f"Activation function not supported: {net.activation}")
        
    if net.architecture == NetworkArchitectureType.MLP:
        layers  = net.num_hidden_layers*[net.num_hidden_neurons]  + [net.num_output_neurons]
        fnn = MLP(layers=layers, tag=tag, 
                  activation=activation, kernel_init=kernel_init, angular_freq=angular_freq)
        return fnn
    elif net.architecture == NetworkArchitectureType.MOD_MLP:
        layers  = net.num_hidden_layers*[net.num_hidden_neurons]  + [net.num_output_neurons]
        fnn = ModMLP(layers=layers, tag=tag, 
                     activation=activation, kernel_init=kernel_init, angular_freq=angular_freq)
        return fnn
    elif net.architecture == NetworkArchitectureType.RESNET:
        num_group_blocks = net.num_group_blocks # : tuple = (3, 3, 3, 3) # todo: read from settings
        c_hidden = net.cnn_hidden_layers # : tuple = (16, 32, 64, 128) # todo: read from settings
        layers  = net.num_hidden_layers*[net.num_hidden_neurons]  + [net.num_output_neurons]
        kernel_size = (3,3) if len(in_bn.shape) == 2 else (3,3,3)
        resnet = ResNet(layers_fnn=layers, num_blocks=num_group_blocks, c_hidden=c_hidden, act_fn=activation, kernel_size=kernel_size)
        return resnet
    else:
        raise Exception('Network architecture is not supported')        

@dataclass
class MLP(nn.Module):
    layers: list[int]
    angular_freq: float = 30 # angular frequency for inputs to first layer
    activation: Callable = jnp.sin
    kernel_init: Callable = sinusoidal_init
    tag: str = "<undef>"
    
    @nn.compact
    def __call__(self, inputs, output_layer_indx=-1):
        x = inputs        

        # Note: arr[0:n] is outputting n elements where the last element has value n-1. 
        #       Hence we are returning the output after the _hidden_ layer (not counting the input layer)
        #       at position output_layer_indx-1 after applying the activation function (unless last layer)
        for i, feat in enumerate(self.layers[0:output_layer_indx]):
            if i == 0:
                x = nn.Dense(features=self.layers[0],
                        kernel_init=self.kernel_init(True),
                        name=f'linear_{self.tag}_0')(self.angular_freq*x)
                x = self.activation(x)
            else:
                x = nn.Dense(features=feat, kernel_init=self.kernel_init(), name=f'linear_{self.tag}_{i}')(x)
                x = self.activation(x)

        if output_layer_indx == -1:
            # output layer (no activation)
            x = nn.Dense(features=self.layers[-1], kernel_init=self.kernel_init(), name=f'linear_{self.tag}_{len(self.layers)-1}')(x)

        return x
    
class ModMLP(nn.Module):
    layers: list[int]
    angular_freq: float = 30 # angular frequency for inputs to first layer
    activation: Callable = jnp.sin
    kernel_init: Callable = sinusoidal_init
    tag: str = "<undef>"
    network_type: NetworkArchitectureType = NetworkArchitectureType.MOD_MLP
    
    @nn.compact
    def __call__(self, inputs, output_layer_indx=-1):
        outputs = inputs # only used if output_layer_indx = 0

        U = nn.Dense(features=self.layers[0], kernel_init=self.kernel_init(True), name=f'transformerU_{self.tag}')(inputs)
        V = nn.Dense(features=self.layers[0], kernel_init=self.kernel_init(True), name=f'transformerV_{self.tag}')(inputs)
        # NOTE: if the models from https://doi.org/10.11583/DTU.24812004 are used, please comment out the following two lines

        U = self.activation(U)
        V = self.activation(V)

        for i, feat in enumerate(self.layers[0:output_layer_indx]):
            if i == 0:
                layer0 = nn.Dense(features=self.layers[0], kernel_init=self.kernel_init(True), name=f'linear_{self.tag}_0')        
                _ = layer0(inputs) # propagate input through to set parameter shape        
                W = layer0.variables['params']['kernel']
                b = layer0.variables['params']['bias']

                outputs = self.activation(self.angular_freq*jnp.dot(inputs, W) + b)                
            else:
                outputs = nn.Dense(features=feat, kernel_init=self.kernel_init(), name=f'linear_{self.tag}_{i}')(outputs)
                outputs = self.activation(outputs)
            
            outputs = jnp.multiply(outputs, U) + jnp.multiply(1 - outputs, V)
        
        if output_layer_indx ==-1:
            # output layer (no activation)
            outputs = nn.Dense(features=self.layers[-1], kernel_init=self.kernel_init(), name=f'linear_{self.tag}_{len(self.layers)-1}')(outputs)

        return outputs

# Conv initialized with kaiming int, but uses fan-out instead of fan-in mode
# Fan-out focuses on the gradient distribution, and is commonly used in ResNets
resnet_kernel_init = nn.initializers.variance_scaling(2.0, mode='fan_out', distribution='normal')

class ResNetBlock(nn.Module):
    act_fn : Callable  # Activation function
    c_out : int   # Output feature size
    subsample : bool = False  # If True, we apply a stride inside F
    kernel_size : tuple = (3, 3)    

    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        strides1 = (1, 1) if len(self.kernel_size) == 2 else (1,1,1)        
        strides2 = (2, 2) if len(self.kernel_size) == 2 else (2,2,2)
        kernel_size = strides1
        z = nn.Conv(self.c_out, kernel_size=self.kernel_size,
                    strides=strides1 if not self.subsample else strides2,
                    kernel_init=resnet_kernel_init,
                    use_bias=False)(x)
        z = nn.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=self.kernel_size,
                    kernel_init=resnet_kernel_init,
                    use_bias=False)(z)
        z = nn.BatchNorm()(z, use_running_average=not train)

        if self.subsample:
            x = nn.Conv(self.c_out, kernel_size=kernel_size, strides=strides2, kernel_init=resnet_kernel_init)(x)

        x_out = self.act_fn(z + x)
        return x_out

class PreActResNetBlock(ResNetBlock):
    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        z = nn.BatchNorm()(x, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=self.kernel_size,
                    strides=(1, 1) if not self.subsample else (2, 2),
                    kernel_init=resnet_kernel_init,
                    use_bias=False)(z)
        z = nn.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=self.kernel_size,
                    kernel_init=resnet_kernel_init,
                    use_bias=False)(z)

        if self.subsample:
            x = nn.BatchNorm()(x, use_running_average=not train)
            x = self.act_fn(x)
            x = nn.Conv(self.c_out,
                        kernel_size=(1, 1),
                        strides=(2, 2),
                        kernel_init=resnet_kernel_init,
                        use_bias=False)(x)

        x_out = z + x
        return x_out
    
class ResNet(nn.Module):
    layers_fnn: list[int]
    num_blocks : tuple
    c_hidden : tuple
    kernel_size: tuple = (3, 3)
    kernel_sine_init: Callable = sinusoidal_init
    act_fn : Callable = nn.relu    
    block_class : nn.Module = ResNetBlock
    network_type: NetworkArchitectureType = NetworkArchitectureType.RESNET

    @nn.compact
    def __call__(self, x, output_layer_indx=-1, train=True):        
        # A first convolution on the original image to scale up the channel size
        x = nn.Conv(self.c_hidden[0], kernel_size=self.kernel_size, 
                    kernel_init=resnet_kernel_init, use_bias=False)(x)
        if self.block_class == ResNetBlock:  # If pre-activation block, we do not apply non-linearities yet
            x = nn.BatchNorm()(x, use_running_average=not train)
            x = self.act_fn(x)

        # Creating the ResNet blocks
        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                # Subsample the first block of each group, except the very first one.
                subsample = (bc == 0 and block_idx > 0)
                # ResNet block
                x = self.block_class(c_out=self.c_hidden[block_idx],
                                     act_fn=self.act_fn,
                                     kernel_size=self.kernel_size,
                                     subsample=subsample)(x, train=train)
        
        x = x.reshape((x.shape[0], -1))

        for _, feat in enumerate(self.layers_fnn[0:output_layer_indx]):
            x = nn.Dense(features=feat, kernel_init=self.kernel_sine_init(True))(x)
            x = jnp.sin(x)
            # x = nn.Dense(features=feat, kernel_init=nn.initializers.xavier_uniform())(x)            
            # x = nn.tanh(x)

        if output_layer_indx ==-1:
            x = nn.Dense(features=self.layers_fnn[-1], kernel_init=self.kernel_sine_init())(x)
            # x = nn.Dense(features=self.layers_fnn[-1], kernel_init=nn.initializers.xavier_uniform())(x)

        return x

class Cnn(nn.Module):
    layers_fnn: list[int]
    num_conv : int
    c_hidden : tuple
    kernel_size: tuple = (3, 3)
    act_fn : Callable = nn.relu

    @nn.compact
    def __call__(self, x, output_layer_indx=-1, train=True):
        for i in range(self.num_conv):
            x = nn.Conv(self.c_hidden[i], kernel_size=self.kernel_size, 
                kernel_init=resnet_kernel_init, padding='SAME', strides=(1,1))(x)
            x = self.act_fn(x)
            x = nn.avg_pool(x, window_shape=(2,2), strides=(2,2), padding='SAME')            

        x = x.reshape((x.shape[0], -1))
        # for i, feat in enumerate(self.layers_fnn[0:-1]):
        x = nn.Dense(features=512, kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.tanh(x)
        if output_layer_indx > -1: # NBJ TODO HACK
            return x
        
        x = nn.Dense(features=256, kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.tanh(x)
            
        x = nn.Dense(features=self.layers_fnn[-1], kernel_init=nn.initializers.xavier_uniform())(x)

        return x