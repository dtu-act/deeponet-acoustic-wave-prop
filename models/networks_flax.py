# ==============================================================================
# Copyright 2023 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
from functools import partial
import flax
import jax.numpy as jnp
from jax.nn import relu
from numpy import dtype, float32
from flax import linen as nn           # The Linen API
from typing import Any, Sequence, Tuple, Union
from jax import random
from jax import core
from dataclasses import dataclass

from models.datastructures import NetworkArchitecture

#https://github.com/google/flax/blob/main/examples/wmt/models.py
#https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.html
#https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/guide4/Research_Projects_with_JAX.html

KeyArray = random.KeyArray
DTypeLikeInexact = Any  # DTypeLikeFloat | DTypeLikeComplex
Array = Any

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
    def init(key: KeyArray,
             shape: core.Shape,
             dtype: DTypeLikeInexact = dtype) -> Array:

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

# https://datascience.stackexchange.com/questions/66944/is-it-wrong-to-use-glorot-initialization-with-relu-activation
# https://stackoverflow.com/questions/48641192/xavier-and-he-normal-initialization-difference/48641573#48641573
def setupFNN(net: NetworkArchitecture, tag: str, mod_fnn):
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

    trunk_layers  = net.num_hidden_layers*[net.num_hidden_neurons]  + [net.num_output_neurons]

    if mod_fnn:
        return modified_MLP(layers=trunk_layers, tag=tag, 
                            activation=activation, kernel_init=kernel_init, angular_freq=angular_freq)
    else:
        return MLP(layers=trunk_layers, tag=tag, 
                   activation=activation, kernel_init=kernel_init, angular_freq=angular_freq)


@dataclass
class MLP(nn.Module):
    layers: list[int]
    angular_freq: float = 30 # angular frequency for inputs to first layer
    activation: callable = jnp.sin
    kernel_init: callable = sinusoidal_init
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

class modified_MLP(nn.Module):
    layers: list[int]
    angular_freq: float = 30 # angular frequency for inputs to first layer
    activation: callable = jnp.sin
    kernel_init: callable = sinusoidal_init
    tag: str = "<undef>"
    
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
    act_fn : callable  # Activation function
    c_out : int   # Output feature size
    subsample : bool = False  # If True, we apply a stride inside F
    kernel_size : tuple = (3, 3)

    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        z = nn.Conv(self.c_out, kernel_size=self.kernel_size,
                    strides=(1, 1) if not self.subsample else (2, 2),
                    kernel_init=resnet_kernel_init,
                    use_bias=False)(x)
        z = nn.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=self.kernel_size,
                    kernel_init=resnet_kernel_init,
                    use_bias=False)(z)
        z = nn.BatchNorm()(z, use_running_average=not train)

        if self.subsample:
            x = nn.Conv(self.c_out, kernel_size=(1, 1), strides=(2, 2), kernel_init=resnet_kernel_init)(x)

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
    kernel_sine_init: callable = sinusoidal_init
    act_fn : callable = nn.relu    
    block_class : nn.Module = ResNetBlock    

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

        for i, feat in enumerate(self.layers_fnn[0:output_layer_indx]):
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
    act_fn : callable = nn.relu

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