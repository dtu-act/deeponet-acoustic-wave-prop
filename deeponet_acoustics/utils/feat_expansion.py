# ==============================================================================
# Copyright 2023 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import numpy as np
from numpy.random import default_rng

def fourierFeatureExpansion_exact_sol(fs: list[float], c: float, Lx: float, Ly: float):
    round_list = lambda l: [round(n) for n in l]
    modes = list(zip( round_list(Lx*2/(c/fs)), round_list(Ly*2/(c/fs)) ))
    
    omega = lambda n: np.sqrt( c**2*np.pi**2 *( (n[0]/Lx)**2 + (n[1]/Ly)**2 ) )
    es_cos = lambda y,n: np.cos(omega(n)*y[:,2]) * np.cos(np.pi*n[0]/Lx*y[:,0]) * np.cos(np.pi*n[1]/Ly*y[:,1])
    es_sin = lambda y,n: np.sin(omega(n)*y[:,2]) * np.cos(np.pi*n[0]/Lx*y[:,0]) * np.cos(np.pi*n[1]/Ly*y[:,1])

    feat = lambda y: np.hstack([
        y,
        np.array(list(map(lambda n: es_cos(y,n), modes))).T,
        np.array(list(map(lambda n: es_sin(y,n), modes))).T
        ])

    return feat

def fourierFeatureExpansion_gaussian(shape: tuple, mean: float, std_dev: float):
    rng = default_rng()
    B = rng.normal(loc=mean, scale=std_dev, size=shape)
    
    return lambda y: np.hstack([
        y,
        np.cos(2*np.pi*B @ y.T).T,
        np.sin(2*np.pi*B @ y.T).T,
        ])

def fourierFeatureExpansion_f0(fs : list):
    fs = np.asarray(fs)
    
    if len(fs) == 0:
        return lambda y: y
    else:
        return lambda y: np.hstack([
            y,
            *np.array(list(map(lambda f: np.cos(2*np.pi*f*y), fs))),
            *np.array(list(map(lambda f: np.sin(2*np.pi*f*y), fs)))
            ])