# ==============================================================================
# Copyright 2023 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial

@partial(jit, static_argnums=(2,))
def kernel(X, X2, gamma=5.0):
    '''
    Input: X  Size1*n_feature (source inputs - output of first FNN layer, aftet the activation)
           X2 Size2*n_feature (target inputs)
    Output: Size1*Size2
    '''

    n1, n2 = X.shape[0], X.shape[0] # batch size

    X = jnp.transpose(X)
    X2 = jnp.transpose(X2)    

    n1sq = jnp.sum(X**2, axis=0)    
    n2sq = jnp.sum(X2**2, axis=0)

    D = jnp.ones((n1, n2)) * n2sq + jnp.transpose((jnp.ones((n2, n1)) * n1sq)) - 2 * jnp.matmul(jnp.transpose(X), X2)
    K = jnp.exp(-gamma * D)

    return K

@partial(jit, static_argnums=(4,))
def calcCEOD(X_p, Y_p, X_q, Y_q, lamda = 1.0):
    # X_p is the output of the first green layer for target model
    # X_q is the output of the first green layer for source model
    # Y_p: labelled data from target (ground truth)
    # Y_q: unlabelled data from source
    
    assert(Y_p.shape[0] == Y_q.shape[0]) # batch size

    if len(X_p.shape) == 3:
        # remove empty dimension
        assert X_p.shape[1] == 1
        assert X_q.shape[1] == 1
        X_p = X_p[:,0,:]
        X_q = X_q[:,0,:]

    bs = Y_p.shape[0]
    nps = bs
    nq = bs
    
    I1 = jnp.eye(bs)
    I2 = jnp.eye(bs)

    # Construct kernels 
    Kxpxp = kernel(X_p, X_p)
    Kxqxq = kernel(X_q, X_q)
    Kxqxp = kernel(X_q, X_p)
    Kypyq = kernel(Y_p, Y_q)
    Kyqyq = kernel(Y_q, Y_q)
    Kypyp = kernel(Y_p, Y_p)
    
    # # Compute CEOD
    a = jnp.matmul((jnp.linalg.inv(Kxpxp+nps*lamda*I1)),Kypyp)
    b = jnp.matmul(a,(jnp.linalg.inv(Kxpxp+nps*lamda*I1)))
    c = jnp.matmul(b,Kxpxp)
    out1 = jnp.trace(c)
    
    a = jnp.matmul((jnp.linalg.inv(Kxqxq+nq*lamda*I2)),Kyqyq)
    b = jnp.matmul(a,(jnp.linalg.inv(Kxqxq+nq*lamda*I2)))
    c = jnp.matmul(b,Kxqxq)
    out2 = jnp.trace(c)
    
    a = jnp.matmul((jnp.linalg.inv(Kxpxp+nps*lamda*I1)),Kypyq)
    b = jnp.matmul(a,(jnp.linalg.inv(Kxqxq+nq*lamda*I2)))
    c = jnp.matmul(b,Kxqxp)
    out3 = jnp.trace(c)

    out = (out1 + out2 - 2*out3) # Cannot be negative
    
    return out