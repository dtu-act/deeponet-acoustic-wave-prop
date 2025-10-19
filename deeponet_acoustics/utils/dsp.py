# ==============================================================================
# Copyright 2023 Technical University of Denmark
# Author: Nikolas Borrel-Jensen
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import jax as jnp
import numpy as np


def calcFFT(y, fs, NFFT=None):
    if NFFT is None:
        N = len(y)
    else:
        N = np.maximum(NFFT, len(y))

    Y = np.fft.fft(y, n=NFFT, norm="forward")  # forward norm 1/N
    Y = 2 * np.abs(
        Y[0 : N // 2]
    )  # mult by 2: compensating for the loss of energy taking only half

    freqs = fs * np.array([i for i in range(N // 2)]) / N

    return Y, freqs


def calcJaxFFT(y, fs, NFFT=None):
    if NFFT is None:
        N = y.shape[0]
    else:
        N = jnp.maximum(NFFT, len(y))

    Y = jnp.fft.fft(y, n=NFFT)
    freqs = fs * jnp.array([i for i in range(N // 2)]) / N
    Y = 2.0 / N * jnp.abs(Y[0 : N // 2])
    return Y, freqs
