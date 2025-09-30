# ==============================================================================
# Copyright 2025 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================

from deeponet_acoustics.datahandlers.datagenerators import DataInterface
import jax.numpy as jnp
from jax.typing import ArrayLike
from flax import linen as nn
from deeponet_acoustics.models.datastructures import NetworkArchitectureType

def networkInfo(model: nn.Module, in_dim: ArrayLike) -> None:
    """Print network setup."""
    import jax
    from deeponet_acoustics.utils.utils import expandCnnData

    is_resnet = model.network_type == NetworkArchitectureType.RESNET
    dummy_data = expandCnnData(jnp.ones(in_dim)) if is_resnet else jnp.expand_dims(jnp.ones(in_dim), axis=0)
    print(model.tabulate(jax.random.PRNGKey(1234), dummy_data))

def datasetInfo(dataset: DataInterface, dataset_val: DataInterface, batch_size_coord: int, batch_size: int) -> None:
    """Print info about the dataset."""
    batch_size_train = min(batch_size, dataset.N)
    batch_size_val = min(batch_size, dataset_val.N)

    print(f"Mesh shape: {dataset.mesh.shape}")
    print(f"Time steps: {len(dataset.tsteps)}")
    print(f"IC shape: {dataset.u_shape}")

    print(f"Train data size: {dataset.P}")
    if batch_size > dataset.N:
        print("NOTE: batch_size_branch > dataset.N")
    print(f"Train batch size (total): {batch_size_coord*batch_size_train}")
    print(f"Train num datasets: {dataset.N}")

    print(f"Val data size: {dataset_val.P}")
    print(f"Val batch size (total): {batch_size_coord*batch_size_val}")
    print(f"Val num datasets: {dataset_val.N}")