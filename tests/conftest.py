# ==============================================================================
# Copyright 2025 Technical University of Denmark
# Author: Nikolas Borrel-Jensen
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
"""Shared pytest fixtures for all tests."""

import tempfile
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_2d_mesh():
    """Create a simple 2D mesh for testing."""
    nx, ny = 10, 10
    x = np.linspace(-2, 2, nx)
    y = np.linspace(-3, 3, ny)
    xx, yy = np.meshgrid(x, y)
    mesh = np.column_stack([xx.flatten(), yy.flatten()])
    return mesh


@pytest.fixture
def sample_1d_mesh():
    """Create a simple 1D mesh for testing."""
    nx = 20
    x = np.linspace(-2, 3, nx)
    mesh = x.reshape(-1, 1)
    return mesh


@pytest.fixture
def sample_3d_mesh():
    """Create a simple 3D mesh for testing."""
    nx, ny, nz = 5, 5, 5
    x = np.linspace(-2, 2, nx)
    y = np.linspace(-3, 3, ny)
    z = np.linspace(-1, 2, nz)
    xx, yy, zz = np.meshgrid(x, y, z)
    mesh = np.column_stack([xx.flatten(), yy.flatten(), zz.flatten()])
    return mesh


@pytest.fixture
def sample_timesteps():
    """Create sample timesteps for testing."""
    return np.linspace(0, 1.0, 20)


@pytest.fixture
def create_mock_h5_file(sample_2d_mesh, sample_timesteps):
    """Factory fixture to create mock HDF5 files for testing."""

    def _create_h5_file(
        filepath, mesh=None, num_sources=3, include_conn=False, spatial_dim=2
    ):
        if mesh is None:
            mesh = sample_2d_mesh

        n_mesh = mesh.shape[0]
        n_time = len(sample_timesteps)

        with h5py.File(filepath, "w") as f:
            # Create mesh
            f.create_dataset("/mesh", data=mesh)

            # Create connectivity if requested
            if include_conn and spatial_dim == 2:
                # Simple triangulation connectivity
                conn = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
                f.create_dataset("/conn", data=conn)

            # Create umesh (uniform mesh for initial conditions)
            if spatial_dim == 2:
                umesh = np.column_stack([mesh[:, 0], mesh[:, 1], np.zeros(n_mesh)])
            elif spatial_dim == 3:
                umesh = mesh
            else:  # 1D
                umesh = np.column_stack(
                    [mesh[:, 0], np.zeros(n_mesh), np.zeros(n_mesh)]
                )

            f.create_dataset("/umesh", data=umesh)
            f["/umesh"].attrs["umesh_shape"] = np.array(umesh.shape, dtype=int)

            # Create pressure fields
            pressures = np.random.randn(n_time, n_mesh).astype(np.float32) * 0.1
            dset = f.create_dataset("/pressures", data=pressures)
            dset.attrs["time_steps"] = sample_timesteps
            dset.attrs["dt"] = np.array([sample_timesteps[1] - sample_timesteps[0]])
            dset.attrs["dx"] = np.array([0.1])
            dset.attrs["c"] = np.array([343.0])
            dset.attrs["c_phys"] = np.array([343.0])
            dset.attrs["rho"] = np.array([1.225])
            dset.attrs["sigma0"] = np.array([0.05])
            dset.attrs["fmax"] = np.array([1000.0])
            dset.attrs["tmax"] = np.array([sample_timesteps[-1]])

            # Create initial pressure field
            upressures = np.random.randn(n_mesh).astype(np.float32) * 0.1
            f.create_dataset("/upressures", data=upressures)

            # Create source positions (one per file)
            source_pos = mesh[n_mesh // 2, :]  # Center of mesh
            f.create_dataset("source_position", data=source_pos)

    return _create_h5_file


@pytest.fixture
def mock_h5_dataset_2d(temp_dir, create_mock_h5_file, sample_2d_mesh):
    """Create a set of mock HDF5 files for 2D testing."""
    num_files = 5
    files = []

    for i in range(num_files):
        filepath = temp_dir / f"test_data_{i}.h5"
        create_mock_h5_file(filepath, mesh=sample_2d_mesh, spatial_dim=2)
        files.append(filepath)

    yield temp_dir, files

    # Cleanup
    for f in files:
        if f.exists():
            f.unlink()


@pytest.fixture
def sample_jax_arrays():
    """Create sample JAX arrays for testing."""
    return {
        "branch_input": jnp.ones((4, 100)),  # batch_size=4, input_dim=100
        "trunk_input": jnp.ones((4, 50, 5)),  # batch_size=4, n_coords=50, coord_dim=5
        "outputs": jnp.ones((4, 50)),  # batch_size=4, n_coords=50
    }


@pytest.fixture
def physics_params():
    """Create sample physics parameters."""
    return {
        "c": 343.0,
        "c_phys": 343.0,
        "rho": 1.225,
        "fmax": 1000.0,
        "dt": 0.0001,
        "sigma0": 0.05,
    }
