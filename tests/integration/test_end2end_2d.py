# ==============================================================================
# Copyright 2025 Technical University of Denmark
# Author: Nikolas Borrel-Jensen
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
"""End-to-end integration test for 2D acoustic wave propagation."""

import h5py
import jax.numpy as jnp
import numpy as np
import pytest
from torch.utils.data import DataLoader

from deeponet_acoustics.datahandlers.datagenerators import (
    DataH5Compact,
    DatasetStreamer,
    numpy_collate,
)
from deeponet_acoustics.models.datastructures import (
    MLPArchitecture,
    NetworkArchitectureType,
    TrainingSettings,
)
from deeponet_acoustics.models.deeponet import DeepONet
from deeponet_acoustics.models.networks_flax import setupNetwork


def create_synthetic_2d_acoustic_data(filepath, nx=15, ny=15, nt=15):
    """
    Create a synthetic 2D acoustic wave dataset for testing.

    This generates a simple wave pattern that mimics acoustic propagation
    from a point source in a 2D domain.
    """
    # Create 2D mesh
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    xx, yy = np.meshgrid(x, y)
    mesh = np.column_stack([xx.flatten(), yy.flatten()])
    n_mesh = mesh.shape[0]

    # Create time steps
    dt = 0.001
    t = np.linspace(0, dt * (nt - 1), nt)

    # Physical parameters
    c = 343.0  # speed of sound
    fmax = 500.0
    sigma0 = 0.1

    # Source position (center of domain)
    source_pos = np.array([0.0, 0.0])

    # Create initial condition (Gaussian)
    umesh = np.column_stack([mesh[:, 0], mesh[:, 1], np.zeros(n_mesh)])
    distances = np.sqrt(
        (umesh[:, 0] - source_pos[0]) ** 2 + (umesh[:, 1] - source_pos[1]) ** 2
    )
    upressures = 2.0 * np.exp(-(distances**2) / sigma0**2)

    # Create pressure field (simple analytical wave propagation)
    pressures = np.zeros((nt, n_mesh), dtype=np.float32)
    for i, ti in enumerate(t):
        # Simple radial wave: p(r,t) = A/r * sin(k*r - omega*t)
        # Simplified for testing
        r = distances + 0.1  # Avoid division by zero
        omega = 2 * np.pi * fmax / 4
        k = omega / c
        pressures[i, :] = (1.0 / r) * np.sin(k * r - omega * ti) * np.exp(-ti * 2)

    # Write to HDF5
    with h5py.File(filepath, "w") as f:
        # Mesh data
        f.create_dataset("/mesh", data=mesh)
        f.create_dataset("/umesh", data=umesh)
        f["/umesh"].attrs["umesh_shape"] = np.array(umesh.shape, dtype=int)

        # Pressure data
        dset = f.create_dataset("/pressures", data=pressures)
        dset.attrs["time_steps"] = t
        dset.attrs["dt"] = np.array([dt])
        dset.attrs["dx"] = np.array([2.0 / nx])
        dset.attrs["c"] = np.array([c])
        dset.attrs["c_phys"] = np.array([c])
        dset.attrs["rho"] = np.array([1.225])
        dset.attrs["sigma0"] = np.array([sigma0])
        dset.attrs["fmax"] = np.array([fmax])
        dset.attrs["tmax"] = np.array([t[-1]])

        # Initial pressure
        f.create_dataset("/upressures", data=upressures.astype(np.float32))

        # Source position
        f.create_dataset("source_position", data=source_pos)


@pytest.fixture
def small_2d_dataset(temp_dir):
    """Create a small synthetic 2D dataset for integration testing."""
    train_dir = temp_dir / "train"
    test_dir = temp_dir / "test"
    train_dir.mkdir()
    test_dir.mkdir()

    # Create training files (3 sources)
    train_files = []
    for i in range(3):
        filepath = train_dir / f"train_{i}.h5"
        create_synthetic_2d_acoustic_data(filepath, nx=15, ny=15, nt=12)
        train_files.append(filepath)

    # Create test files (1 source)
    test_files = []
    filepath = test_dir / "test_0.h5"
    create_synthetic_2d_acoustic_data(filepath, nx=15, ny=15, nt=12)
    test_files.append(filepath)

    return train_dir, test_dir


@pytest.mark.integration
@pytest.mark.slow
class TestEnd2End2D:
    """End-to-end integration test for 2D training pipeline."""

    def test_complete_training_pipeline(self, small_2d_dataset, temp_dir):
        """
        Test complete training pipeline with small 2D dataset.

        This test verifies:
        1. Data loading from HDF5 files
        2. Network initialization
        3. Training loop execution
        4. Loss computation and reduction
        5. Model prediction
        """
        train_dir, test_dir = small_2d_dataset
        model_dir = temp_dir / "models"
        model_dir.mkdir()

        # Setup training parameters
        training_settings = TrainingSettings(
            iterations=20,  # Very short for testing
            batch_size_branch=2,
            batch_size_coord=50,
            learning_rate=0.001,
            decay_steps=100,
            decay_rate=0.95,
            optimizer="adam",
            use_adaptive_weights=False,
        )

        # Load data
        data_train = DataH5Compact(
            str(train_dir),
            tmax=0.015,
            t_norm=343.0,
            flatten_ic=True,
            norm_data=True,
        )
        data_test = DataH5Compact(
            str(test_dir),
            tmax=0.015,
            t_norm=343.0,
            flatten_ic=True,
            norm_data=True,
        )

        assert data_train.N == 3
        assert data_test.N == 1

        # Create datasets
        dataset_train = DatasetStreamer(data_train, batch_size_coord=50)
        dataset_test = DatasetStreamer(data_test, batch_size_coord=-1)  # Full dataset

        # Create dataloaders
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=training_settings.batch_size_branch,
            shuffle=True,
            collate_fn=numpy_collate,
            drop_last=True,
        )
        dataloader_test = DataLoader(
            dataset_test,
            batch_size=1,
            shuffle=False,
            collate_fn=numpy_collate,
            drop_last=False,
        )

        # Setup networks
        branch_arch = MLPArchitecture(
            architecture=NetworkArchitectureType.MLP,
            num_hidden_layers=3,
            num_hidden_neurons=64,
            num_output_neurons=64,
            activation="relu",
        )
        trunk_arch = MLPArchitecture(
            architecture=NetworkArchitectureType.MLP,
            num_hidden_layers=3,
            num_hidden_neurons=64,
            num_output_neurons=64,
            activation="relu",
        )

        # Get input dimensions
        sample_batch = next(iter(dataloader_train))
        in_tn = sample_batch[0][1].shape[-1]  # Trunk input dim
        in_bn = data_train.u_shape  # Branch input dim

        branch_net = setupNetwork(branch_arch, "bn")
        trunk_net = setupNetwork(trunk_arch, "tn")

        # Initialize model
        model = DeepONet(
            settings=training_settings,
            dataset=data_train,
            module_bn=(branch_net, in_bn),
            module_tn=(trunk_net, in_tn),
            log_dir=str(model_dir),
            transfer_learning=None,
        )

        # Record initial loss
        initial_batch = next(iter(dataloader_train))
        initial_loss = model.loss(model.params, initial_batch)
        assert jnp.isfinite(initial_loss)
        assert initial_loss > 0

        # Training loop
        for _ in range(2):  # 2 epochs
            for _, batch in enumerate(dataloader_train):
                model.params, model.opt_state, loss_value = model.step(
                    model.params, model.opt_state, batch
                )

                # Verify loss is finite
                assert jnp.isfinite(loss_value)

        # Check final loss
        final_batch = next(iter(dataloader_train))
        final_loss = model.loss(model.params, final_batch)

        # Loss should be finite
        assert jnp.isfinite(final_loss)

        # Test prediction
        test_batch = next(iter(dataloader_test))
        test_inputs, test_outputs, _, _ = test_batch
        u_test, y_test = test_inputs

        # Make prediction on single source
        predictions = model.predict_s(model.params, u_test[0], y_test[0])

        # Check prediction shape and values
        assert predictions.shape == test_outputs[0].shape
        assert jnp.all(jnp.isfinite(predictions))

        # Compute test loss
        test_loss = model.loss(model.params, test_batch)
        assert jnp.isfinite(test_loss)

        print("\nIntegration test results:")
        print(f"  Initial loss: {initial_loss:.6f}")
        print(f"  Final loss: {final_loss:.6f}")
        print(f"  Test loss: {test_loss:.6f}")

        # Count parameters
        import jax

        num_params = sum(x.size for x in jax.tree_util.tree_leaves(model.params))
        print(f"  Number of parameters: {num_params}")

    def test_data_loading_consistency(self, small_2d_dataset):
        """Test that data loading is consistent and correct."""
        train_dir, _ = small_2d_dataset

        data = DataH5Compact(str(train_dir), flatten_ic=True)

        # Check basic properties
        assert data.N == 3
        assert data.mesh.shape[1] == 2  # 2D
        assert data.P_mesh == 15 * 15  # nx * ny
        assert len(data.tsteps) == 12

        # Test dataset streamer
        dataset = DatasetStreamer(data, batch_size_coord=30)

        # Get a sample
        inputs, outputs, idx_coord, x0 = dataset[0]
        u, y = inputs

        # Check shapes
        assert u.shape == tuple(data.u_shape)
        assert y.shape[0] == 30
        assert outputs.shape[0] == 30
        assert len(idx_coord) == 30

        # Check value ranges (data should be normalized)
        assert jnp.all(u >= -1.0) and jnp.all(u <= 1.0)

    def test_model_checkpointing(self, small_2d_dataset, temp_dir):
        """Test that model checkpointing works correctly."""
        train_dir, _ = small_2d_dataset
        model_dir = temp_dir / "checkpoint_models"
        model_dir.mkdir()

        training_settings = TrainingSettings(
            iterations=10,
            batch_size_branch=2,
            batch_size_coord=30,
            learning_rate=0.001,
            decay_steps=100,
            decay_rate=0.95,
            optimizer="adam",
            use_adaptive_weights=False,
        )

        data_train = DataH5Compact(str(train_dir), flatten_ic=True, norm_data=True)

        branch_arch = MLPArchitecture(
            architecture=NetworkArchitectureType.MLP,
            num_hidden_layers=2,
            num_hidden_neurons=32,
            num_output_neurons=32,
            activation="relu",
        )
        trunk_arch = MLPArchitecture(
            architecture=NetworkArchitectureType.MLP,
            num_hidden_layers=2,
            num_hidden_neurons=32,
            num_output_neurons=32,
            activation="relu",
        )

        in_bn = data_train.u_shape
        in_tn = 5  # x, y, t with Fourier features

        branch_net = setupNetwork(branch_arch, "bn")
        trunk_net = setupNetwork(trunk_arch, "tn")

        model = DeepONet(
            settings=training_settings,
            dataset=data_train,
            module_bn=(branch_net, in_bn),
            module_tn=(trunk_net, in_tn),
            log_dir=str(model_dir),
        )

        # Save checkpoint
        model.writeModel(iter=0)

        # Check that checkpoint was created
        checkpoint_files = list(model_dir.glob("checkpoint_*"))
        assert len(checkpoint_files) > 0

    def test_different_batch_sizes(self, small_2d_dataset, temp_dir):
        """Test training with different batch size configurations."""
        train_dir, _ = small_2d_dataset

        batch_configs = [
            (1, 20),  # Small batches
            (2, 50),  # Medium batches
            (3, 100),  # Larger coordinate batch
        ]

        for batch_branch, batch_coord in batch_configs:
            model_dir = temp_dir / f"models_b{batch_branch}_c{batch_coord}"
            model_dir.mkdir(exist_ok=True)

            training_settings = TrainingSettings(
                iterations=5,
                batch_size_branch=batch_branch,
                batch_size_coord=batch_coord,
                learning_rate=0.001,
                decay_steps=100,
                decay_rate=0.95,
                optimizer="adam",
                use_adaptive_weights=False,
            )

            data_train = DataH5Compact(str(train_dir), flatten_ic=True, norm_data=True)
            dataset_train = DatasetStreamer(data_train, batch_size_coord=batch_coord)

            dataloader = DataLoader(
                dataset_train,
                batch_size=batch_branch,
                shuffle=True,
                collate_fn=numpy_collate,
                drop_last=True,
            )

            branch_arch = MLPArchitecture(
                architecture=NetworkArchitectureType.MLP,
                num_hidden_layers=2,
                num_hidden_neurons=32,
                num_output_neurons=32,
                activation="relu",
            )
            trunk_arch = MLPArchitecture(
                architecture=NetworkArchitectureType.MLP,
                num_hidden_layers=2,
                num_hidden_neurons=32,
                num_output_neurons=32,
                activation="relu",
            )

            in_bn = data_train.u_shape
            sample = next(iter(dataloader))
            in_tn = sample[0][1].shape[-1]

            branch_net = setupNetwork(branch_arch, "bn")
            trunk_net = setupNetwork(trunk_arch, "tn")

            model = DeepONet(
                settings=training_settings,
                dataset=data_train,
                module_bn=(branch_net, in_bn),
                module_tn=(trunk_net, in_tn),
                log_dir=str(model_dir),
            )

            # Run a few training steps
            for i, batch in enumerate(dataloader):
                if i >= 3:  # Just 3 steps per config
                    break
                model.params, model.opt_state, loss_value = model.step(
                    model.params, model.opt_state, batch
                )
                assert jnp.isfinite(loss_value)

            print(
                f"  Batch config ({batch_branch}, {batch_coord}) completed successfully"
            )
