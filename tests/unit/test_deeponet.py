# ==============================================================================
# Copyright 2025 Technical University of Denmark
# Author: Nikolas Borrel-Jensen
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
"""Unit tests for DeepONet model."""

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
import pytest

from deeponet_acoustics.models.datastructures import (
    NetworkArchitectureType,
    TrainingSettings,
)
from deeponet_acoustics.models.deeponet import DeepONet, exponential_decay


@pytest.mark.unit
class TestExponentialDecay:
    """Test suite for exponential decay scheduler."""

    def test_initial_step_size(self):
        """Test that initial step returns the base step size."""
        step_size = 0.001
        decay_steps = 1000
        decay_rate = 0.9

        schedule = exponential_decay(step_size, decay_steps, decay_rate)

        assert np.isclose(schedule(0), step_size)

    def test_decay_progression(self):
        """Test that learning rate decays over time."""
        step_size = 0.001
        decay_steps = 1000
        decay_rate = 0.9

        schedule = exponential_decay(step_size, decay_steps, decay_rate)

        lr_0 = schedule(0)
        lr_500 = schedule(500)
        lr_1000 = schedule(1000)

        # Learning rate should decrease
        assert lr_0 > lr_500 > lr_1000

    def test_decay_with_offset(self):
        """Test decay schedule with step offset."""
        step_size = 0.001
        decay_steps = 1000
        decay_rate = 0.9
        step_offset = 500

        schedule = exponential_decay(step_size, decay_steps, decay_rate, step_offset)

        # At step 0 with offset 500, should equal schedule at step 500 without offset
        schedule_no_offset = exponential_decay(step_size, decay_steps, decay_rate)

        assert np.isclose(schedule(0), schedule_no_offset(500))


@pytest.mark.unit
class TestDeepONetComponents:
    """Test suite for DeepONet components."""

    @pytest.fixture
    def simple_fnn_module(self):
        """Create a simple MLP module for testing."""

        class SimpleMLP(nn.Module):
            network_type: NetworkArchitectureType = NetworkArchitectureType.MLP

            @nn.compact
            def __call__(self, x):
                x = nn.Dense(32)(x)
                x = nn.relu(x)
                x = nn.Dense(16)(x)
                return x

        return SimpleMLP()

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset interface."""

        class MockDataset:
            N = 10  # Number of sources
            P = 500  # Total coordinate points
            u_shape = [50]  # Input shape
            Pmesh = 50
            tsteps = np.linspace(0, 1, 10)

            def __init__(self):
                pass

        return MockDataset()

    @pytest.fixture
    def basic_training_settings(self):
        """Create basic training settings."""
        return TrainingSettings(
            iterations=100,
            batch_size_branch=2,
            batch_size_coord=50,
            learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9,
            optimizer="adam",
            use_adaptive_weights=False,
        )

    def test_deeponet_initialization(
        self, simple_fnn_module, mock_dataset, basic_training_settings, temp_dir
    ):
        """Test DeepONet initialization without transfer learning."""
        branch_module = simple_fnn_module
        trunk_module = simple_fnn_module

        model = DeepONet(
            settings=basic_training_settings,
            dataset=mock_dataset,
            module_bn=(branch_module, 50),
            module_tn=(trunk_module, 4),
            log_dir=str(temp_dir),
            transfer_learning=None,
        )

        # Check that model has required attributes
        assert hasattr(model, "params")
        assert hasattr(model, "branch_apply")
        assert hasattr(model, "trunk_apply")
        assert hasattr(model, "optimizer")
        assert hasattr(model, "opt_state")

        # Check params structure
        assert "bn" in model.params
        assert "tn" in model.params
        assert "b0" in model.params

    def test_deeponet_operator_net(
        self, simple_fnn_module, mock_dataset, basic_training_settings, temp_dir
    ):
        """Test operator network forward pass."""
        branch_module = simple_fnn_module
        trunk_module = simple_fnn_module

        model = DeepONet(
            settings=basic_training_settings,
            dataset=mock_dataset,
            module_bn=(branch_module, 50),
            module_tn=(trunk_module, 4),
            log_dir=str(temp_dir),
        )

        # Create test inputs
        branch_latent = jnp.ones(16)  # Output from branch network
        y = jnp.ones(4)  # Trunk network input

        output = model.operator_net(model.params, branch_latent, y)

        # Output should be a scalar
        assert output.shape == ()
        assert jnp.isfinite(output)

    def test_deeponet_branch_net(
        self, simple_fnn_module, mock_dataset, basic_training_settings, temp_dir
    ):
        """Test branch network forward pass."""
        branch_module = simple_fnn_module
        trunk_module = simple_fnn_module

        model = DeepONet(
            settings=basic_training_settings,
            dataset=mock_dataset,
            module_bn=(branch_module, 50),
            module_tn=(trunk_module, 4),
            log_dir=str(temp_dir),
        )

        # Create test input
        u = jnp.ones(50)  # Branch network input

        output = model.branch_net(model.params, u)

        # Output should have expected dimension
        assert output.ndim == 1
        assert output.shape[0] == 16  # Should match trunk output size

    def test_deeponet_predict(
        self, simple_fnn_module, mock_dataset, basic_training_settings, temp_dir
    ):
        """Test prediction function."""
        branch_module = simple_fnn_module
        trunk_module = simple_fnn_module

        model = DeepONet(
            settings=basic_training_settings,
            dataset=mock_dataset,
            module_bn=(branch_module, 50),
            module_tn=(trunk_module, 4),
            log_dir=str(temp_dir),
        )

        # Create test inputs
        U_star = jnp.ones(50)  # Single branch input
        Y_star = jnp.ones((10, 4))  # Multiple coordinate points

        predictions = model.predict_s(model.params, U_star, Y_star)

        # Predictions should have same length as coordinate points
        assert predictions.shape[0] == Y_star.shape[0]
        assert jnp.all(jnp.isfinite(predictions))

    def test_deeponet_loss_computation(
        self, simple_fnn_module, mock_dataset, basic_training_settings, temp_dir
    ):
        """Test loss computation."""
        branch_module = simple_fnn_module
        trunk_module = simple_fnn_module

        model = DeepONet(
            settings=basic_training_settings,
            dataset=mock_dataset,
            module_bn=(branch_module, 50),
            module_tn=(trunk_module, 4),
            log_dir=str(temp_dir),
        )

        # Create mock batch
        batch_size = 2
        n_coords = 20

        u = jnp.ones((batch_size, 50))
        y = jnp.ones((batch_size, n_coords, 4))
        outputs = jnp.ones((batch_size, n_coords))
        idx_coord = jnp.arange(n_coords * batch_size).reshape(batch_size, n_coords)

        batch = ((u, y), outputs, idx_coord)

        loss_value = model.loss(model.params, batch)

        # Loss should be a scalar
        assert loss_value.shape == ()
        assert loss_value >= 0
        assert jnp.isfinite(loss_value)

    def test_deeponet_step(
        self, simple_fnn_module, mock_dataset, basic_training_settings, temp_dir
    ):
        """Test single training step."""
        branch_module = simple_fnn_module
        trunk_module = simple_fnn_module

        model = DeepONet(
            settings=basic_training_settings,
            dataset=mock_dataset,
            module_bn=(branch_module, 50),
            module_tn=(trunk_module, 4),
            log_dir=str(temp_dir),
        )

        # Create mock batch
        batch_size = 2
        n_coords = 20

        u = jnp.ones((batch_size, 50))
        y = jnp.ones((batch_size, n_coords, 4))
        outputs = jnp.zeros((batch_size, n_coords))
        idx_coord = jnp.arange(n_coords * batch_size).reshape(batch_size, n_coords)

        batch = ((u, y), outputs, idx_coord)

        # Take one training step
        params_old = model.params
        new_params, new_opt_state, loss_value = model.step(
            model.params, model.opt_state, batch
        )

        # Parameters should be updated
        assert not jnp.allclose(
            new_params["bn"]["params"]["Dense_0"]["kernel"],
            params_old["bn"]["params"]["Dense_0"]["kernel"],
        )

        # Loss should be computed
        assert jnp.isfinite(loss_value)

    def test_deeponet_with_adaptive_weights(
        self, simple_fnn_module, mock_dataset, basic_training_settings, temp_dir
    ):
        """Test DeepONet with adaptive weights enabled."""
        # Enable adaptive weights
        settings = TrainingSettings(
            iterations=100,
            batch_size_branch=2,
            batch_size_coord=50,
            learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9,
            optimizer="adam",
            use_adaptive_weights=True,
        )

        branch_module = simple_fnn_module
        trunk_module = simple_fnn_module

        model = DeepONet(
            settings=settings,
            dataset=mock_dataset,
            module_bn=(branch_module, 50),
            module_tn=(trunk_module, 4),
            log_dir=str(temp_dir),
        )

        # Check that adaptive weights are initialized
        assert "adaptive_weights" in model.params
        expected_shape = (min(settings.batch_size_branch, mock_dataset.N) *
                         min(settings.batch_size_coord, mock_dataset.P),)
        assert model.params["adaptive_weights"].shape == expected_shape
