# ==============================================================================
# Copyright 2025 Technical University of Denmark
# Author: Nikolas Borrel-Jensen
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
"""Unit tests for loss functions."""

import jax.numpy as jnp
import pytest

from deeponet_acoustics.models.loss_functions import loss


@pytest.mark.unit
class TestLossFunctions:
    """Test suite for loss functions."""

    def test_loss_basic(self):
        """Test basic loss calculation."""

        def mock_branch_net(params, u):
            """Mock branch network that returns identity."""
            return u

        def mock_operator_net(params, branch_latent, y):
            """Mock operator network."""
            return jnp.sum(branch_latent) + jnp.sum(y)

        # Create mock batch data
        batch_size = 2
        coord_batch_size = 10
        input_dim = 5

        u = jnp.ones((batch_size, input_dim))
        y = jnp.ones((batch_size, coord_batch_size, 3))
        outputs = jnp.ones((batch_size, coord_batch_size))
        idx_coord = jnp.arange(coord_batch_size * batch_size).reshape(batch_size, coord_batch_size)

        batch = ((u, y), outputs, idx_coord)
        params = {}

        loss_value = loss(
            params, batch, mock_branch_net, mock_operator_net, apply_adaptive_weights=False
        )

        # Loss should be a scalar
        assert loss_value.shape == ()
        # Loss should be non-negative
        assert loss_value >= 0

    def test_loss_perfect_prediction(self):
        """Test loss when predictions are perfect."""

        def mock_branch_net(params, u):
            return jnp.zeros_like(u)

        def mock_operator_net(params, branch_latent, y):
            # Return values that match outputs
            return 0.0

        batch_size = 2
        coord_batch_size = 10

        u = jnp.ones((batch_size, 5))
        y = jnp.ones((batch_size, coord_batch_size, 3))
        outputs = jnp.zeros((batch_size, coord_batch_size))  # Matches prediction
        idx_coord = jnp.arange(coord_batch_size * batch_size).reshape(batch_size, coord_batch_size)

        batch = ((u, y), outputs, idx_coord)
        params = {}

        loss_value = loss(
            params, batch, mock_branch_net, mock_operator_net, apply_adaptive_weights=False
        )

        # Loss should be very small (zero or near-zero)
        assert jnp.isclose(loss_value, 0.0, atol=1e-5)

    def test_loss_with_different_batch_sizes(self):
        """Test loss with various batch sizes."""

        def mock_branch_net(params, u):
            return u * 2

        def mock_operator_net(params, branch_latent, y):
            return jnp.sum(branch_latent) * jnp.sum(y) * 0.01

        for batch_size in [1, 2, 4, 8]:
            coord_batch_size = 5

            u = jnp.ones((batch_size, 10))
            y = jnp.ones((batch_size, coord_batch_size, 4))
            outputs = jnp.ones((batch_size, coord_batch_size))
            idx_coord = jnp.arange(coord_batch_size * batch_size).reshape(batch_size, coord_batch_size)

            batch = ((u, y), outputs, idx_coord)
            params = {}

            loss_value = loss(
                params, batch, mock_branch_net, mock_operator_net, apply_adaptive_weights=False
            )

            assert loss_value.shape == ()
            assert loss_value >= 0

    def test_loss_output_shapes(self):
        """Test that loss always outputs scalar regardless of input shapes."""

        def mock_branch_net(params, u):
            # Return latent representation
            return jnp.ones(10)

        def mock_operator_net(params, branch_latent, y):
            # operator_net takes single coord point, returns scalar
            return jnp.sum(branch_latent) * jnp.sum(y) * 0.1

        test_cases = [
            (2, 10, 5, 4),  # (batch, coord_batch, input_dim, coord_dim)
            (1, 20, 10, 3),
            (4, 5, 8, 5),
        ]

        for batch_size, coord_batch_size, input_dim, coord_dim in test_cases:
            u = jnp.ones((batch_size, input_dim))
            y = jnp.ones((batch_size, coord_batch_size, coord_dim))
            outputs = jnp.ones((batch_size, coord_batch_size))
            idx_coord = jnp.arange(coord_batch_size * batch_size).reshape(batch_size, coord_batch_size)

            batch = ((u, y), outputs, idx_coord)
            params = {}

            loss_value = loss(
                params, batch, mock_branch_net, mock_operator_net, apply_adaptive_weights=False
            )

            # Loss must always be scalar
            assert loss_value.shape == ()
            assert loss_value >= 0
