# ==============================================================================
# Copyright 2025 Technical University of Denmark
# Author: Nikolas Borrel-Jensen
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
"""Unit tests for utility functions."""

import jax.numpy as jnp
import numpy as np
import pytest

from deeponet_acoustics.utils.utils import (
    expandCnnData,
    getNearestFromCoordinates,
)


@pytest.mark.unit
class TestExpandCnnData:
    """Test suite for expandCnnData function."""

    def test_expand_2d_data(self):
        """Test expanding 2D data for CNN."""
        # 2D input: (height, width)
        u = jnp.ones((10, 10))
        expanded = expandCnnData(u)

        # Should add batch and channel dimensions: (1, height, width, 1)
        assert expanded.shape == (1, 10, 10, 1)

    def test_expand_3d_data(self):
        """Test expanding 3D data for CNN."""
        # 3D input: (depth, height, width)
        u = jnp.ones((5, 10, 10))
        expanded = expandCnnData(u)

        # Should add batch and channel dimensions: (1, depth, height, width, 1)
        assert expanded.shape == (1, 5, 10, 10, 1)

    def test_expand_preserves_values(self):
        """Test that expansion preserves values."""
        u = jnp.arange(12).reshape(3, 4)
        expanded = expandCnnData(u)

        # Values should be preserved
        assert jnp.allclose(expanded[0, :, :, 0], u)

    def test_expand_invalid_dimension(self):
        """Test that invalid dimensions raise an exception."""
        # 1D array should raise exception
        u = jnp.ones(10)

        with pytest.raises(Exception):
            expandCnnData(u)

        # 4D array should raise exception
        u = jnp.ones((2, 3, 4, 5))

        with pytest.raises(Exception):
            expandCnnData(u)


@pytest.mark.unit
class TestGetNearestFromCoordinates:
    """Test suite for getNearestFromCoordinates function."""

    def test_nearest_1d(self):
        """Test finding nearest coordinates in 1D grid."""
        grid = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        recvs_per_srcs = np.array([[[0.5]], [[2.3]], [[1.1]], [[3.8]]])

        r0, r0_indxs = getNearestFromCoordinates(grid, recvs_per_srcs)

        assert (r0 == np.array([[[0.0]], [[2.0]], [[1.0]], [[4.0]]])).all()
        assert (r0_indxs == np.array([[0], [2], [1], [4]])).all()

    def test_nearest_2d(self):
        """Test finding nearest coordinates in 2D grid."""
        # Create a simple 2D grid
        x = np.linspace(0, 4, 5)
        y = np.linspace(0, 4, 5)
        xx, yy = np.meshgrid(x, y)
        grid = np.column_stack([xx.flatten(), yy.flatten()])

        # Query points
        recvs_per_srcs = np.array([[[0.5, 0.5]], [[2.0, 2.0]]])

        r0, r0_indxs = getNearestFromCoordinates(grid, recvs_per_srcs)

        assert (r0 == np.array([[[0.0], [0.0]], [[2.0], [2.0]]])).all()
        assert (grid[r0_indxs] == r0).all()

    def test_nearest_3d(self):
        """Test finding nearest coordinates in 3D grid."""
        # Create a simple 3D grid
        x = np.linspace(0, 2, 3)
        y = np.linspace(0, 2, 3)
        z = np.linspace(0, 2, 3)
        xx, yy, zz = np.meshgrid(x, y, z)
        grid = np.column_stack([xx.flatten(), yy.flatten(), zz.flatten()])

        recvs_per_srcs = np.array([[[1.0, 1.0, 1.0]], [[0.3, 0.4, 2.3]]])

        r0, r0_indxs = getNearestFromCoordinates(grid, recvs_per_srcs)

        assert (r0 == np.array([[[1.0, 1.0, 1.0]], [[0.0, 0.0, 2.0]]])).all()
        assert (grid[r0_indxs] == r0).all()
