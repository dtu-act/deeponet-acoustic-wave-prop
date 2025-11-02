# ==============================================================================
# Copyright 2025 Technical University of Denmark
# Author: Nikolas Borrel-Jensen
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
"""Unit tests for data generators and data handlers."""

import jax.numpy as jnp
import numpy as np
import pytest

from deeponet_acoustics.datahandlers.datagenerators import (
    DataH5Compact,
    DatasetStreamer,
    _calculate_u_pressure_minmax,
    numpy_collate,
)
from deeponet_acoustics.models.datastructures import SimulationDataType


@pytest.mark.unit
class TestDataH5Compact:
    """Test suite for DataH5Compact class."""

    def test_initialization(self, mock_h5_dataset_2d):
        """Test basic initialization of DataH5Compact."""
        data_path, files = mock_h5_dataset_2d

        data = DataH5Compact(
            str(data_path), tmax=1.0, t_norm=1.0, flatten_ic=True, data_prune=1
        )

        assert data.N == len(files)
        assert data.simulationDataType == SimulationDataType.H5COMPACT
        assert data.mesh.shape[1] == 2  # 2D data
        assert len(data.datasets) == len(files)
        assert data.P > 0
        assert data.P_mesh > 0

    def test_mesh_properties(self, mock_h5_dataset_2d):
        """Test mesh properties and dimensions."""
        data_path, _ = mock_h5_dataset_2d

        data = DataH5Compact(str(data_path), flatten_ic=True)

        assert data.mesh.ndim == 2
        assert data.P_mesh == data.mesh.shape[0]
        assert data.P == data.P_mesh * len(data.tsteps)
        assert len(data.tt) == data.P

    def test_data_pruning(self, mock_h5_dataset_2d):
        """Test data pruning functionality."""
        data_path, _ = mock_h5_dataset_2d

        data_full = DataH5Compact(str(data_path), data_prune=1)
        data_pruned = DataH5Compact(str(data_path), data_prune=2)

        # Pruned data should have fewer mesh points
        assert data_pruned.P_mesh < data_full.P_mesh
        assert data_pruned.P_mesh == (data_full.P_mesh + 1) // 2

    def test_time_truncation(self, mock_h5_dataset_2d):
        """Test time truncation with tmax parameter."""
        data_path, _ = mock_h5_dataset_2d

        tmax_truncated = 0.5
        data_full = DataH5Compact(str(data_path), tmax=float("inf"))
        data_truncated = DataH5Compact(str(data_path), tmax=tmax_truncated)

        # Truncated data should have fewer timesteps
        assert len(data_truncated.tsteps) <= len(data_full.tsteps)
        assert np.all(data_truncated.tsteps <= tmax_truncated)

    def test_normalization(self, mock_h5_dataset_2d):
        """Test spatial and temporal normalization."""
        data_path, _ = mock_h5_dataset_2d

        data_normalized = DataH5Compact(str(data_path), norm_data=True)
        data_unnormalized = DataH5Compact(str(data_path), norm_data=False)

        # Normalized mesh should be in [-1, 1]
        assert np.all(data_normalized.mesh >= -1.0)
        assert np.all(data_normalized.mesh <= 1.0)

        # Check that normalization was applied
        # (meshes are already in [-1,1] in fixture, so test normalization properties)
        assert data_normalized.normalize_data
        assert not data_unnormalized.normalize_data

        # Test that xmin and xmax are the same
        # min / max are set for the original non-normalized data (TODO; rename)
        assert np.isclose(data_normalized.xmin, data_unnormalized.xmin)
        assert np.isclose(data_normalized.xmax, data_unnormalized.xmax)

    def test_spatial_normalization_functions(self, mock_h5_dataset_2d):
        """Test normalization and denormalization functions."""
        data_path, _ = mock_h5_dataset_2d

        data = DataH5Compact(str(data_path), norm_data=False)

        # Test spatial normalization
        test_coords = np.array([[data.xmin, data.xmin], [data.xmax, data.xmax]])
        normalized = data.normalize_spatial(test_coords)

        # Check boundaries map to [-1, 1]
        assert np.allclose(normalized, [[-1, -1], [1, 1]])

        # Test round-trip
        denormalized = data.denormalize_spatial(normalized)
        assert np.allclose(test_coords, denormalized)

    def test_temporal_normalization_functions(self, mock_h5_dataset_2d):
        """Test temporal normalization and denormalization."""
        data_path, _ = mock_h5_dataset_2d

        data = DataH5Compact(str(data_path), norm_data=False)

        # Test temporal normalization
        test_times = np.array([0.0, 1.0, 2.0])
        normalized = data.normalize_temporal(test_times)
        assert np.min(normalized.flatten()) == 0

        denormalized = data.denormalize_temporal(normalized)

        # Check round-trip
        assert np.allclose(test_times, denormalized)

    def test_xxyyzztt_property(self, mock_h5_dataset_2d):
        """Test combined spatial-temporal coordinate array."""
        data_path, _ = mock_h5_dataset_2d

        data = DataH5Compact(str(data_path))

        xxyyzztt = data.xxyyzztt

        # Should have shape (P, spatial_dim + 1)
        assert xxyyzztt.shape[0] == data.P
        assert xxyyzztt.shape[1] == data.mesh.shape[1] + 1  # spatial dims + time

    def test_u_shape_flattened(self, mock_h5_dataset_2d):
        """Test u_shape when flattened."""
        data_path, _ = mock_h5_dataset_2d

        data = DataH5Compact(str(data_path), flatten_ic=True)

        assert len(data.u_shape) == 1
        assert data.u_shape[0] > 0

    def test_tags_field(self, mock_h5_dataset_2d):
        """Test field tags."""
        data_path, _ = mock_h5_dataset_2d

        data = DataH5Compact(str(data_path))

        assert data.tags_field == ["/pressures"]
        assert data.tag_ufield == "/upressures"


@pytest.mark.unit
class TestDatasetStreamer:
    """Test suite for DatasetStreamer class."""

    def test_initialization(self, mock_h5_dataset_2d):
        """Test DatasetStreamer initialization."""
        data_path, files = mock_h5_dataset_2d

        data = DataH5Compact(str(data_path))
        dataset = DatasetStreamer(data, batch_size_coord=100)

        assert len(dataset) == len(files)
        assert dataset.N == len(files)
        assert dataset.P == data.P
        assert dataset.P_mesh == data.P_mesh

    def test_getitem(self, mock_h5_dataset_2d):
        """Test getting individual items from dataset."""
        data_path, _ = mock_h5_dataset_2d

        data = DataH5Compact(str(data_path))
        dataset = DatasetStreamer(data, batch_size_coord=50)

        # Get first item
        inputs, outputs, idx_coord, x0 = dataset[0]
        u, y = inputs

        # Check shapes
        assert u.shape == tuple(data.u_shape)
        assert y.shape[0] == 50  # batch_size_coord
        assert outputs.shape[0] == 50
        assert idx_coord.shape[0] == 50

        # Check value ranges (normalized)
        assert jnp.all(u >= -1) and jnp.all(u <= 1)

    def test_full_dataset_mode(self, mock_h5_dataset_2d):
        """Test full dataset mode (batch_size_coord=-1)."""
        data_path, _ = mock_h5_dataset_2d

        data = DataH5Compact(str(data_path))
        dataset = DatasetStreamer(data, batch_size_coord=-1)

        inputs, outputs, idx_coord, x0 = dataset[0]
        u, y = inputs

        # Should return full dataset
        assert y.shape[0] == data.P
        assert outputs.shape[0] == data.P
        assert idx_coord.shape[0] == data.P

    def test_coordinate_sampling(self, mock_h5_dataset_2d):
        """Test that coordinate sampling works correctly."""
        data_path, _ = mock_h5_dataset_2d

        data = DataH5Compact(str(data_path))
        batch_size = 30
        dataset = DatasetStreamer(data, batch_size_coord=batch_size)

        # Get multiple samples to check randomness
        _, _, idx1, _ = dataset[0]
        _, _, idx2, _ = dataset[0]

        # Indices should be different (with high probability)
        # since sampling is random
        assert (idx1 != idx2).any()
        assert idx1.shape[0] == batch_size
        assert idx2.shape[0] == batch_size

    def test_feature_extraction_function(self, mock_h5_dataset_2d):
        """Test custom feature extraction function."""
        data_path, _ = mock_h5_dataset_2d

        data = DataH5Compact(str(data_path))

        # Define a feature extraction function that doubles coordinates
        def double_coords(y):
            return y * 2

        dataset = DatasetStreamer(
            data, batch_size_coord=50, y_feat_extract_fn=double_coords
        )

        inputs, _, _, _ = dataset[0]
        _, y = inputs

        # Feature extraction should be applied
        assert y.shape[0] == 50


@pytest.mark.unit
class TestNumpyCollate:
    """Test suite for numpy_collate function."""

    def test_collate_arrays(self):
        """Test collating numpy arrays."""
        batch = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]

        result = numpy_collate(batch)

        assert isinstance(result, jnp.ndarray)
        assert result.shape == (3, 3)

    def test_collate_tuples(self):
        """Test collating tuples of arrays."""
        batch = [
            (np.array([1, 2]), np.array([3, 4])),
            (np.array([5, 6]), np.array([7, 8])),
        ]

        result = numpy_collate(batch)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].shape == (2, 2)
        assert result[1].shape == (2, 2)

    def test_collate_nested(self):
        """Test collating nested structures."""
        batch = [
            ((np.array([1]), np.array([2])), np.array([3])),
            ((np.array([4]), np.array([5])), np.array([6])),
        ]

        result = numpy_collate(batch)

        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], list)


@pytest.mark.unit
class TestUPressures:
    """Test suite for u_pressures functionality."""

    def test_calculate_u_pressure_minmax(self):
        """Test basic min/max calculation with mock datasets."""

        # Create mock datasets with known min/max values
        class MockDataset:
            def __init__(self, data):
                self.data = {"/upressures": data}

            def __getitem__(self, key):
                return self.data[key]

        # Create datasets with known ranges
        datasets = [
            MockDataset(np.array([-1.5, 0.0, 1.5])),
            MockDataset(np.array([-2.0, 0.5, 2.0])),
            MockDataset(np.array([-1.0, 0.0, 1.0])),
        ]

        p_min, p_max = _calculate_u_pressure_minmax(datasets, "/upressures")

        assert p_min == -2.0
        assert p_max == 2.0

    def test_u_pressures_normalization(self, mock_h5_dataset_2d):
        """Test that u_pressures returns properly normalized values."""
        data_path, _ = mock_h5_dataset_2d

        data = DataH5Compact(str(data_path))

        # Get normalized pressures
        u_norm = data.u_pressures(0)

        # Should be normalized to [-1, 1]
        assert np.all(u_norm >= -1.0)
        assert np.all(u_norm <= 1.0)

        # Check shape matches u_shape
        assert u_norm.shape == tuple(data.u_shape)

    def test_u_pressures_different_indices(self, mock_h5_dataset_2d):
        """Test u_pressures with different dataset indices."""
        data_path, _ = mock_h5_dataset_2d

        data = DataH5Compact(str(data_path))

        # Get pressures for different indices
        u0 = data.u_pressures(0)
        u1 = data.u_pressures(1)

        # Should have same shape
        assert u0.shape == u1.shape

        # Should both be normalized
        assert np.all(u0 >= -1.0) and np.all(u0 <= 1.0)
        assert np.all(u1 >= -1.0) and np.all(u1 <= 1.0)
