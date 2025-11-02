# ==============================================================================
# Copyright 2025 Technical University of Denmark
# Author: Nikolas Borrel-Jensen
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import itertools
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import h5py
import jax.numpy as jnp
import numpy as np
from torch.utils.data import Dataset

import deeponet_acoustics.datahandlers.io as IO
from deeponet_acoustics.datahandlers.io import XdmfReader
from deeponet_acoustics.models.datastructures import SimulationDataType


def _calculate_u_pressure_minmax(
    datasets: list, tag_ufield: str, max_samples: int = 500
) -> tuple[float, float]:
    """Calculate min and max pressure values from datasets.

    Args:
        datasets: List of datasets (h5py.File or mock objects)
        tag_ufield: Tag/key for accessing pressure data in datasets
        max_samples: Maximum number of samples to use for estimation (default: 500)

    Returns:
        Tuple of (p_min, p_max) as floats
    """
    num_samples = min(max_samples, len(datasets))
    p_min_vals = []
    p_max_vals = []

    print(f"Estimating pressure min/max from {num_samples} samples...")
    for i in range(num_samples):
        upressures = datasets[i][tag_ufield][:]
        p_min_vals.append(np.min(upressures))
        p_max_vals.append(np.max(upressures))

    p_min = float(np.min(p_min_vals))
    p_max = float(np.max(p_max_vals))
    print(f"Pressure range: [{p_min:.4f}, {p_max:.4f}]")

    return p_min, p_max


def _normalize_spatial(data: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    """Normalize spatial coordinates to [-1, 1] range.

    Args:
        data: Data to normalize
        xmin: Minimum value for normalization
        xmax: Maximum value for normalization

    Returns:
        Normalized data in [-1, 1] range
    """
    return 2 * (data - xmin) / (xmax - xmin) - 1


def _normalize_temporal(data: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    """Normalize temporal coordinates.

    Args:
        data: Data to normalize
        xmin: Minimum spatial value for normalization
        xmax: Maximum spatial value for normalization

    Returns:
        Normalized temporal data
    """
    return data / (xmax - xmin) / 2


def _denormalize_spatial(data: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    """Denormalize spatial coordinates from [-1, 1] range.

    Args:
        data: Normalized data to denormalize
        xmin: Minimum value for denormalization
        xmax: Maximum value for denormalization

    Returns:
        Denormalized spatial data
    """
    return (data + 1) / 2 * (xmax - xmin) + xmin


def _denormalize_temporal(data: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    """Denormalize temporal coordinates.

    Args:
        data: Normalized data to denormalize
        xmin: Minimum spatial value for denormalization
        xmax: Maximum spatial value for denormalization

    Returns:
        Denormalized temporal data
    """
    return data * 2 * (xmax - xmin)


class DataInterface(ABC):
    # Required attributes that concrete classes must define
    simulationDataType: SimulationDataType
    datasets: list[h5py.File]
    mesh: np.ndarray
    tags_field: list[str]
    tag_ufield: str
    tt: np.ndarray
    data_prune: int
    N: int
    u_shape: np.ndarray | list[int]
    tsteps: np.ndarray

    @property
    @abstractmethod
    def P(self) -> int:
        """Total number of time/space points."""
        pass

    @abstractmethod
    def u_pressures(self, idx: int) -> np.ndarray:
        """Get normalized u pressures for a given dataset index."""
        pass


class DataXdmf(DataInterface):
    simulationDataType: SimulationDataType = SimulationDataType.XDMF

    datasets: list[h5py.File]
    mesh: np.ndarray
    u_shape: list[int]
    tsteps: np.ndarray
    tt: np.ndarray
    tags_field: list[str]
    tag_ufield: str
    data_prune: int
    N: int

    xmin: float
    xmax: float
    normalize_data: bool

    def __init__(
        self,
        data_path,
        tmax=float("inf"),
        t_norm=1,
        flatten_ic=True,
        data_prune=1,
        norm_data=False,
        MAXNUM_DATASETS=sys.maxsize,
    ):
        filenames_xdmf = IO.pathsToFileType(data_path, ".xdmf", exclude="rectilinear")
        self.normalize_data = norm_data

        # NOTE: we assume meshes, tags, etc are the same across all xdmf datasets
        xdmf = XdmfReader(filenames_xdmf[0], tmax=tmax / t_norm)
        self.tags_field = xdmf.tags_field
        self.tag_ufield = xdmf.tag_ufield
        self.data_prune = data_prune
        self.tsteps = xdmf.tsteps * t_norm

        with h5py.File(xdmf.filenameH5) as r:
            self.mesh = np.array(r[xdmf.tag_mesh][:: self.data_prune])
            self.xmin, self.xmax = np.min(self.mesh), np.max(self.mesh)
            umesh_obj = r[xdmf.tag_umesh]
            umesh = np.array(umesh_obj[:])
            self.u_shape = (
                [len(umesh)]
                if flatten_ic
                else jnp.array(umesh_obj.attrs[xdmf.tag_ushape][:], dtype=int).tolist()
            )

        if norm_data:
            self.mesh = self.normalizeSpatial(self.mesh)
            self.tsteps = self.normalizeTemporal(self.tsteps)

        self.tt = np.repeat(self.tsteps, self.mesh.shape[0])

        self.datasets = []
        for i in range(0, min(MAXNUM_DATASETS, len(filenames_xdmf))):
            filename = filenames_xdmf[i]
            if Path(filename).exists():
                xdmf = XdmfReader(filename, tmax / t_norm)
                self.datasets.append(
                    h5py.File(xdmf.filenameH5, "r")
                )  # add file handles and keeps open
            else:
                print(f"Could not be found (ignoring): {filename}")

        self.N = len(self.datasets)

        # Calculate min and max pressure values from sampled datasets
        self._u_p_min, self._u_p_max = _calculate_u_pressure_minmax(
            self.datasets, self.tag_ufield
        )

    # --- required abstract properties implemented ---
    @property
    def Pmesh(self):
        """Total number of mesh points."""
        return self.mesh.shape[0]

    @property
    def P(self):
        """Total number of time/space points."""
        return self.Pmesh * len(self.tsteps)

    def normalizeSpatial(self, data):
        return _normalize_spatial(data, self.xmin, self.xmax)

    def normalizeTemporal(self, data):
        return _normalize_temporal(data, self.xmin, self.xmax)

    def u_pressures(self, idx: int) -> np.ndarray:
        """Get normalized u pressures for a given dataset index."""
        dataset = self.datasets[idx]
        u_norm = _normalize_spatial(
            dataset[self.tag_ufield][:], self._u_p_min, self._u_p_max
        )
        return jnp.reshape(u_norm, self.u_shape)

    def __del__(self):
        for dataset in self.datasets:
            dataset.close()


class DataH5Compact(DataInterface):
    simulationDataType: SimulationDataType = SimulationDataType.H5COMPACT

    datasets: list[h5py.File]
    mesh: np.ndarray
    u_shape: np.ndarray
    tsteps: np.ndarray
    tt: np.ndarray
    tags_field: list[str]
    tag_ufield: str
    data_prune: int
    N: int

    xmin: float
    xmax: float
    normalize_data: bool
    conn: np.ndarray

    # MAXNUM_DATASETS: SET TO E.G: 500 WHEN DEBUGGING ON MACHINES WITH LESS RESOURCES
    def __init__(
        self,
        data_path,
        tmax=float("inf"),
        t_norm=1,
        flatten_ic=True,
        data_prune=1,
        norm_data=False,
        MAXNUM_DATASETS=sys.maxsize,
    ):
        filenamesH5 = IO.pathsToFileType(data_path, ".h5", exclude="rectilinear")
        self.data_prune = data_prune
        self.normalize_data = norm_data

        # NOTE: we assume meshes, tags, etc are the same accross all xdmf datasets
        tag_mesh = "/mesh"
        tag_conn = "/conn"
        tag_umesh = "/umesh"
        tag_ushape = "umesh_shape"
        self.tags_field = ["/pressures"]
        self.tag_ufield = "/upressures"

        with h5py.File(filenamesH5[0]) as r:
            self.mesh = np.array(r[tag_mesh][:: self.data_prune])
            self.conn = (
                np.array(r[tag_conn])
                if self.data_prune == 1 and tag_conn in r
                else np.array([])
            )
            self.xmin, self.xmax = np.min(self.mesh), np.max(self.mesh)

            umesh_obj = r[tag_umesh]
            umesh = np.array(umesh_obj[:])
            self.u_shape = (
                jnp.array([len(umesh)], dtype=int)
                if flatten_ic
                else jnp.array(umesh_obj.attrs[tag_ushape][:], dtype=int)
            )
            self.tsteps = r[self.tags_field[0]].attrs["time_steps"]
            self.tsteps = jnp.array([t for t in self.tsteps if t <= tmax / t_norm])
            self.tsteps = self.tsteps * t_norm

            if self.normalize_data:
                self.mesh = self.normalizeSpatial(self.mesh)
                self.tsteps = self.normalizeTemporal(self.tsteps)

        self.tt = np.repeat(self.tsteps, self.mesh.shape[0])
        self.N = len(filenamesH5)

        self.datasets = []
        for i in range(0, min(MAXNUM_DATASETS, len(filenamesH5))):
            filename = filenamesH5[i]
            if Path(filename).exists():
                self.datasets.append(
                    h5py.File(filename, "r")
                )  # add file handles and keeps open
            else:
                print(f"Could not be found (ignoring): {filename}")

        # Calculate min and max pressure values from sampled datasets
        self._u_p_min, self._u_p_max = _calculate_u_pressure_minmax(
            self.datasets, self.tag_ufield
        )

    # --- required abstract properties implemented ---
    @property
    def Pmesh(self):
        """Total number of mesh points."""
        return self.mesh.shape[0]

    @property
    def P(self):
        """Total number of time/space points."""
        return self.Pmesh * len(self.tsteps)

    @property
    def xxyyzztt(self):
        xxyyzz = np.tile(self.mesh, (len(self.tsteps), 1))
        return np.hstack((xxyyzz, self.tt.reshape(-1, 1)))

    def normalizeSpatial(self, data):
        return _normalize_spatial(data, self.xmin, self.xmax)

    def normalizeTemporal(self, data):
        return _normalize_temporal(data, self.xmin, self.xmax)

    def denormalizeSpatial(self, data):
        return _denormalize_spatial(data, self.xmin, self.xmax)

    def denormalizeTemporal(self, data):
        return _denormalize_temporal(data, self.xmin, self.xmax)

    def u_pressures(self, idx: int) -> np.ndarray:
        """Get normalized u pressures for a given dataset index."""
        dataset = self.datasets[idx]
        u_norm = _normalize_spatial(
            dataset[self.tag_ufield][:], self._u_p_min, self._u_p_max
        )
        return jnp.reshape(u_norm, self.u_shape)

    def __del__(self):
        for dataset in self.datasets:
            dataset.close()


class DatasetH5Mock:
    """Mimicking a H5 dataset.

    Wrapping a dictionary inside a class with the same interface as the HDF5 object.
    """

    data: dict

    def __init__(self, data: dict):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __contains__(self, key):
        return key in self.data

    def close(self):
        # mocking HdF5 close method
        pass


class DataSourceOnly(DataInterface):
    """Used for inference only where arbitrary source positions can be used.

    The mesh is loaded from HDF5 to ensure the grid distribution is the same
    as for the trained model required for the branch net.
    """

    simulationDataType: SimulationDataType = SimulationDataType.SOURCE_ONLY

    datasets: list[h5py.File]
    mesh: np.ndarray[float]
    tags_field: list[str]
    tag_ufield: str
    tsteps: np.ndarray[float]
    tt: np.ndarray[float]
    N: int
    u_shape: np.ndarray[np.int64]
    conn: np.ndarray[np.int64]
    data_prune: int

    def __init__(
        self,
        data_path: str,
        source_pos: np.ndarray[float],
        params,
        tmax: float = float("inf"),
        t_norm: float = 1.0,
        flatten_ic: bool = True,
        data_prune: int = 1,
        norm_data: bool = False,
        p_minmax: tuple[float, float] = (-2.0, 2.0),
    ) -> None:
        self.data_prune = data_prune
        self._normalize_data = norm_data

        tag_mesh = "/mesh"
        tag_conn = "/conn"
        tag_umesh = "/umesh"
        tag_ushape = "umesh_shape"
        self.tags_field = ["/pressures"]
        self.tag_ufield = "/upressures"

        filenamesH5 = IO.pathsToFileType(data_path, ".h5", exclude="rectilinear")

        with h5py.File(filenamesH5[0]) as r:
            self.mesh = np.array(r[tag_mesh][:: self.data_prune])
            self.conn = (
                np.array(r[tag_conn])
                if self.data_prune == 1 and tag_conn in r
                else np.array([])
            )
            self._xmin, self._xmax = (
                float(np.min(self.mesh)),
                float(np.max(self.mesh)),
            )

            umesh_obj = r[tag_umesh]
            if flatten_ic:
                self.u_shape = jnp.array([len(umesh_obj[:])], dtype=int)
            else:
                self.u_shape = jnp.array(umesh_obj.attrs[tag_ushape][:], dtype=int)

            tsteps = r[self.tags_field[0]].attrs["time_steps"]
            tsteps = jnp.array([t for t in tsteps if t <= tmax / t_norm])
            self.tsteps = tsteps * t_norm

            if self._normalize_data:
                self.mesh = self.normalizeSpatial(self.mesh)
                self.tsteps = self.normalizeTemporal(self.tsteps)

            gaussianSrc = lambda x, y, z, xyz0, sigma, ampl: ampl * np.exp(
                -((x - xyz0[0]) ** 2 + (y - xyz0[1]) ** 2 + (z - xyz0[2]) ** 2)
                / sigma**2
            )

            self.datasets: list[h5py.File] = []
            sigma0 = params.c / (np.pi * params.fmax / 2)

            self.N = len(source_pos)

            for i in range(self.N):
                x0 = source_pos[i]
                ic_field = gaussianSrc(
                    umesh_obj[:, 0],
                    umesh_obj[:, 1],
                    umesh_obj[:, 2],
                    x0,
                    sigma0,
                    p_minmax[1],
                )
                self.datasets.append(
                    DatasetH5Mock({self.tag_ufield: ic_field, "source_position": x0})
                )

        # Calculate min and max pressure values from generated datasets
        self._u_p_min, self._u_p_max = _calculate_u_pressure_minmax(
            self.datasets, self.tag_ufield, max_samples=self.N
        )

        self.tt = np.repeat(self.tsteps, self.mesh.shape[0])

    @property
    def Pmesh(self) -> int:
        """Total number of mesh points."""
        return self.mesh.shape[0]

    @property
    def P(self) -> int:
        """Total number of time/space points."""
        return self.Pmesh * len(self.tsteps)

    @property
    def xmin(self) -> float:
        return self._xmin

    @property
    def xmax(self) -> float:
        return self._xmax

    @property
    def normalize_data(self) -> bool:
        return self._normalize_data

    @property
    def xxyyzztt(self) -> np.ndarray[float]:
        """Spatio-temporal coordinates stacked as [x, y, z, t]."""
        xxyyzz = np.tile(self.mesh, (len(self.tsteps), 1))
        return np.hstack((xxyyzz, self.tt.reshape(-1, 1)))

    def normalizeSpatial(self, data: np.ndarray[float]) -> np.ndarray[float]:
        return _normalize_spatial(data, self._xmin, self._xmax)

    def normalizeTemporal(self, data: np.ndarray[float]) -> np.ndarray[float]:
        return _normalize_temporal(data, self._xmin, self._xmax)

    def denormalizeSpatial(self, data: np.ndarray[float]) -> np.ndarray[float]:
        return _denormalize_spatial(data, self._xmin, self._xmax)

    def denormalizeTemporal(self, data: np.ndarray[float]) -> np.ndarray[float]:
        return _denormalize_temporal(data, self._xmin, self._xmax)

    def u_pressures(self, idx: int) -> np.ndarray:
        """Get normalized u pressures for a given dataset index."""
        dataset = self.datasets[idx]
        u_norm = _normalize_spatial(
            dataset[self.tag_ufield], self._u_p_min, self._u_p_max
        )
        return jnp.reshape(u_norm, self.u_shape)

    def close(self) -> None:
        for dataset in self.datasets:
            dataset.close()

    def __del__(self) -> None:
        self.close()


class DatasetStreamer(Dataset):
    Pmesh: int
    P: int
    batch_size_coord: int

    data: DataInterface
    p_minmax: tuple[float, float]

    itercount: itertools.count

    __y_feat_extract_fn = Callable[[list], list]

    total_time = 0

    @property
    def N(self):
        return self.data.N

    @property
    def Pmesh(self):
        """Total number of mesh points."""
        return self.data.mesh.shape[0]

    @property
    def P(self):
        """Total number of time/space points."""
        return self.Pmesh * self.data.tsteps.shape[0]

    def __init__(
        self, data, batch_size_coord=-1, y_feat_extract_fn=None, p_minmax=(-2.0, 2.0)
    ):
        # batch_size_coord: set to -1 if full dataset should be used (e.g. for validation data)
        self.data = data
        self.p_minmax = p_minmax

        self.batch_size_coord = (
            batch_size_coord if batch_size_coord <= self.P else self.P
        )
        self.__y_feat_extract_fn = (
            (lambda y: y) if y_feat_extract_fn is None else y_feat_extract_fn
        )

        self.itercount = itertools.count()
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.data.datasets)

    def __getitem__(self, idx):
        dataset = self.data.datasets[idx]
        u = self.data.u_pressures(idx)

        start_time_0 = time.perf_counter()
        if self.batch_size_coord > 0:
            indxs_coord = self.rng.choice(
                self.P, (self.batch_size_coord), replace=False
            )
        else:
            indxs_coord = jnp.arange(0, self.P)
        end_time_0 = time.perf_counter()
        self.total_time += end_time_0 - start_time_0

        xxyyzz = self.data.mesh[np.mod(indxs_coord, self.data.Pmesh), :]
        tt = self.data.tt[indxs_coord].reshape(-1, 1)
        y = self.__y_feat_extract_fn(np.hstack((xxyyzz, tt)))

        # collect all field data for all timesteps - might be memory consuming
        # If memory load gets too heavy, consider selecting points at each timestep
        num_tsteps = len(self.data.tsteps)
        if self.data.simulationDataType == SimulationDataType.H5COMPACT:
            s = dataset[self.data.tags_field[0]][
                0:num_tsteps, :: self.data.data_prune
            ].flatten()[indxs_coord]
        elif self.data.simulationDataType == SimulationDataType.XDMF:
            s = np.empty((self.P), dtype=jnp.float32)
            for j in range(num_tsteps):
                s[j * self.data.Pmesh : (j + 1) * self.Pmesh] = dataset[
                    self.data.tags_field[j]
                ][:: self.data.data_prune]
            s = s[indxs_coord]
        elif self.data.simulationDataType == SimulationDataType.SOURCE_ONLY:
            s = []
        else:
            raise Exception("Data format unknown: should be H5COMPACT or XDMF")

        # normalize
        x0 = (
            self.data.normalizeSpatial(dataset["source_position"][:])
            if "source_position" in dataset
            else []
        )

        inputs = jnp.asarray(u), jnp.asarray(y)
        return inputs, jnp.asarray(s), indxs_coord, x0


def getNumberOfSources(data_path: str):
    return len(IO.pathsToFileType(data_path, ".h5", exclude="rectilinear"))


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return jnp.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
