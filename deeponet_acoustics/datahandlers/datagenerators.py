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


class DataInterface(ABC):
    @property
    @abstractmethod
    def simulationDataType(self) -> SimulationDataType:
        """Type of the simulation data."""
        pass

    @property
    @abstractmethod
    def datasets(self) -> list[h5py.File]:
        """list of h5py files."""
        pass

    @property
    @abstractmethod
    def mesh(self) -> np.ndarray:
        """Mesh data (non-uniform)."""
        pass

    @property
    @abstractmethod
    def tags_field(self) -> list[str]:
        """Field tags for time/coordinate inputs."""
        pass

    @property
    @abstractmethod
    def tag_ufield(self) -> str:
        """Unique field tag for input function (uniformly distributed)."""
        pass

    @property
    @abstractmethod
    def tt(self) -> np.ndarray:
        """Time values"""
        pass

    @property
    @abstractmethod
    def data_prune(self) -> int:
        """Pruning parameter."""
        pass

    @property
    @abstractmethod
    def N(self) -> int:
        """Number of sources."""
        pass

    @property
    @abstractmethod
    def u_shape(self) -> list[int]:
        """Shape of the input function of the initial condition."""
        pass

    @property
    @abstractmethod
    def P(self) -> np.ndarray:
        """Total number of time/space points."""
        pass

    @property
    @abstractmethod
    def tsteps(self) -> np.ndarray:
        """Time steps."""
        pass


class DataXdmf(DataInterface):
    simulationDataType: SimulationDataType = SimulationDataType.XDMF

    P: int
    Pmesh: int

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
        self._tags_field = xdmf.tags_field
        self._tag_ufield = xdmf.tag_ufield
        self._data_prune = data_prune
        self._tsteps = xdmf.tsteps * t_norm

        with h5py.File(xdmf.filenameH5) as r:
            self._mesh = np.array(r[xdmf.tag_mesh][:: self._data_prune])
            self.xmin, self.xmax = np.min(self._mesh), np.max(self._mesh)
            umesh_obj = r[xdmf.tag_umesh]
            umesh = np.array(umesh_obj[:])
            self._u_shape = (
                [len(umesh)]
                if flatten_ic
                else jnp.array(umesh_obj.attrs[xdmf.tag_ushape][:], dtype=int).tolist()
            )

        if norm_data:
            self._mesh = self.normalizeSpatial(self._mesh)
            self._tsteps = self.normalizeTemporal(self._tsteps)

        self._tt = np.repeat(self._tsteps, self._mesh.shape[0])

        self._datasets = []
        for i in range(0, min(MAXNUM_DATASETS, len(filenames_xdmf))):
            filename = filenames_xdmf[i]
            if Path(filename).exists():
                xdmf = XdmfReader(filename, tmax / t_norm)
                self._datasets.append(
                    h5py.File(xdmf.filenameH5, "r")
                )  # add file handles and keeps open
            else:
                print(f"Could not be found (ignoring): {filename}")

        self._N = len(self._datasets)

    # --- required abstract properties implemented ---
    @property
    def datasets(self):
        return self._datasets

    @property
    def mesh(self):
        return self._mesh

    @property
    def u_shape(self):
        return self._u_shape

    @property
    def tsteps(self):
        return self._tsteps

    @property
    def tt(self):
        return self._tt

    @property
    def tags_field(self):
        return self._tags_field

    @property
    def tag_ufield(self):
        return self._tag_ufield

    @property
    def data_prune(self):
        return self._data_prune

    @property
    def N(self):
        return self._N

    @property
    def Pmesh(self):
        """Total number of mesh points."""
        return self.mesh.shape[0]

    @property
    def P(self):
        """Total number of time/space points."""
        return self.Pmesh * self.tsteps.shape[0]

    def normalizeSpatial(self, data):
        return 2 * (data - self.xmin) / (self.xmax - self.xmin) - 1

    def normalizeTemporal(self, data):
        return data / (self.xmax - self.xmin) / 2

    def __del__(self):
        for dataset in self._datasets:
            dataset.close()


class DataH5Compact(DataInterface):
    simulationDataType: SimulationDataType = SimulationDataType.H5COMPACT

    P: int
    Pmesh: int
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
        self._data_prune = data_prune
        self.normalize_data = norm_data

        # NOTE: we assume meshes, tags, etc are the same accross all xdmf datasets
        tag_mesh = "/mesh"
        tag_conn = "/conn"
        tag_umesh = "/umesh"
        tag_ushape = "umesh_shape"
        self._tags_field = ["/pressures"]
        self._tag_ufield = "/upressures"

        with h5py.File(filenamesH5[0]) as r:
            self._mesh = np.array(r[tag_mesh][:: self._data_prune])
            self.conn = (
                np.array(r[tag_conn])
                if self._data_prune == 1 and tag_conn in r
                else np.array([])
            )
            self.xmin, self.xmax = np.min(self._mesh), np.max(self._mesh)

            umesh_obj = r[tag_umesh]
            umesh = np.array(umesh_obj[:])
            self._u_shape = (
                jnp.array([len(umesh)], dtype=int)
                if flatten_ic
                else jnp.array(umesh_obj.attrs[tag_ushape][:], dtype=int)
            )
            self._tsteps = r[self._tags_field[0]].attrs["time_steps"]
            self._tsteps = jnp.array([t for t in self._tsteps if t <= tmax / t_norm])
            self._tsteps = self._tsteps * t_norm

            if self.normalize_data:
                self._mesh = self.normalizeSpatial(self._mesh)
                self._tsteps = self.normalizeTemporal(self._tsteps)

        self._tt = np.repeat(self._tsteps, self._mesh.shape[0])
        self._N = len(filenamesH5)

        self._datasets = []
        for i in range(0, min(MAXNUM_DATASETS, len(filenamesH5))):
            filename = filenamesH5[i]
            if Path(filename).exists():
                self._datasets.append(
                    h5py.File(filename, "r")
                )  # add file handles and keeps open
            else:
                print(f"Could not be found (ignoring): {filename}")

    # --- required abstract properties implemented ---
    @property
    def datasets(self):
        return self._datasets

    @property
    def mesh(self):
        return self._mesh

    @property
    def u_shape(self):
        return self._u_shape

    @property
    def tsteps(self):
        return self._tsteps

    @property
    def tt(self):
        return self._tt

    @property
    def tags_field(self):
        return self._tags_field

    @property
    def tag_ufield(self):
        return self._tag_ufield

    @property
    def data_prune(self):
        return self._data_prune

    @property
    def N(self):
        return self._N

    @property
    def Pmesh(self):
        """Total number of mesh points."""
        return self.mesh.shape[0]

    @property
    def P(self):
        """Total number of time/space points."""
        return self.Pmesh * self.tsteps.shape[0]

    @property
    def xxyyzztt(self):
        xxyyzz = np.tile(self.mesh, (len(self.tsteps), 1))
        return np.hstack((xxyyzz, self.tt.reshape(-1, 1)))

    def normalizeSpatial(self, data):
        return 2 * (data - self.xmin) / (self.xmax - self.xmin) - 1

    def normalizeTemporal(self, data):
        return data / (self.xmax - self.xmin) / 2

    def denormalizeSpatial(self, data):
        return (data + 1) / 2 * (self.xmax - self.xmin) + self.xmin

    def denormalizeTemporal(self, data):
        return data * 2 * (self.xmax - self.xmin)

    def __del__(self):
        for dataset in self._datasets:
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
        self._data_prune = data_prune
        self._normalize_data = norm_data

        tag_mesh = "/mesh"
        tag_conn = "/conn"
        tag_umesh = "/umesh"
        tag_ushape = "umesh_shape"
        self._tags_field = ["/pressures"]
        self._tag_ufield = "/upressures"

        filenamesH5 = IO.pathsToFileType(data_path, ".h5", exclude="rectilinear")

        with h5py.File(filenamesH5[0]) as r:
            self._mesh = np.array(r[tag_mesh][:: self._data_prune])
            self._conn = (
                np.array(r[tag_conn])
                if self._data_prune == 1 and tag_conn in r
                else np.array([])
            )
            self._xmin, self._xmax = (
                float(np.min(self._mesh)),
                float(np.max(self._mesh)),
            )

            umesh_obj = r[tag_umesh]
            if flatten_ic:
                self._u_shape = jnp.array([len(umesh_obj[:])], dtype=int)
            else:
                self._u_shape = jnp.array(umesh_obj.attrs[tag_ushape][:], dtype=int)

            tsteps = r[self._tags_field[0]].attrs["time_steps"]
            tsteps = jnp.array([t for t in tsteps if t <= tmax / t_norm])
            self._tsteps = tsteps * t_norm

            if self._normalize_data:
                self._mesh = self.normalizeSpatial(self._mesh)
                self._tsteps = self.normalizeTemporal(self._tsteps)

            gaussianSrc = lambda x, y, z, xyz0, sigma, ampl: ampl * np.exp(
                -((x - xyz0[0]) ** 2 + (y - xyz0[1]) ** 2 + (z - xyz0[2]) ** 2)
                / sigma**2
            )

            self._datasets: list[h5py.File] = []
            sigma0 = params.c / (np.pi * params.fmax / 2)

            self._N = len(source_pos)

            for i in range(self._N):
                x0 = source_pos[i]
                ic_field = gaussianSrc(
                    umesh_obj[:, 0],
                    umesh_obj[:, 1],
                    umesh_obj[:, 2],
                    x0,
                    sigma0,
                    p_minmax[1],
                )
                self._datasets.append(
                    DatasetH5Mock({self._tag_ufield: ic_field, "source_position": x0})
                )

        self._tt = np.repeat(self._tsteps, self._mesh.shape[0])

    @property
    def datasets(self) -> list[h5py.File]:
        return self._datasets

    @property
    def mesh(self) -> np.ndarray[float]:
        return self._mesh

    @property
    def tags_field(self) -> list[str]:
        return self._tags_field

    @property
    def tag_ufield(self) -> str:
        return self._tag_ufield

    @property
    def tsteps(self) -> np.ndarray[float]:
        return self._tsteps

    @property
    def tt(self) -> np.ndarray[float]:
        return self._tt

    @property
    def N(self) -> int:
        return self._N

    @property
    def Pmesh(self) -> int:
        """Total number of mesh points."""
        return self.mesh.shape[0]

    @property
    def P(self) -> int:
        """Total number of time/space points."""
        return self.Pmesh * self.tsteps.shape[0]

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
    def u_shape(self) -> np.ndarray[np.int64]:
        return self._u_shape

    @property
    def conn(self) -> np.ndarray[np.int64]:
        return self._conn

    @property
    def xxyyzztt(self) -> np.ndarray[float]:
        """Spatio-temporal coordinates stacked as [x, y, z, t]."""
        xxyyzz = np.tile(self.mesh, (len(self.tsteps), 1))
        return np.hstack((xxyyzz, self.tt.reshape(-1, 1)))

    def normalizeSpatial(self, data: np.ndarray[float]) -> np.ndarray[float]:
        return 2 * (data - self._xmin) / (self._xmax - self._xmin) - 1

    def normalizeTemporal(self, data: np.ndarray[float]) -> np.ndarray[float]:
        return data / (self._xmax - self._xmin) / 2

    def denormalizeSpatial(self, data: np.ndarray[float]) -> np.ndarray[float]:
        return (data + 1) / 2 * (self._xmax - self._xmin) + self._xmin

    def denormalizeTemporal(self, data: np.ndarray[float]) -> np.ndarray[float]:
        return data * 2 * (self._xmax - self._xmin)

    def close(self) -> None:
        for dataset in self._datasets:
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
        u_norm = (
            2
            * (dataset[self.data.tag_ufield][:] - self.p_minmax[0])
            / (self.p_minmax[1] - self.p_minmax[0])
            - 1
        )
        u = jnp.reshape(u_norm, self.data.u_shape)

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
