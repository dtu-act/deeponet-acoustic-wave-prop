# ==============================================================================
# Copyright 2023 Technical University of Denmark
# Author: Nikolas Borrel-Jensen
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import os
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import meshio
import numpy as np


class XdmfReader:
    root: ET.Element
    filenameH5: str  # assume filename is the same for all datasets
    tsteps: list[float]  # physical time
    tag_mesh: str
    tags_field: list[str]
    tag_umesh: str
    tag_ufield: str
    tag_ushape: str
    num_nodes: int

    # https://docs.python.org/3/library/xml.etree.elementtree.html
    def __init__(self, filename, tmax=float("inf")):
        base_path = os.path.dirname(filename)

        tree = ET.parse(filename)
        root = tree.getroot()

        self.tsteps = list(
            map(lambda x: float(x.attrib["Value"]), root.findall(".//Time"))
        )
        self.tsteps = np.array([t for t in self.tsteps if t < tmax])

        mesh_dataitem = root.find(".//*[@Name='mesh']/Geometry/DataItem")
        dataitem_str = re.sub(r"[\n\t\s]*", "", mesh_dataitem.text)
        self.filenameH5 = os.path.join(base_path, dataitem_str.split(":/")[0])
        self.tag_mesh = dataitem_str.split(":/")[1]
        self.tag_umesh = "/udata0"  # TODO: read from separate Xdmf header

        self.num_nodes = int(root.find(".//*[@Name='p']/DataItem").attrib["Dimensions"])

        # HACK: assume datasets are named data{i}
        self.tags_field = []
        for i, _ in enumerate(self.tsteps):
            self.tags_field.append(f"/data{i + 2}")

        self.tag_ufield = "/udata2"
        self.tag_ushape = "/umesh_shape"


def pathsToFileType(path, filetype, exclude=""):
    paths_to_file = []
    parent_dir_list = [
        i for i in os.listdir(path) if os.path.isdir(os.path.join(path, i))
    ]
    if len(parent_dir_list) == 0:
        parent_dir_list = [path]  # no subfolders
    for parent_dir_name in parent_dir_list:
        data_dir = os.path.join(path, parent_dir_name)
        for filename in os.listdir(data_dir):
            if filename.endswith(filetype) and (
                exclude == "" or filename.find(exclude) == -1
            ):
                paths_to_file.append(os.path.join(data_dir, filename))

    paths_to_file.sort()

    return paths_to_file


def writeTriangleXdmf(grid, conn, t_range, p, filename):
    cells = [("triangle", conn[:, 0:3])]

    writeXdmf(grid, t_range, p, filename, cells)


def writeTetraXdmf(grid, conn, t_range, p, filename):
    tetra_facets = np.zeros([len(conn) * 4, 4])
    for i, data in enumerate(conn):
        indx = i * 4
        tetra_facets[indx, :] = np.append([3], np.sort([data[0], data[1], data[2]]))
        tetra_facets[indx + 1, :] = np.append([3], np.sort([data[0], data[1], data[3]]))
        tetra_facets[indx + 2, :] = np.append([3], np.sort([data[0], data[2], data[3]]))
        tetra_facets[indx + 3, :] = np.append([3], np.sort([data[1], data[2], data[3]]))
    cells = [("tetra", tetra_facets)]

    writeXdmf(grid, t_range, p, filename, cells)


def writeXdmf(grid, t_range, p, filename, cells=[]):
    if len(cells) == 0:
        vertex = np.array(
            [
                [
                    i,
                ]
                for i in range(len(grid))
            ]
        )
        cells = [("vertex", vertex)]

    with meshio.xdmf.TimeSeriesWriter(filename) as writer:
        writer.write_points_cells(grid, cells)
        for i, t in enumerate(t_range):
            writer.write_data(t, point_data={"p": p[i, :]})

    moveHD5(filename)


def readXdmf(filename):
    with meshio.xdmf.TimeSeriesReader(filename) as reader:
        points, cells = reader.read_points_cells()
        t_range = np.zeros(reader.num_steps)
        p = np.zeros((reader.num_steps, points.shape[0]))

        for k in range(reader.num_steps):
            t, point_data, _ = reader.read_data(k)
            t_range[k] = t
            p[k, :] = point_data["p"]

    return points, t_range, p


def moveHD5(filename):
    # handle bug in meshio: move file from local dir to correct folder
    h5_filename = Path(filename).stem + ".h5"
    h5_move_to = os.path.join(os.path.dirname(filename), h5_filename)
    shutil.move(h5_filename, h5_move_to)
