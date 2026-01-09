# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


def write_ply_vertices(
    vertices: np.ndarray, output_path: str, binary: bool = True
) -> None:
    """
    Write vertex coordinates to a PLY file.

    Args:
        vertices: Array of 3D vertex positions (N, 3) as float32
        output_path: Path to output PLY file
        binary: If True, write binary little-endian format (default).
                If False, write ASCII format.
    """
    n_vertices = len(vertices)

    if n_vertices > 0:
        vertices = np.asarray(vertices, dtype=np.float32)
        if vertices.ndim == 1:
            vertices = vertices.reshape(-1, 3)
        assert (
            vertices.shape[1] == 3
        ), f"Expected (N, 3) array, got shape {vertices.shape}"

    format_str = "binary_little_endian 1.0" if binary else "ascii 1.0"
    header = f"""ply
format {format_str}
element vertex {n_vertices}
property float x
property float y
property float z
end_header
"""

    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))

        if n_vertices > 0:
            if binary:
                vertices.astype("<f4").tofile(f)
            else:
                for x, y, z in vertices:
                    f.write(f"{x} {y} {z}\n".encode("ascii"))


def read_ply_vertices(input_path: str) -> np.ndarray:
    """
    Read vertex coordinates from a PLY file.

    Supports ASCII and binary (little-endian and big-endian) PLY formats.
    Only reads x, y, z float properties from the vertex element.

    Args:
        input_path: Path to input PLY file

    Returns:
        Array of 3D vertex positions (N, 3) as float32
    """
    with open(input_path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        format_type = None
        n_vertices = 0
        properties = []
        in_vertex_element = False

        for line in header_lines:
            parts = line.split()
            if not parts:
                continue

            if parts[0] == "format":
                format_type = parts[1]
            elif parts[0] == "element" and parts[1] == "vertex":
                n_vertices = int(parts[2])
                in_vertex_element = True
            elif parts[0] == "element" and parts[1] != "vertex":
                in_vertex_element = False
            elif parts[0] == "property" and in_vertex_element:
                prop_type = parts[1]
                prop_name = parts[2]
                properties.append((prop_name, prop_type))

        if n_vertices == 0:
            return np.empty((0, 3), dtype=np.float32)

        prop_info = {}
        for i, (name, ptype) in enumerate(properties):
            if name in ("x", "y", "z"):
                prop_info[name] = (i, ptype)

        if not all(name in prop_info for name in ("x", "y", "z")):
            raise ValueError("PLY file must have x, y, z properties in vertex element")

        type_map = {
            "float": ("<f4", 4),
            "float32": ("<f4", 4),
            "double": ("<f8", 8),
            "float64": ("<f8", 8),
            "int": ("<i4", 4),
            "int32": ("<i4", 4),
            "uint": ("<u4", 4),
            "uint32": ("<u4", 4),
            "short": ("<i2", 2),
            "int16": ("<i2", 2),
            "ushort": ("<u2", 2),
            "uint16": ("<u2", 2),
            "char": ("<i1", 1),
            "int8": ("<i1", 1),
            "uchar": ("<u1", 1),
            "uint8": ("<u1", 1),
        }

        x_idx, _ = prop_info["x"]
        y_idx, _ = prop_info["y"]
        z_idx, _ = prop_info["z"]

        if format_type == "ascii":
            vertices = np.empty((n_vertices, 3), dtype=np.float32)

            for i in range(n_vertices):
                line = f.readline().decode("ascii").strip()
                values = line.split()
                vertices[i, 0] = float(values[x_idx])
                vertices[i, 1] = float(values[y_idx])
                vertices[i, 2] = float(values[z_idx])

        elif format_type in ("binary_little_endian", "binary_big_endian"):
            endian = "<" if format_type == "binary_little_endian" else ">"

            dtype_list = []
            for name, ptype in properties:
                if ptype not in type_map:
                    raise ValueError(f"Unsupported property type: {ptype}")
                np_dtype, _ = type_map[ptype]
                if endian == ">":
                    np_dtype = ">" + np_dtype[1:]
                dtype_list.append((name, np_dtype))

            vertex_dtype = np.dtype(dtype_list)
            data = np.frombuffer(
                f.read(n_vertices * vertex_dtype.itemsize), dtype=vertex_dtype
            )

            vertices = np.column_stack(
                [
                    data["x"].astype(np.float32),
                    data["y"].astype(np.float32),
                    data["z"].astype(np.float32),
                ]
            )
        else:
            raise ValueError(f"Unsupported PLY format: {format_type}")

        return vertices
