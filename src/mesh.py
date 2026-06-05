"""网格加载模块 — OBJ/GLB 解析与几何属性提取。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import trimesh


@dataclass
class MeshData:
    """三角网格数据容器。

    Attributes:
        vertices: 顶点坐标 [N, 3]
        faces:    三角面索引 [M, 3] (0-based)
        uvs:      纹理坐标 [K, 2]
        uv_idx:   面-UV 索引 [M, 3]
    """

    vertices: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray
    uv_idx: np.ndarray

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_vertices(self) -> int:
        """顶点数量。"""
        return self.vertices.shape[0]

    @property
    def num_faces(self) -> int:
        """三角面数量。"""
        return self.faces.shape[0]

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def compute_vertex_normals(self) -> np.ndarray:
        """计算顶点法线 (面积加权平均)。

        Returns:
            法线数组 [N, 3]，每个法线为单位向量。
        """
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]

        # 面法线 = 叉积 (正比于面积)
        face_normals = np.cross(v1 - v0, v2 - v0)  # [M, 3]

        # 用面积加权累积到顶点
        vertex_normals = np.zeros_like(self.vertices)
        np.add.at(vertex_normals, self.faces[:, 0], face_normals)
        np.add.at(vertex_normals, self.faces[:, 1], face_normals)
        np.add.at(vertex_normals, self.faces[:, 2], face_normals)

        # 归一化
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        vertex_normals = vertex_normals / norms
        return vertex_normals

    def to_torch(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """将网格数据转换为 PyTorch 张量。

        Returns:
            (vertices, faces, uvs, uv_idx) — 分别为 float32 / int64 张量。
        """
        v = torch.from_numpy(self.vertices.astype(np.float32))
        f = torch.from_numpy(self.faces.astype(np.int64))
        uv = torch.from_numpy(self.uvs.astype(np.float32))
        uvi = torch.from_numpy(self.uv_idx.astype(np.int64))
        return v, f, uv, uvi


def load_mesh(path: str | Path) -> MeshData:
    """加载 OBJ / GLB 网格文件。

    Args:
        path: 网格文件路径 (.obj / .glb / .gltf)。

    Returns:
        MeshData 实例。
    """
    path = Path(path)
    scene_or_mesh = trimesh.load(str(path), force="mesh", process=False)

    # trimesh.load 在某些情况下返回 Scene，统一取第一个 geometry
    if isinstance(scene_or_mesh, trimesh.Scene):
        geoms = list(scene_or_mesh.geometry.values())
        if not geoms:
            raise ValueError(f"文件 {path} 中未找到网格几何体")
        mesh_obj = geoms[0]
    else:
        mesh_obj = scene_or_mesh

    vertices = np.array(mesh_obj.vertices, dtype=np.float64)
    faces = np.array(mesh_obj.faces, dtype=np.int64)

    # 提取 UV 坐标
    if hasattr(mesh_obj.visual, "uv") and mesh_obj.visual.uv is not None:
        uvs = np.array(mesh_obj.visual.uv, dtype=np.float64)
        if uvs.ndim == 2 and uvs.shape[1] == 2:
            uv_idx = np.array(mesh_obj.faces, dtype=np.int64)
            # 修正 UV 到 [0, 1] 范围：某些模型（如 glTF）V 轴在 [-1, 0]
            # nvdiffrast 要求 UV 在 [0, 1]
            uvs[:, 1] = uvs[:, 1] % 1.0  # V: [-1, 0] → [0, 1]
        else:
            uvs = np.zeros((0, 2), dtype=np.float64)
            uv_idx = np.zeros_like(faces, dtype=np.int64)
    else:
        uvs = np.zeros((0, 2), dtype=np.float64)
        uv_idx = np.zeros_like(faces, dtype=np.int64)

    return MeshData(vertices=vertices, faces=faces, uvs=uvs, uv_idx=uv_idx)
