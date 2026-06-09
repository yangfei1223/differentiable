"""网格加载模块 — OBJ/GLB 解析与几何属性提取。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import trimesh

from src.gltf_loader import load_gltf


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
    normals: np.ndarray = None           # 顶点法线 [Nn, 3]
    normal_idx: np.ndarray = None        # 面-法线索引 [M, 3]
    tangents: np.ndarray = None          # 顶点切线 [Nn, 3]
    bitangents: np.ndarray = None        # 顶点副切线 [Nn, 3]

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

    def compute_vertex_tangents(self) -> Tuple[np.ndarray, np.ndarray]:
        """计算顶点切线和副切线 (Mikktspace 风格)。

        根据三角形边和 UV 差值计算面切线，再面积加权平均到顶点。
        正交化到法线后，副切线通过 cross(normal, tangent) 重建。

        Returns:
            (tangents, bitangents) — 各为 [N, 3] 单位向量。
        """
        if self.uvs.shape[0] == 0:
            raise ValueError("网格无 UV 坐标，无法计算切线")

        tangents = np.zeros((len(self.vertices), 3), dtype=np.float64)
        bitangents = np.zeros((len(self.vertices), 3), dtype=np.float64)

        for fi in range(len(self.faces)):
            vi0, vi1, vi2 = self.faces[fi]
            ui0, ui1, ui2 = self.uv_idx[fi]

            p0, p1, p2 = self.vertices[vi0], self.vertices[vi1], self.vertices[vi2]
            uv0, uv1, uv2 = self.uvs[ui0], self.uvs[ui1], self.uvs[ui2]

            e1 = p1 - p0
            e2 = p2 - p0
            du1 = uv1[0] - uv0[0]
            dv1 = uv1[1] - uv0[1]
            du2 = uv2[0] - uv0[0]
            dv2 = uv2[1] - uv0[1]

            det = du1 * dv2 - du2 * dv1
            if abs(det) < 1e-10:
                continue

            invdet = 1.0 / det
            face_tangent = (dv2 * e1 - dv1 * e2) * invdet
            face_bitangent = (-du2 * e1 + du1 * e2) * invdet

            # 面积加权累积
            area = np.linalg.norm(np.cross(e1, e2)) * 0.5
            for vi in (vi0, vi1, vi2):
                tangents[vi] += face_tangent * area
                bitangents[vi] += face_bitangent * area

        # 正交化到法线，重建 bitangent
        for i in range(len(self.vertices)):
            n = self.normals[i]
            t = tangents[i]

            # Gram-Schmidt: 去掉 tangent 中与法线平行的分量
            t = t - np.dot(t, n) * n
            tn = np.linalg.norm(t)
            if tn > 1e-10:
                t /= tn
            else:
                # 退化时用任意与 n 垂直的方向
                ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                t = np.cross(n, ref)
                t /= np.linalg.norm(t)

            tangents[i] = t
            # bitangent = cross(normal, tangent) 确保右手系
            bitangents[i] = np.cross(n, t)

        return tangents, bitangents

    def to_torch(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """将网格数据转换为 PyTorch 张量。

        Returns:
            (vertices, faces, uvs, uv_idx, normals, normal_idx, tangents, bitangents)
        """
        v = torch.from_numpy(self.vertices.astype(np.float32))
        f = torch.from_numpy(self.faces.astype(np.int64))
        uv = torch.from_numpy(self.uvs.astype(np.float32))
        uvi = torch.from_numpy(self.uv_idx.astype(np.int64))
        if self.normals is not None:
            n = torch.from_numpy(self.normals.astype(np.float32))
            ni = torch.from_numpy(self.normal_idx.astype(np.int64))
        else:
            n = torch.zeros_like(v)
            ni = torch.zeros_like(f)
        if self.tangents is not None:
            t = torch.from_numpy(self.tangents.astype(np.float32))
            bt = torch.from_numpy(self.bitangents.astype(np.float32))
        else:
            t = torch.zeros_like(v)
            bt = torch.zeros_like(v)
        return v, f, uv, uvi, n, ni, t, bt


@dataclass
class SubMeshData:
    """Single submesh extracted from a glTF file.

    Similar to MeshData but represents one primitive within a multi-mesh scene.
    """
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray
    uv_idx: np.ndarray
    normals: np.ndarray = None
    normal_idx: np.ndarray = None
    tangents: np.ndarray = None
    bitangents: np.ndarray = None
    material_name: str | None = None

    @property
    def num_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def num_faces(self) -> int:
        return self.faces.shape[0]

    @classmethod
    def from_dict(cls, d: dict) -> SubMeshData:
        """Construct from a gltf_loader dict."""
        sub = cls(
            name=d["name"],
            vertices=d["vertices"],
            faces=d["faces"],
            uvs=d["uvs"],
            uv_idx=d["uv_idx"],
            normals=d.get("normals"),
            normal_idx=d.get("normal_idx"),
            material_name=d.get("material_name"),
        )
        # Compute normals if missing
        if sub.normals is None:
            sub.normals = MeshData(
                vertices=sub.vertices, faces=sub.faces,
                uvs=sub.uvs, uv_idx=sub.uv_idx,
            ).compute_vertex_normals()
        if sub.normal_idx is None:
            sub.normal_idx = np.array(sub.faces, dtype=np.int64)
        # Compute tangents
        temp = MeshData(
            vertices=sub.vertices, faces=sub.faces,
            uvs=sub.uvs, uv_idx=sub.uv_idx,
            normals=sub.normals, normal_idx=sub.normal_idx,
        )
        sub.tangents, sub.bitangents = temp.compute_vertex_tangents()
        return sub

    def to_torch(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert to PyTorch tensors (same format as MeshData.to_torch)."""
        v = torch.from_numpy(self.vertices.astype(np.float32))
        f = torch.from_numpy(self.faces.astype(np.int64))
        uv = torch.from_numpy(self.uvs.astype(np.float32))
        uvi = torch.from_numpy(self.uv_idx.astype(np.int64))
        n = torch.from_numpy(self.normals.astype(np.float32)) if self.normals is not None else torch.zeros_like(v)
        ni = torch.from_numpy(self.normal_idx.astype(np.int64)) if self.normal_idx is not None else torch.zeros_like(f)
        t = torch.from_numpy(self.tangents.astype(np.float32)) if self.tangents is not None else torch.zeros_like(v)
        bt = torch.from_numpy(self.bitangents.astype(np.float32)) if self.bitangents is not None else torch.zeros_like(v)
        return v, f, uv, uvi, n, ni, t, bt


@dataclass
class MultiMeshData:
    """Collection of submeshes from a multi-mesh glTF file."""
    submeshes: list[SubMeshData]

    @property
    def num_submeshes(self) -> int:
        return len(self.submeshes)

    @property
    def total_vertices(self) -> int:
        return sum(s.num_vertices for s in self.submeshes)

    @property
    def total_faces(self) -> int:
        return sum(s.num_faces for s in self.submeshes)


def load_mesh(path: str | Path) -> MeshData | MultiMeshData:
    """加载 OBJ / GLB 网格文件。

    Args:
        path: 网格文件路径 (.obj / .glb / .gltf)。

    Returns:
        MeshData for single-mesh files, MultiMeshData for multi-mesh GLBs.
    """
    path = Path(path)

    if path.suffix.lower() in (".glb", ".gltf"):
        subs = load_gltf(path)
        if len(subs) == 1:
            # Single mesh — use existing MeshData path
            d = subs[0]
            mesh_data = MeshData(
                vertices=d["vertices"], faces=d["faces"],
                uvs=d["uvs"], uv_idx=d["uv_idx"],
                normals=d.get("normals"), normal_idx=d.get("normal_idx"),
            )
            if mesh_data.normals is None:
                mesh_data.normals = mesh_data.compute_vertex_normals()
            mesh_data.tangents, mesh_data.bitangents = mesh_data.compute_vertex_tangents()
            return mesh_data
        else:
            # Multi mesh
            submesh_list = [SubMeshData.from_dict(d) for d in subs]
            return MultiMeshData(submeshes=submesh_list)

    # OBJ fallback (existing trimesh path)
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

    # 提取顶点法线
    if hasattr(mesh_obj, 'vertex_normals'):
        normals = np.array(mesh_obj.vertex_normals, dtype=np.float64)
    else:
        normals = None

    normal_idx = np.array(mesh_obj.faces, dtype=np.int64)

    if normals is None:
        temp = MeshData(vertices=vertices, faces=faces, uvs=uvs, uv_idx=uv_idx,
                        normals=np.zeros_like(vertices), normal_idx=normal_idx)
        normals = temp.compute_vertex_normals()

    mesh_data = MeshData(vertices=vertices, faces=faces, uvs=uvs, uv_idx=uv_idx,
                         normals=normals, normal_idx=normal_idx)

    # 计算切线
    tangents, bitangents = mesh_data.compute_vertex_tangents()
    mesh_data.tangents = tangents
    mesh_data.bitangents = bitangents

    return mesh_data
