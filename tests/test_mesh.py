"""网格模块测试 — OBJ/GLB 加载与几何属性计算。"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.mesh import MeshData, load_mesh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_minimal_obj(tmp: Path) -> Path:
    """创建一个包含 4 顶点、2 三角面、UV 坐标的最小 OBJ 文件。"""
    obj_path = tmp / "mini.obj"
    obj_path.write_text(
        "# minimal test OBJ\n"
        "v 0.0 0.0 0.0\n"
        "v 1.0 0.0 0.0\n"
        "v 1.0 1.0 0.0\n"
        "v 0.0 1.0 0.0\n"
        "vt 0.0 0.0\n"
        "vt 1.0 0.0\n"
        "vt 1.0 1.0\n"
        "vt 0.0 1.0\n"
        "f 1/1 2/2 3/3\n"
        "f 1/1 3/3 4/4\n",
        encoding="utf-8",
    )
    return obj_path


def _make_quad_mesh() -> MeshData:
    """构造一个 XY 平面上的正方形 (两三角形)。"""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    uvs = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    uv_idx = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return MeshData(vertices=vertices, faces=faces, uvs=uvs, uv_idx=uv_idx)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadObj:
    """test_load_obj — 从 OBJ 文件加载并验证形状。"""

    def test_shapes(self, tmp_path: Path):
        obj_path = _write_minimal_obj(tmp_path)
        mesh = load_mesh(str(obj_path))

        # 4 vertices, 2 faces, 4 UVs, 2 UV index triangles
        assert mesh.vertices.shape == (4, 3)
        assert mesh.faces.shape == (2, 3)
        assert mesh.uvs.shape == (4, 2)
        assert mesh.uv_idx.shape == (2, 3)

    def test_vertex_values(self, tmp_path: Path):
        obj_path = _write_minimal_obj(tmp_path)
        mesh = load_mesh(str(obj_path))

        expected_v = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64
        )
        np.testing.assert_allclose(mesh.vertices, expected_v, atol=1e-6)

    def test_face_indices(self, tmp_path: Path):
        obj_path = _write_minimal_obj(tmp_path)
        mesh = load_mesh(str(obj_path))

        # OBJ is 1-indexed; our loader should convert to 0-indexed
        expected_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        np.testing.assert_array_equal(mesh.faces, expected_f)


class TestComputeVertexNormals:
    """test_compute_vertex_normals — 面法线加权 → 单位向量。"""

    def test_unit_normals(self):
        mesh = _make_quad_mesh()
        normals = mesh.compute_vertex_normals()

        # normals 应该是 (N, 3) 形状
        assert normals.shape == (4, 3)

        # 每个法线应该是单位向量
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_normals_direction(self):
        """XY 平面上的正方形，法线应该指向 +Z。"""
        mesh = _make_quad_mesh()
        normals = mesh.compute_vertex_normals()

        # 所有法线的 Z 分量应该接近 1
        np.testing.assert_allclose(normals[:, 2], 1.0, atol=1e-6)


class TestTorchConversion:
    """test_torch_conversion — to_torch() 的类型与值验证。"""

    def test_dtypes(self):
        mesh = _make_quad_mesh()
        v, f, uv, uvi, n, ni = mesh.to_torch()

        assert v.dtype == torch.float32
        assert f.dtype == torch.int64
        assert uv.dtype == torch.float32
        assert uvi.dtype == torch.int64

    def test_values_match(self):
        mesh = _make_quad_mesh()
        v, f, uv, uvi, n, ni = mesh.to_torch()

        np.testing.assert_allclose(v.numpy(), mesh.vertices.astype(np.float32), atol=1e-6)
        np.testing.assert_array_equal(f.numpy(), mesh.faces)
        np.testing.assert_allclose(uv.numpy(), mesh.uvs.astype(np.float32), atol=1e-6)
        np.testing.assert_array_equal(uvi.numpy(), mesh.uv_idx)

    def test_no_grad(self):
        mesh = _make_quad_mesh()
        v, f, uv, uvi, n, ni = mesh.to_torch()

        assert not v.requires_grad
        assert not f.requires_grad


class TestProperties:
    """num_vertices / num_faces 属性。"""

    def test_num_vertices(self):
        mesh = _make_quad_mesh()
        assert mesh.num_vertices == 4

    def test_num_faces(self):
        mesh = _make_quad_mesh()
        assert mesh.num_faces == 2


# ---------------------------------------------------------------------------
# Normals tests
# ---------------------------------------------------------------------------


def test_mesh_data_has_normals():
    """MeshData 应包含法线字段。"""
    import numpy as np
    from src.mesh import MeshData

    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    uvs = np.zeros((3, 2), dtype=np.float64)
    uv_idx = np.array([[0, 1, 2]], dtype=np.int64)
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float64)

    mesh = MeshData(vertices=verts, faces=faces, uvs=uvs, uv_idx=uv_idx, normals=normals)
    assert mesh.normals is not None
    assert mesh.normals.shape == (3, 3)


def test_mesh_data_compute_vertex_normals():
    """compute_vertex_normals 应返回单位法线。"""
    import numpy as np
    from src.mesh import MeshData

    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    uvs = np.zeros((3, 2), dtype=np.float64)
    uv_idx = np.array([[0, 1, 2]], dtype=np.int64)
    normals = np.zeros_like(verts)

    mesh = MeshData(vertices=verts, faces=faces, uvs=uvs, uv_idx=uv_idx, normals=normals)
    vn = mesh.compute_vertex_normals()
    assert vn.shape == (3, 3)
    for i in range(3):
        assert abs(vn[i, 2] - 1.0) < 0.01


def test_load_mesh_includes_normals(tmp_path):
    """load_mesh 应提取顶点法线。"""
    import numpy as np
    from src.mesh import load_mesh

    obj_content = """v 0 0 0
v 1 0 0
v 0 1 0
vn 0 0 1
vn 0 0 1
vn 0 0 1
vt 0 0
vt 1 0
vt 0 1
f 1/1/1 2/2/2 3/3/3
"""
    obj_path = tmp_path / "test.obj"
    obj_path.write_text(obj_content)

    mesh = load_mesh(str(obj_path))
    assert mesh.normals is not None
    assert mesh.normals.shape[0] == 3
    assert mesh.normals.shape[1] == 3
