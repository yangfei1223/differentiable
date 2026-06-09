# Multi-Submesh PBR Baking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the PBR baking pipeline to support per-submesh independent textures with multi-mesh rendering, using the original high-poly geometry aligned with GT.

**Architecture:** New `gltf_loader.py` parses multi-mesh GLB via pygltflib. `mesh.py` gains `SubMeshData` + `MultiMeshData`. `PBRShadingModel` supports dict of textures keyed by submesh name. Trainer detects multi-mesh and runs per-submesh render + composite loop. Single-mesh path fully unchanged.

**Tech Stack:** pygltflib, PyTorch, nvdiffrast, numpy

**Branch:** `feature/multi-submesh`

---

## Task 1: glTF Loader — `src/gltf_loader.py`

**Files:**
- Create: `src/gltf_loader.py`
- Test: `tests/test_gltf_loader.py`

- [ ] **Step 1: Write failing tests for gltf_loader**

```python
# tests/test_gltf_loader.py
"""Tests for pygltflib-based glTF loader."""
import numpy as np
import pytest
from pathlib import Path

ASSETS = Path(__file__).parent.parent / "data"

def _piano_path():
    return str(ASSETS / "piano_260604" / "scene" / "lowpoly.glb")

def _helmet_path():
    return str(ASSETS / "helmet_260604" / "scene" / "lowpoly.glb")

def _piano_original_path():
    return str(ASSETS / "piano_260604" / "scene" / "original_with_mats.glb")

class TestLoadGLTF:
    def test_load_single_mesh_glb(self):
        """Single-mesh GLB should return one SubMeshData."""
        from src.gltf_loader import load_gltf
        result = load_gltf(_helmet_path())
        assert isinstance(result, list)
        assert len(result) == 1
        sub = result[0]
        assert sub["name"] is not None
        assert sub["vertices"].shape[1] == 3
        assert sub["faces"].shape[1] == 3
        assert sub["uvs"].shape[1] == 2
        assert sub["uv_idx"].shape[1] == 3

    def test_load_multi_mesh_glb(self):
        """Multi-mesh GLB should return multiple SubMeshData dicts."""
        from src.gltf_loader import load_gltf
        result = load_gltf(_piano_path())
        assert isinstance(result, list)
        assert len(result) == 6  # piano has 6 submeshes
        for sub in result:
            assert "name" in sub
            assert "vertices" in sub
            assert "faces" in sub
            assert "uvs" in sub
            assert "uv_idx" in sub
            assert "normals" in sub
            assert "normal_idx" in sub

    def test_uv_range(self):
        """UVs should be in [0, 1] after V-axis fix."""
        from src.gltf_loader import load_gltf
        result = load_gltf(_piano_path())
        for sub in result:
            assert sub["uvs"][:, 0].min() >= -0.01
            assert sub["uvs"][:, 0].max() <= 1.01
            assert sub["uvs"][:, 1].min() >= -0.01
            assert sub["uvs"][:, 1].max() <= 1.01

    def test_faces_are_valid_indices(self):
        """Face indices should be within vertex count."""
        from src.gltf_loader import load_gltf
        result = load_gltf(_piano_path())
        for sub in result:
            n_verts = sub["vertices"].shape[0]
            assert sub["faces"].min() >= 0
            assert sub["faces"].max() < n_verts

    def test_uv_idx_are_valid_indices(self):
        """UV indices should be within UV count."""
        from src.gltf_loader import load_gltf
        result = load_gltf(_piano_path())
        for sub in result:
            n_uvs = sub["uvs"].shape[0]
            assert sub["uv_idx"].min() >= 0
            assert sub["uv_idx"].max() < n_uvs

    def test_original_piano_loads(self):
        """Original high-poly piano should load with more vertices than lowpoly."""
        from src.gltf_loader import load_gltf
        result = load_gltf(_piano_original_path())
        assert len(result) == 6
        total_verts = sum(s["vertices"].shape[0] for s in result)
        assert total_verts > 90000  # original has ~93K vertices
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gltf_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.gltf_loader'`

- [ ] **Step 3: Implement `src/gltf_loader.py`**

```python
# src/gltf_loader.py
"""glTF loader — pygltflib-based, extracts per-mesh geometry + material refs."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from pygltflib import GLTF2


def load_gltf(path: str | Path) -> list[dict[str, Any]]:
    """Load a glTF/GLB file, returning a list of per-submesh dicts.

    Each dict has keys:
        name, vertices, faces, uvs, uv_idx, normals, normal_idx, material_name

    Args:
        path: Path to .glb/.gltf file.

    Returns:
        List of submesh dicts. Single-mesh files return a list of length 1.
    """
    path = Path(path)
    gltf = GLTF2().from_path(str(path))

    # Load binary data
    if gltf.binary is not None:
        bin_data = gltf.binary
    else:
        bin_data = b""

    # Build buffer view cache
    def _get_buffer_data(buffer_view_idx):
        bv = gltf.bufferViews[buffer_view_idx]
        start = bv.byteOffset or 0
        end = start + bv.byteLength
        return bin_data[start:end]

    # Parse all accessors into numpy arrays
    accessor_cache = {}
    for i, acc in enumerate(gltf.accessors or []):
        dtype_map = {
            5120: np.int8, 5121: np.uint8,
            5122: np.int16, 5123: np.uint16,
            5125: np.uint32, 5126: np.float32,
        }
        dtype = dtype_map[acc.componentType]
        raw = _get_buffer_data(acc.bufferView)

        shape_map = {
            "SCALAR": (1,), "VEC2": (2,), "VEC3": (3,),
            "VEC4": (4,), "MAT4": (4, 4),
        }
        shape = shape_map[acc.type]
        count = acc.count
        arr = np.frombuffer(raw, dtype=dtype).reshape(count, *shape).copy()

        # Apply sparse accessor if present
        if acc.sparse is not None:
            sp = acc.sparse
            idx_dtype = dtype_map[gltf.accessors[sp.indices.bufferView].componentType] \
                if hasattr(sp.indices, 'bufferView') else np.uint32
            # Simplified sparse handling — skip for now, most models don't use it
            pass

        accessor_cache[i] = arr

    # Traverse scene graph to find all mesh nodes with transforms
    submeshes = []
    visited_nodes = set()

    def _traverse_node(node_idx, parent_transform):
        if node_idx in visited_nodes:
            return
        visited_nodes.add(node_idx)
        node = gltf.nodes[node_idx]

        # Compute local transform
        local = np.eye(4, dtype=np.float64)
        if node.matrix is not None and len(node.matrix) == 16:
            local = np.array(node.matrix, dtype=np.float64).reshape(4, 4)
        else:
            if node.translation is not None:
                t = np.array(node.translation, dtype=np.float64)
                local[:3, 3] = t
            if node.rotation is not None:
                q = np.array(node.rotation, dtype=np.float64)  # wxyz
                qx, qy, qz, qw = q[1], q[2], q[3], q[0]
                r = np.eye(4, dtype=np.float64)
                r[0, 0] = 1 - 2*(qy*qy + qz*qz)
                r[0, 1] = 2*(qx*qy - qw*qz)
                r[0, 2] = 2*(qx*qz + qw*qy)
                r[1, 0] = 2*(qx*qy + qw*qz)
                r[1, 1] = 1 - 2*(qx*qx + qz*qz)
                r[1, 2] = 2*(qy*qz - qw*qx)
                r[2, 0] = 2*(qx*qz - qw*qy)
                r[2, 1] = 2*(qy*qz + qw*qx)
                r[2, 2] = 1 - 2*(qx*qx + qy*qy)
                local = r @ local
            if node.scale is not None:
                s = np.array(node.scale, dtype=np.float64)
                local[:3, :3] *= s

        transform = parent_transform @ local

        # If node has a mesh, extract it
        if node.mesh is not None:
            mesh = gltf.meshes[node.mesh]
            mesh_name = mesh.name or f"mesh_{node.mesh}"

            for pi, prim in enumerate(mesh.primitives):
                # Get positions
                pos = accessor_cache[prim.attributes.POSITION]
                verts = pos.astype(np.float64)

                # Get indices
                if prim.indices is not None:
                    faces = accessor_cache[prim.indices].astype(np.int64)
                else:
                    faces = np.arange(verts.shape[0], dtype=np.int64).reshape(-1, 3)

                # Get normals
                normals = None
                if hasattr(prim.attributes, 'NORMAL') and prim.attributes.NORMAL is not None:
                    normals = accessor_cache[prim.attributes.NORMAL].astype(np.float64)

                # Get UVs (TEXCOORD_0)
                uvs = np.zeros((0, 2), dtype=np.float64)
                uv_idx = np.zeros_like(faces, dtype=np.int64)
                if hasattr(prim.attributes, 'TEXCOORD_0') and prim.attributes.TEXCOORD_0 is not None:
                    uvs = accessor_cache[prim.attributes.TEXCOORD_0].astype(np.float64)[:, :2]
                    uv_idx = np.array(faces, dtype=np.int64)

                # Material name
                mat_name = None
                if prim.material is not None:
                    mat = gltf.materials[prim.material]
                    mat_name = mat.name or f"material_{prim.material}"

                # Apply node transform to vertices
                if not np.allclose(transform, np.eye(4)):
                    ones = np.ones((verts.shape[0], 1), dtype=np.float64)
                    verts_h = np.concatenate([verts, ones], axis=1)  # [V, 4]
                    verts_t = (transform[:3, :] @ verts_h.T).T  # [V, 3]
                    verts = verts_t

                    # Transform normals (rotation only, no translation/scale)
                    normal_transform = np.linalg.inv(transform[:3, :3]).T
                    if normals is not None:
                        normals = (normal_transform @ normals.T).T
                        norms = np.linalg.norm(normals, axis=1, keepdims=True)
                        norms = np.maximum(norms, 1e-10)
                        normals = normals / norms

                sub_name = f"{mesh_name}_prim{pi}" if len(mesh.primitives) > 1 else mesh_name

                submeshes.append({
                    "name": sub_name,
                    "vertices": verts,
                    "faces": faces,
                    "uvs": uvs,
                    "uv_idx": uv_idx,
                    "normals": normals,
                    "normal_idx": np.array(faces, dtype=np.int64),
                    "material_name": mat_name,
                })

        # Recurse children
        if node.children is not None:
            for child_idx in node.children:
                _traverse_node(child_idx, transform)

    # Start traversal from scene roots
    root_transform = np.eye(4, dtype=np.float64)
    scene = gltf.scenes[gltf.scene or 0]
    if scene.nodes is not None:
        for node_idx in scene.nodes:
            _traverse_node(node_idx, root_transform)

    # If no submeshes found via scene graph, fall back to flat mesh list
    if len(submeshes) == 0:
        for mi, mesh in enumerate(gltf.meshes or []):
            for pi, prim in enumerate(mesh.primitives):
                pos = accessor_cache[prim.attributes.POSITION]
                verts = pos.astype(np.float64)

                if prim.indices is not None:
                    faces = accessor_cache[prim.indices].astype(np.int64)
                else:
                    faces = np.arange(verts.shape[0], dtype=np.int64).reshape(-1, 3)

                normals = None
                if hasattr(prim.attributes, 'NORMAL') and prim.attributes.NORMAL is not None:
                    normals = accessor_cache[prim.attributes.NORMAL].astype(np.float64)

                uvs = np.zeros((0, 2), dtype=np.float64)
                uv_idx = np.zeros_like(faces, dtype=np.int64)
                if hasattr(prim.attributes, 'TEXCOORD_0') and prim.attributes.TEXCOORD_0 is not None:
                    uvs = accessor_cache[prim.attributes.TEXCOORD_0].astype(np.float64)[:, :2]
                    uv_idx = np.array(faces, dtype=np.int64)

                mat_name = None
                if prim.material is not None:
                    mat = gltf.materials[prim.material]
                    mat_name = mat.name or f"material_{prim.material}"

                sub_name = mesh.name or f"mesh_{mi}"

                submeshes.append({
                    "name": sub_name,
                    "vertices": verts,
                    "faces": faces,
                    "uvs": uvs,
                    "uv_idx": uv_idx,
                    "normals": normals,
                    "normal_idx": np.array(faces, dtype=np.int64),
                    "material_name": mat_name,
                })

    # UV V-axis fix: glTF V=0 at bottom → V=1 at top (nvdiffrast convention)
    for sub in submeshes:
        if sub["uvs"].shape[0] > 0:
            sub["uvs"][:, 1] = sub["uvs"][:, 1] % 1.0

    return submeshes
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gltf_loader.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/gltf_loader.py tests/test_gltf_loader.py
git commit -m "feat: add pygltflib-based glTF loader with multi-mesh support"
```

---

## Task 2: Data Classes — `SubMeshData`, `MultiMeshData` in `src/mesh.py`

**Files:**
- Modify: `src/mesh.py`
- Test: `tests/test_mesh.py`

- [ ] **Step 1: Write failing tests for new data classes**

```python
# Append to tests/test_mesh.py

class TestSubMeshData:
    def test_from_gltf_dict(self):
        """SubMeshData should be constructable from gltf_loader dict."""
        from src.mesh import SubMeshData
        d = {
            "name": "test_mesh",
            "vertices": np.random.randn(100, 3).astype(np.float64),
            "faces": np.array([[0, 1, 2]] * 10, dtype=np.int64),
            "uvs": np.random.rand(100, 2).astype(np.float64),
            "uv_idx": np.array([[0, 1, 2]] * 10, dtype=np.int64),
            "normals": np.random.randn(100, 3).astype(np.float64),
            "normal_idx": np.array([[0, 1, 2]] * 10, dtype=np.int64),
            "material_name": "test_mat",
        }
        sub = SubMeshData.from_dict(d)
        assert sub.name == "test_mesh"
        assert sub.vertices.shape == (100, 3)
        assert sub.num_faces == 10
        assert sub.material_name == "test_mat"
        # Should have tangents computed
        assert sub.tangents is not None
        assert sub.tangents.shape == (100, 3)

class TestMultiMeshData:
    def test_from_gltf_piano(self):
        """Loading piano lowpoly should return MultiMeshData with 6 submeshes."""
        from src.mesh import load_mesh, MultiMeshData
        mesh = load_mesh("data/piano_260604/scene/lowpoly.glb")
        assert isinstance(mesh, MultiMeshData)
        assert mesh.num_submeshes == 6
        for sub in mesh.submeshes:
            assert sub.vertices.shape[1] == 3
            assert sub.tangents is not None

    def test_single_mesh_still_works(self):
        """Loading single-mesh helmet should still return MeshData."""
        from src.mesh import load_mesh, MeshData
        mesh = load_mesh("data/helmet_260604/scene/lowpoly.glb")
        assert isinstance(mesh, MeshData)
        assert mesh.vertices.shape[1] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mesh.py::TestSubMeshData -v`
Expected: FAIL with `ImportError` or `AttributeError`

- [ ] **Step 3: Add `SubMeshData`, `MultiMeshData` to `src/mesh.py`**

Add to `src/mesh.py` after the existing `MeshData` class:

```python
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

    def to_torch(self):
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
```

- [ ] **Step 4: Modify `load_mesh()` to return `MultiMeshData` for multi-mesh GLBs**

Add at the top of `mesh.py`:
```python
from src.gltf_loader import load_gltf
```

Modify the existing `load_mesh()` function to detect multi-mesh:

```python
def load_mesh(path: str | Path) -> MeshData | MultiMeshData:
    """Load OBJ / GLB mesh file.

    Returns MeshData for single-mesh files, MultiMeshData for multi-mesh GLBs.
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
    # ... existing OBJ loading code unchanged ...
```

- [ ] **Step 5: Run all mesh tests**

Run: `pytest tests/test_mesh.py -v`
Expected: All tests PASS

- [ ] **Step 6: Run full test suite to verify backward compat**

Run: `pytest tests/ -v`
Expected: 100+ tests PASS, no regressions

- [ ] **Step 7: Commit**

```bash
git add src/mesh.py tests/test_mesh.py
git commit -m "feat: add SubMeshData, MultiMeshData, extend load_mesh for multi-mesh GLBs"
```

---

## Task 3: PBR Model — Multi-Texture Support

**Files:**
- Modify: `src/shading/pbr_model.py`
- Test: `tests/test_pbr_model_multi.py`

- [ ] **Step 1: Write failing tests for multi-texture PBR model**

```python
# tests/test_pbr_model_multi.py
"""Tests for PBRShadingModel multi-texture support."""
import pytest
import torch
from src.config import Config, PBRConfig
from src.shading.pbr_model import PBRShadingModel

class TestPBRModelMulti:
    def test_init_multi_textures(self):
        """init_textures with submesh_names should create dict of textures."""
        cfg = Config(render_mode="pbr", pbr=PBRConfig())
        model = PBRShadingModel(cfg)
        names = ["body", "strings", "keys"]
        model.init_textures(64, submesh_names=names)
        assert model.is_multi
        assert len(model.mat_textures) == 3
        for name in names:
            assert name in model.mat_textures
            assert model.mat_textures[name].shape == (1, 64, 64, 8)

    def test_parameters_multi(self):
        """parameters() should include all submesh textures + env_map."""
        cfg = Config(render_mode="pbr", pbr=PBRConfig())
        model = PBRShadingModel(cfg)
        model.init_textures(64, submesh_names=["a", "b"])
        params = model.parameters()
        assert len(params) == 3  # 2 textures + 1 env_map

    def test_state_dict_multi(self):
        """state_dict/load_state_dict round-trip for multi-texture."""
        cfg = Config(render_mode="pbr", pbr=PBRConfig())
        model = PBRShadingModel(cfg)
        model.init_textures(64, submesh_names=["a", "b"])
        state = model.state_dict()
        assert "mat_textures" in state
        assert isinstance(state["mat_textures"], dict)
        assert len(state["mat_textures"]) == 2

        # Round-trip
        model2 = PBRShadingModel(cfg)
        model2.init_textures(64, submesh_names=["a", "b"])
        model2.load_state_dict(state)
        for name in ["a", "b"]:
            assert torch.allclose(model.mat_textures[name], model2.mat_textures[name])

    def test_single_mesh_backward_compat(self):
        """init_textures without submesh_names should use single texture."""
        cfg = Config(render_mode="pbr", pbr=PBRConfig())
        model = PBRShadingModel(cfg)
        model.init_textures(64)
        assert not model.is_multi
        assert model.mat_texture is not None
        assert model.mat_texture.shape == (1, 64, 64, 8)

    def test_shade_submesh(self):
        """shade_submesh should use the named texture."""
        cfg = Config(render_mode="pbr", pbr=PBRConfig())
        model = PBRShadingModel(cfg)
        model.init_textures(64, submesh_names=["a", "b"])
        # Set different values for each submesh
        model.mat_textures["a"].data.fill_(0.0)
        model.mat_textures["b"].data.fill_(1.0)
        # shade_submesh should not crash (full integration test in Task 4)
        assert model.mat_textures["a"].mean().item() != model.mat_textures["b"].mean().item()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pbr_model_multi.py -v`
Expected: FAIL

- [ ] **Step 3: Implement multi-texture support in `PBRShadingModel`**

Modify `src/shading/pbr_model.py`:

1. Add fields:
```python
self.mat_textures: dict[str, nn.Parameter] = {}
self.is_multi: bool = False
```

2. Modify `init_textures`:
```python
def init_textures(self, resolution: int, submesh_names: list[str] | None = None) -> None:
    pbr_cfg = self.config.pbr
    eh, ew = pbr_cfg.env_map_res

    if submesh_names is not None:
        self.is_multi = True
        self.mat_textures = {}
        for name in submesh_names:
            self.mat_textures[name] = nn.Parameter(
                init_material_texture(resolution).data.to(self.device)
            )
    else:
        self.is_multi = False
        self.mat_texture = nn.Parameter(init_material_texture(resolution).data.to(self.device))

    self.env_map = EnvironmentMap(height=eh, width=ew).to(self.device)
```

3. Add `shade_submesh`:
```python
def shade_submesh(self, name: str, rast_out, texc, world_pos, normals,
                  view_dirs, camera, resolution, tangents=None, bitangents=None):
    """Shade a specific submesh using its named texture."""
    import nvdiffrast.torch as dr

    tex = self.mat_textures[name]

    # 1. Sample material
    mat_raw = dr.texture(tex, texc, filter_mode="linear", boundary_mode="clamp")
    base_color, roughness, metallic, _ = decode_material(mat_raw)

    # 2. Normal mapping
    tex_normal_raw = dr.texture(tex, texc, filter_mode="linear", boundary_mode="clamp")
    _, _, _, tex_normal = decode_material(tex_normal_raw)
    if tangents is not None and bitangents is not None:
        world_normal = (
            tangents * tex_normal[..., 0:1] +
            bitangents * tex_normal[..., 1:2] +
            normals * tex_normal[..., 2:3]
        )
        normals = F.normalize(world_normal, dim=-1)

    # 3. Reflect direction
    NdotV = (normals * view_dirs).sum(dim=-1, keepdim=True).clamp(0, 1)
    reflect_dir = 2.0 * NdotV * normals - view_dirs
    reflect_dir = reflect_dir / (reflect_dir.norm(dim=-1, keepdim=True) + 1e-8)

    # 4. Diffuse
    irradiance = self.env_map.sample_diffuse(normals)
    F0 = compute_F0(base_color, metallic)
    kd = (1.0 - metallic) * (1.0 - F0)
    diffuse = kd * base_color * irradiance

    # 5. Specular
    prefiltered_color = self.env_map.sample_specular(reflect_dir, roughness)
    NdotV_flat = NdotV.reshape(-1)
    roughness_flat = roughness.reshape(-1)
    scale, bias = sample_brdf(self.brdf_lut.to(self.device), NdotV_flat, roughness_flat)
    scale = scale.reshape(*NdotV.shape)
    bias = bias.reshape(*NdotV.shape)
    specular = (F0 * scale + bias) * prefiltered_color

    # 6. Combine
    rgb = diffuse + specular
    rgb = rgb.clamp(0.0, 1.0)

    # 7. Mask
    mask = (rast_out[..., 3] > 0).float()
    rgb = rgb * mask.unsqueeze(-1)

    self._last_debug = {
        "diffuse": diffuse.detach(),
        "specular": specular.detach(),
        "base_color": base_color.detach(),
        "roughness": roughness.detach(),
        "metallic": metallic.detach(),
        "normal": normals.detach(),
    }
    return rgb, mask
```

4. Modify `parameters()`:
```python
def parameters(self) -> list[nn.Parameter]:
    if self.is_multi:
        return list(self.mat_textures.values()) + [self.env_map.raw]
    return [self.mat_texture, self.env_map.raw]
```

5. Modify `state_dict()` / `load_state_dict()`:
```python
def state_dict(self) -> dict:
    if self.is_multi:
        return {
            "render_mode": "pbr",
            "is_multi": True,
            "mat_textures": {k: v.data.detach().cpu() for k, v in self.mat_textures.items()},
            "env_map": self.env_map.raw.data.detach().cpu(),
        }
    return {
        "render_mode": "pbr",
        "is_multi": False,
        "mat_texture": self.mat_texture.data.detach().cpu(),
        "env_map": self.env_map.raw.data.detach().cpu(),
    }

def load_state_dict(self, state: dict) -> None:
    if state.get("is_multi"):
        self.is_multi = True
        if "mat_textures" in state:
            self.mat_textures = {
                k: nn.Parameter(v.to(self.device))
                for k, v in state["mat_textures"].items()
            }
    else:
        self.is_multi = False
        if "mat_texture" in state:
            self.mat_texture = nn.Parameter(state["mat_texture"].to(self.device))
    if "env_map" in state:
        self.env_map.raw = nn.Parameter(state["env_map"].to(self.device))
    if "brdf_lut" in state:
        self.brdf_lut = state["brdf_lut"]
```

6. Modify `export()` to support multi:
```python
def export(self, output_dir: str) -> list[str]:
    if self.is_multi:
        return self._export_multi(output_dir)
    return self._export_single(output_dir)

def _export_single(self, output_dir: str) -> list[str]:
    # ... existing export code moved here unchanged ...

def _export_multi(self, output_dir: str) -> list[str]:
    import numpy as np
    from PIL import Image
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for name, tex in self.mat_textures.items():
        sub_dir = os.path.join(output_dir, name)
        os.makedirs(sub_dir, exist_ok=True)
        base_color, roughness, metallic, tex_normal = decode_material(tex)
        # base_color
        bc = base_color[0].clamp(0, 1).pow(1.0 / 2.2).detach().cpu().numpy()
        bc = (bc * 255).astype(np.uint8)
        p = os.path.join(sub_dir, "base_color.png")
        Image.fromarray(bc, "RGB").save(p); paths.append(p)
        # roughness
        r = roughness[0].clamp(0, 1).detach().cpu().numpy().repeat(3, axis=-1)
        r = (r * 255).astype(np.uint8)
        p = os.path.join(sub_dir, "roughness.png")
        Image.fromarray(r, "RGB").save(p); paths.append(p)
        # metallic
        m = metallic[0].clamp(0, 1).detach().cpu().numpy().repeat(3, axis=-1)
        m = (m * 255).astype(np.uint8)
        p = os.path.join(sub_dir, "metallic.png")
        Image.fromarray(m, "RGB").save(p); paths.append(p)
        # normal
        n_img = tex_normal[0].detach().cpu().numpy()
        n_img = ((n_img + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)
        p = os.path.join(sub_dir, "normal_map.png")
        Image.fromarray(n_img, "RGB").save(p); paths.append(p)
    # env_map
    p = os.path.join(output_dir, "env_map.png")
    self.env_map.export_image(p); paths.append(p)
    return paths
```

7. Modify `get_material_texture` / `set_material_texture`:
```python
def get_material_texture(self) -> torch.Tensor | dict[str, torch.Tensor]:
    if self.is_multi:
        return {k: v.data.detach().cpu() for k, v in self.mat_textures.items()}
    return self.mat_texture.data.detach().cpu()

def set_material_texture(self, texture: torch.Tensor | dict[str, torch.Tensor]) -> None:
    if isinstance(texture, dict):
        self.mat_textures = {
            k: nn.Parameter(v.to(self.device).contiguous())
            for k, v in texture.items()
        }
    else:
        self.mat_texture = nn.Parameter(texture.to(self.device).contiguous())
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_pbr_model_multi.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/shading/pbr_model.py tests/test_pbr_model_multi.py
git commit -m "feat: PBRShadingModel multi-texture support with backward compat"
```

---

## Task 4: Trainer — Multi-Mesh Training Loop

**Files:**
- Modify: `src/trainer.py`
- Test: `tests/test_trainer.py` (extend existing)

- [ ] **Step 1: Write failing test for multi-mesh trainer init**

```python
# Append to tests/test_trainer.py

class TestTrainerMultiMesh:
    def test_multi_mesh_trainer_init(self):
        """Trainer should detect MultiMeshData and create multiple renderers."""
        cfg = Config(
            render_mode="pbr",
            data=DataConfig(
                mesh_path="data/piano_260604/scene/lowpoly.glb",
                gt_dir="data/piano_260604/gt",
                camera_path="data/piano_260604/cameras.json",
            ),
            texture=TextureConfig(base_resolution=32),
            training=TrainingConfig(
                num_epochs=2, batch_size=2,
                resolution_schedule=[ResolutionStep(0, 32)],
            ),
        )
        trainer = Trainer(cfg)
        assert trainer.is_multi
        assert len(trainer.renderers) == 6
        assert len(trainer.submesh_names) == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trainer.py::TestTrainerMultiMesh -v`
Expected: FAIL with `AttributeError: 'Trainer' has no attribute 'is_multi'`

- [ ] **Step 3: Modify `Trainer.__init__` to handle `MultiMeshData`**

In `src/trainer.py`, modify `__init__`:

```python
from src.mesh import MeshData, MultiMeshData

class Trainer:
    def __init__(self, config, shading_model=None):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_multi = False
        self.renderers = {}  # dict for multi-mesh
        self.submesh_names = []

        # ---- 1. Load mesh ----
        mesh = load_mesh(config.data.mesh_path)

        if isinstance(mesh, MultiMeshData):
            self.is_multi = True
            self.multi_mesh = mesh
            self.submesh_names = [s.name for s in mesh.submeshes]
            # Create one renderer per submesh
            for sub in mesh.submeshes:
                v, f, uv, uvi, n, ni, t, bt = sub.to_torch()
                self.renderers[sub.name] = DifferentiableRenderer(
                    vertices=v, faces=f, uvs=uv, uv_idx=uvi,
                    normals=n, normal_idx=ni, tangents=t, bitangents=bt,
                    resolution=config.texture.base_resolution, device=self.device,
                )
        else:
            self.vertices, self.faces, self.uvs, self.uv_idx, self.normals, self.normal_idx, self.tangents, self.bitangents = mesh.to_torch()

        # ---- 2. Dataset ----
        self.dataset = GTDataset(
            gt_dir=config.data.gt_dir,
            camera_path=config.data.camera_path,
        )

        # ---- 3. Shading model ----
        if shading_model is not None:
            self.model = shading_model
        else:
            from src.shading import create_shading_model
            self.model = create_shading_model(config.render_mode, config)

        if self.is_multi:
            self.model.init_textures(config.texture.base_resolution, submesh_names=self.submesh_names)
        else:
            self.model.init_textures(config.texture.base_resolution)

        # ---- 4-8: unchanged ----
        self._rebuild_optimizer()
        self.scheduler = MultiStepLR(...)
        self.logger = create_logger(...)
        self.criterion = CombinedLoss(...)
        self.resolution_schedule = ...
        self.current_resolution = self._current_resolution(0)
        if self.is_multi:
            self.renderer = None  # no single renderer for multi
        else:
            self.renderer = self._create_renderer(self.current_resolution)
        self.history = ...
```

- [ ] **Step 4: Modify training loop to handle multi-mesh**

In `train()` method, add PBR multi-mesh branch:

```python
# Inside the batch loop, after camera setup:
if self.config.render_mode == "pbr" and self.is_multi:
    # Multi-mesh PBR path
    rendered_total = torch.zeros(1, res, res, 3, device=self.device)
    mask_total = torch.zeros(1, res, res, device=self.device)

    for sub_name in self.submesh_names:
        sub_renderer = self.renderers[sub_name]
        rast, texc, wpos, inorm, vdir, tang, btang = sub_renderer.rasterize_and_interpolate(camera)
        rgb_sub, mask_sub = self.model.shade_submesh(
            sub_name, rast, texc, wpos, inorm, vdir, camera, res, tang, btang)

        # Accumulate (works for non-overlapping meshes)
        rendered_total = rendered_total + rgb_sub
        mask_total = torch.max(mask_total, mask_sub)

    rendered = rendered_total
    mask = mask_total
    rendered = rendered.flip(1)
    mask = mask.flip(1)
elif self.config.render_mode == "pbr":
    # existing single-mesh PBR path
    ...
```

Similarly for PSNR evaluation and the `init_textures`/`_resize_textures`/`_apply_seam_padding` methods — add `is_multi` branches.

- [ ] **Step 5: Run multi-mesh trainer test**

Run: `pytest tests/test_trainer.py::TestTrainerMultiMesh -v`
Expected: PASS

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/trainer.py tests/test_trainer.py
git commit -m "feat: Trainer multi-mesh training loop with composite rendering"
```

---

## Task 5: Logger & Video — Multi-Mesh Debug Output

**Files:**
- Modify: `src/shading/pbr_logger.py`
- Modify: `src/video.py`

- [ ] **Step 1: Modify `PBRLogger.export_debug` for multi-mesh**

In `export_debug()`, add multi-mesh compare images and video:

```python
def export_debug(self, model, renderer, dataset, output_dir, epoch, history, device, current_resolution):
    # ... existing export code ...

    if hasattr(self, '_is_multi') and self._is_multi:
        self._export_compare_multi(model, self._renderers, dataset, output_dir, device, current_resolution)
    else:
        self._export_compare(model, renderer, dataset, output_dir, device, current_resolution)
```

The key change: compare images render each submesh separately, then composite into a single atlas.

For video: `render_video` needs a multi-mesh path that loops over submesh renderers.

- [ ] **Step 2: Add multi-mesh video support to `src/video.py`**

Add a `render_video_multi()` function that accepts a dict of renderers and a multi-texture model:

```python
def render_video_multi(mesh, renderers, shading_model, output_path, submesh_names, **vk):
    """Multi-mesh video: render all submeshes per frame, composite."""
    # Same orbit camera setup as render_video
    # Per frame: loop submesh_names, render each, composite, write frame
    ...
```

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/shading/pbr_logger.py src/video.py
git commit -m "feat: multi-mesh compare images, video rendering, and debug export"
```

---

## Task 6: Integration Test — Piano Multi-Mesh PBR Training

**Files:**
- Create: `configs/train_pbr_piano_multi.yaml`
- Test: `tests/test_integration_multi.py`

- [ ] **Step 1: Create piano multi-mesh config**

```yaml
# configs/train_pbr_piano_multi.yaml
render_mode: pbr

data:
  mesh_path: data/piano_260604/scene/original_with_mats.glb
  gt_dir: data/piano_260604/gt
  camera_path: data/piano_260604/cameras.json

texture:
  base_resolution: 512
  target_resolution: 2048

training:
  num_epochs: 10
  lr: 0.01
  batch_size: 4
  lr_decay: 0.5
  lr_decay_epochs: [3, 6, 8]
  resolution_schedule:
    - epoch: 0
      resolution: 512

pbr:
  env_map_res: [256, 512]
  brdf_lut_size: 256
  env_tv_weight: 0.0005
  env_l2_weight: 0.0001
```

- [ ] **Step 2: Write integration test**

```python
# tests/test_integration_multi.py
"""Integration test: multi-mesh PBR training on piano."""
import pytest
from src.config import load_config
from src.trainer import Trainer

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_piano_multi_mesh_training():
    """Multi-mesh piano training should run without errors."""
    cfg = load_config("configs/train_pbr_piano_multi.yaml")
    trainer = Trainer(cfg)
    assert trainer.is_multi
    assert len(trainer.renderers) == 6

    # Run a few epochs
    trainer.train(output_dir="output/test_piano_multi", checkpoint_every=0)
    assert len(trainer.history["epoch"]) > 0
    assert trainer.history["psnr"][-1] > 0
```

- [ ] **Step 3: Run integration test**

Run: `pytest tests/test_integration_multi.py -v`
Expected: PASS (runs 10 epochs, ~30 seconds)

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add configs/train_pbr_piano_multi.yaml tests/test_integration_multi.py
git commit -m "feat: piano multi-mesh PBR config and integration test"
```

---

## Task 7: Smoke Test — Run Full Piano Training

**Files:**
- No new files

- [ ] **Step 1: Run 2000-epoch piano multi-mesh training**

```bash
python main.py --config configs/train_pbr_piano_multi.yaml --mode train
```

Expected: Training completes, PSNR improves over epochs, no crashes.

- [ ] **Step 2: Verify helmet still works (backward compat)**

```bash
python main.py --config configs/train_pbr.yaml --mode train
```

Expected: Single-mesh helmet training unchanged.

- [ ] **Step 3: Commit & push if all good**

```bash
git push origin feature/multi-submesh
```
