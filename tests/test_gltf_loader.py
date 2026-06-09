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
