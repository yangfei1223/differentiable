"""Tests for the runtime asset packaging script."""
import json
import sys
import zipfile
from pathlib import Path

import pytest


def test_manifest_minimal_structure(tmp_path):
    """A packaged zip must contain a manifest with required fields."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.package_runtime_asset import build_manifest

    manifest = build_manifest(
        scene_name="helmet",
        glb_path="geometry/scene.glb",
        submeshes=[{"name": "helmet", "textures": {
            "base_color": "textures/helmet/base_color.png",
            "roughness": "textures/helmet/roughness.png",
            "metallic": "textures/helmet/metallic.png",
            "normal_map": "textures/helmet/normal_map.png",
        }}],
        env_map_path="textures/env_map.png",
        brdf_lut_path="textures/brdf_lut.png",
        epoch=2000,
        psnr_db=20.81,
    )

    assert manifest["schema_version"] == 1
    assert manifest["scene_name"] == "helmet"
    assert manifest["geometry"]["glb_path"] == "geometry/scene.glb"
    assert manifest["environment"]["env_map_path"] == "textures/env_map.png"
    assert manifest["brdf_lut_path"] == "textures/brdf_lut.png"
    assert len(manifest["submeshes"]) == 1
    assert manifest["submeshes"][0]["name"] == "helmet"
    assert manifest["submeshes"][0]["match_by"] == "primitive_name"
    assert manifest["generator"]["render_mode"] == "pbr"
    assert manifest["generator"]["epoch"] == 2000
    assert manifest["generator"]["psnr_db"] == 20.81


def test_discover_submeshes_single(tmp_path):
    """Single-mesh training output has flat texture files."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.package_runtime_asset import discover_submeshes

    epoch_dir = tmp_path / "epoch"
    epoch_dir.mkdir()
    for tex in ("base_color.png", "roughness.png", "metallic.png", "normal_map.png"):
        (epoch_dir / tex).write_bytes(b"\x89PNG fake")

    submeshes = discover_submeshes(epoch_dir, scene_name="helmet", glb_submesh_names=["helmet"])

    assert len(submeshes) == 1
    assert submeshes[0]["name"] == "helmet"
    assert submeshes[0]["match_by"] == "primitive_name"
    assert submeshes[0]["textures"]["base_color"] == "textures/helmet/base_color.png"


def test_discover_submeshes_multi(tmp_path):
    """Multi-mesh training output has Object_*/ subdirectories."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.package_runtime_asset import discover_submeshes

    epoch_dir = tmp_path / "epoch"
    for sub in ("Object_0", "Object_1"):
        sub_dir = epoch_dir / sub
        sub_dir.mkdir(parents=True)
        for tex in ("base_color.png", "roughness.png", "metallic.png", "normal_map.png"):
            (sub_dir / tex).write_bytes(b"\x89PNG fake")

    submeshes = discover_submeshes(
        epoch_dir,
        scene_name="piano",
        glb_submesh_names=["mesh_0", "mesh_1"],
    )

    assert len(submeshes) == 2
    # Submesh order matches GLB primitive order; names from glb_submesh_names
    assert submeshes[0]["name"] == "mesh_0"
    assert submeshes[0]["textures"]["base_color"] == "textures/mesh_0/base_color.png"
    assert submeshes[1]["name"] == "mesh_1"
