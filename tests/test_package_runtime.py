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
    """Multi-mesh training output has Object_*/ subdirectories matched by name."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.package_runtime_asset import discover_submeshes

    epoch_dir = tmp_path / "epoch"
    # Use realistic submesh names matching both filesystem dir names and GLB names
    glb_names = ["Object_0", "Object_1"]
    for sub in glb_names:
        sub_dir = epoch_dir / sub
        sub_dir.mkdir(parents=True)
        for tex in ("base_color.png", "roughness.png", "metallic.png", "normal_map.png"):
            (sub_dir / tex).write_bytes(b"\x89PNG fake")

    submeshes = discover_submeshes(
        epoch_dir,
        scene_name="piano",
        glb_submesh_names=glb_names,
    )

    assert len(submeshes) == 2
    # Submesh order follows GLB order; names match input
    assert submeshes[0]["name"] == "Object_0"
    assert submeshes[0]["textures"]["base_color"] == "textures/Object_0/base_color.png"
    assert submeshes[1]["name"] == "Object_1"


def test_extract_glb_submesh_names_uses_gltf_loader(tmp_path, monkeypatch):
    """GLB submesh names come from src.gltf_loader, matching training-time names."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.package_runtime_asset import extract_glb_submesh_names

    fake_submeshes = [
        {"name": "helmet_mesh"},
        {"name": "visor_mesh"},
    ]

    def fake_load(path):
        class FakeMultiMesh:
            submeshes = [type("S", (), {"name": s["name"]})() for s in fake_submeshes]
        return FakeMultiMesh()

    monkeypatch.setattr("scripts.package_runtime_asset.load_mesh", fake_load)

    names = extract_glb_submesh_names("dummy.glb")

    assert names == ["helmet_mesh", "visor_mesh"]


def test_package_asset_creates_valid_zip(tmp_path, monkeypatch):
    """End-to-end: package_asset creates a zip with manifest + GLB + textures."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.package_runtime_asset import package_asset

    # Source GLB
    glb_path = tmp_path / "scene.glb"
    glb_path.write_bytes(b"fake glb bytes")

    # Source epoch dir (single-mesh layout)
    epoch_dir = tmp_path / "epoch2000"
    epoch_dir.mkdir()
    for tex in ("base_color.png", "roughness.png", "metallic.png", "normal_map.png"):
        (epoch_dir / tex).write_bytes(b"\x89PNG fake")
    (epoch_dir / "env_map.png").write_bytes(b"\x89PNG fake")
    (epoch_dir / "brdf_lut.png").write_bytes(b"\x89PNG fake")

    # Stub GLB submesh extraction
    monkeypatch.setattr(
        "scripts.package_runtime_asset.extract_glb_submesh_names",
        lambda p: ["helmet"],
    )

    output_zip = tmp_path / "out" / "helmet_pbr.zip"
    package_asset(
        glb_path=str(glb_path),
        epoch_dir=epoch_dir,
        scene_name="helmet",
        output_path=output_zip,
        epoch=2000,
        psnr_db=20.81,
    )

    assert output_zip.exists()
    with zipfile.ZipFile(output_zip) as zf:
        names = zf.namelist()
        assert "manifest.json" in names
        assert "geometry/scene.glb" in names
        assert "textures/env_map.png" in names
        assert "textures/brdf_lut.png" in names
        assert "textures/helmet/base_color.png" in names
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["scene_name"] == "helmet"
        assert manifest["submeshes"][0]["name"] == "helmet"


def test_update_scenes_index_appends_entry(tmp_path):
    """scenes_index.json gets a new entry when packaging a new scene."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.package_runtime_asset import update_scenes_index

    index_path = tmp_path / "scenes_index.json"
    update_scenes_index(
        index_path=index_path,
        scene_name="helmet",
        zip_filename="helmet_pbr.zip",
        psnr_db=20.81,
        epoch=2000,
    )
    update_scenes_index(
        index_path=index_path,
        scene_name="piano",
        zip_filename="piano_pbr.zip",
        psnr_db=28.80,
        epoch=2000,
    )

    data = json.loads(index_path.read_text())
    assert len(data) == 2
    assert data[0]["name"] == "helmet"
    assert data[1]["name"] == "piano"
    assert data[1]["file"] == "/scenes/piano_pbr.zip"

    # Re-update helmet → should replace, not duplicate
    update_scenes_index(
        index_path=index_path,
        scene_name="helmet",
        zip_filename="helmet_pbr.zip",
        psnr_db=21.0,
        epoch=2000,
    )
    data = json.loads(index_path.read_text())
    assert len(data) == 2
    helmet = [e for e in data if e["name"] == "helmet"][0]
    assert helmet["psnr_db"] == 21.0


def test_discover_submeshes_missing_dir_raises(tmp_path):
    """Raises when a GLB submesh name has no matching filesystem dir."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.package_runtime_asset import discover_submeshes

    epoch_dir = tmp_path / "epoch"
    # Only create Object_0, but GLB claims two submeshes
    (epoch_dir / "Object_0").mkdir(parents=True)
    for tex in ("base_color.png", "roughness.png", "metallic.png", "normal_map.png"):
        (epoch_dir / "Object_0" / tex).write_bytes(b"\x89PNG fake")

    with pytest.raises(ValueError, match="not found in epoch_dir"):
        discover_submeshes(epoch_dir, "test", ["Object_0", "Object_1"])


def test_discover_submeshes_empty_glb_names(tmp_path):
    """Empty glb_submesh_names falls back to scene_name."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.package_runtime_asset import discover_submeshes

    epoch_dir = tmp_path / "epoch"
    epoch_dir.mkdir()
    for tex in ("base_color.png", "roughness.png", "metallic.png", "normal_map.png"):
        (epoch_dir / tex).write_bytes(b"\x89PNG fake")

    submeshes = discover_submeshes(epoch_dir, scene_name="fallback_name", glb_submesh_names=[])
    assert len(submeshes) == 1
    assert submeshes[0]["name"] == "fallback_name"


def test_build_submesh_entry_missing_texture_raises(tmp_path):
    """Missing required texture raises FileNotFoundError."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.package_runtime_asset import _build_submesh_entry

    tex_dir = tmp_path / "tex"
    tex_dir.mkdir()
    # Only create 3 of 4 required textures
    for tex in ("base_color.png", "roughness.png", "metallic.png"):
        (tex_dir / tex).write_bytes(b"\x89PNG fake")

    with pytest.raises(FileNotFoundError, match="Missing required texture"):
        _build_submesh_entry("test", tex_dir, "textures/test")


def test_package_asset_missing_env_map_raises(tmp_path, monkeypatch):
    """package_asset raises FileNotFoundError when env_map.png is missing."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.package_runtime_asset import package_asset

    glb_path = tmp_path / "scene.glb"
    glb_path.write_bytes(b"fake")
    epoch_dir = tmp_path / "epoch"
    epoch_dir.mkdir()
    for tex in ("base_color.png", "roughness.png", "metallic.png", "normal_map.png"):
        (epoch_dir / tex).write_bytes(b"\x89PNG fake")
    # Note: NO env_map.png or brdf_lut.png

    monkeypatch.setattr(
        "scripts.package_runtime_asset.extract_glb_submesh_names",
        lambda p: ["test"],
    )

    with pytest.raises(FileNotFoundError, match="env_map.png"):
        package_asset(
            glb_path=str(glb_path),
            epoch_dir=epoch_dir,
            scene_name="test",
            output_path=tmp_path / "out.zip",
            epoch=100,
        )


def test_package_asset_generates_brdf_lut_from_pt(tmp_path, monkeypatch):
    """When brdf_lut.pt exists, packaging regenerates a proper data PNG."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.package_runtime_asset import package_asset
    import zipfile
    import io
    from PIL import Image
    import numpy as np

    # Create fake brdf_lut.pt with known values
    import torch
    fake_lut = torch.zeros(256, 256, 2)
    fake_lut[:, :, 0] = 0.5   # scale = 0.5 everywhere → R=128
    fake_lut[:, :, 1] = 0.25  # bias = 0.25 → G=64
    pt_path = tmp_path / "epoch" / "brdf_lut.pt"
    pt_path.parent.mkdir(parents=True)
    torch.save(fake_lut, pt_path)
    # Also create brdf_lut.png to verify .pt takes precedence
    (pt_path.parent / "brdf_lut.png").write_bytes(b"fake debug png")

    # Other required files
    glb_path = tmp_path / "scene.glb"
    glb_path.write_bytes(b"fake glb")
    epoch_dir = tmp_path / "epoch"
    for tex in ("base_color.png", "roughness.png", "metallic.png", "normal_map.png"):
        (epoch_dir / tex).write_bytes(b"\x89PNG fake")
    (epoch_dir / "env_map.png").write_bytes(b"\x89PNG fake")

    monkeypatch.setattr(
        "scripts.package_runtime_asset.extract_glb_submesh_names",
        lambda p: ["test"],
    )

    output_zip = tmp_path / "out.zip"
    package_asset(
        glb_path=str(glb_path),
        epoch_dir=epoch_dir,
        scene_name="test",
        output_path=output_zip,
        epoch=100,
        psnr_db=None,
    )

    # Read back the generated brdf_lut.png from the zip
    with zipfile.ZipFile(output_zip) as zf:
        brdf_png = zf.read("textures/brdf_lut.png")
        img = Image.open(io.BytesIO(brdf_png))
        assert img.size == (256, 256), f"Expected 256x256, got {img.size}"
        arr = np.array(img)
        # R channel should be ~128 (0.5 * 255 = 127.5)
        assert abs(arr[:, :, 0].mean() - 128) < 2, f"R mean={arr[:,:,0].mean()}"
        # G channel should be ~64 (0.25 * 255 = 63.75)
        assert abs(arr[:, :, 1].mean() - 64) < 2, f"G mean={arr[:,:,1].mean()}"


def test_extract_glb_submesh_names_single_mesh_fallback(monkeypatch):
    """extract_glb_submesh_names handles MeshData (non-multi) fallback."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.package_runtime_asset import extract_glb_submesh_names

    class FakeSingleMesh:
        name = "single_mesh_name"
        # Note: no 'submeshes' attribute

    monkeypatch.setattr(
        "scripts.package_runtime_asset.load_mesh",
        lambda p: FakeSingleMesh(),
    )

    names = extract_glb_submesh_names("dummy.glb")
    assert names == ["single_mesh_name"]


def test_extract_glb_submesh_names_no_name_uses_default(monkeypatch):
    """extract_glb_submesh_names falls back to 'mesh_0' when mesh has no name."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.package_runtime_asset import extract_glb_submesh_names

    class FakeUnnamedMesh:
        # No 'name' attribute at all
        pass

    monkeypatch.setattr(
        "scripts.package_runtime_asset.load_mesh",
        lambda p: FakeUnnamedMesh(),
    )

    names = extract_glb_submesh_names("dummy.glb")
    assert names == ["mesh_0"]


def test_discover_submeshes_non_numerical_order(tmp_path):
    """GLB submesh order may differ from filesystem alphabetical order.

    Regression test for piano scene where gltf_loader returns:
      ['Object_0', 'Object_5', 'Object_1', 'Object_2', 'Object_3', 'Object_4']
    but filesystem dirs are alphabetical. Matching must be by NAME, not by position.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.package_runtime_asset import discover_submeshes, package_asset
    import zipfile
    import json

    epoch_dir = tmp_path / "epoch"
    glb_names = ["Object_0", "Object_5", "Object_1", "Object_2", "Object_3", "Object_4"]
    for name in glb_names:
        sub_dir = epoch_dir / name
        sub_dir.mkdir(parents=True)
        # Use distinct fake bytes per submesh so we can verify mapping
        for tex in ("base_color.png", "roughness.png", "metallic.png", "normal_map.png"):
            (sub_dir / tex).write_bytes(f"{name}_{tex}".encode())

    submeshes = discover_submeshes(epoch_dir, "piano", glb_names)

    # Verify all 6 submeshes discovered
    assert len(submeshes) == 6
    # Verify names match GLB order
    assert [s["name"] for s in submeshes] == glb_names
    # Verify textures paths use the submesh name (not a positional index)
    for sm in submeshes:
        name = sm["name"]
        assert sm["textures"]["base_color"] == f"textures/{name}/base_color.png"

    # End-to-end: package and verify texture content matches name
    glb_path = tmp_path / "scene.glb"
    glb_path.write_bytes(b"fake glb")
    (epoch_dir / "env_map.png").write_bytes(b"\x89PNG fake env")
    (epoch_dir / "brdf_lut.png").write_bytes(b"\x89PNG fake brdf")
    output_zip = tmp_path / "out.zip"

    import scripts.package_runtime_asset as pra
    # Bypass GLB extraction (we don't have a real GLB)
    orig_extract = pra.extract_glb_submesh_names
    pra.extract_glb_submesh_names = lambda p: glb_names
    try:
        package_asset(
            glb_path=str(glb_path),
            epoch_dir=epoch_dir,
            scene_name="piano",
            output_path=output_zip,
            epoch=2000,
            psnr_db=28.0,
        )
    finally:
        pra.extract_glb_submesh_names = orig_extract

    # Verify zip contents: texture bytes match submesh name
    with zipfile.ZipFile(output_zip) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        assert [s["name"] for s in manifest["submeshes"]] == glb_names
        for name in glb_names:
            content = zf.read(f"textures/{name}/base_color.png").decode()
            assert content == f"{name}_base_color.png", (
                f"Wrong texture mapping for {name}: got '{content}'"
            )
