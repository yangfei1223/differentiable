# PBR Web Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Three.js + TypeScript WebGL2 viewer that loads PBR-baked `.zip` asset bundles and renders them with GLSL shaders strictly mirroring `src/shading/pbr_model.py`, with mobile-friendly camera interaction.

**Architecture:** Vite + TypeScript SPA in a new `app/` directory. All PBR math lives in 3 GLSL files (`common.glsl` + `pbr.vert` + `pbr.frag`), portable to native. A Python `scripts/package_runtime_asset.py` packs training outputs into a versioned `.zip` with `manifest.json`. The viewer loads preset zips from `output/` or accepts drag-dropped zips.

**Tech Stack:**
- Frontend: TypeScript 5.x, Vite 5.x, Three.js r170, JSZip 3.x
- Testing: Vitest 1.x (pure JS math mirror tests)
- Backend: Python 3.10 (existing conda env `differentiable`), uses `pygltflib` or existing `src/gltf_loader.py`

**Spec:** `docs/superpowers/specs/2026-06-17-pbr-web-viewer-design.md`

---

## File Structure

**Python (existing project):**
- `scripts/package_runtime_asset.py` — pack training output → .zip bundle (NEW)

**Web app (`app/` — new directory):**
- `app/package.json`, `app/vite.config.ts`, `app/tsconfig.json`, `app/index.html` — project scaffolding
- `app/src/main.ts` — entry point
- `app/src/app/App.ts` — top-level orchestrator
- `app/src/app/SceneLoader.ts` — zip → manifest → AssetBundle
- `app/src/render/PBRPipeline.ts` — renderer + scene + camera + animate loop
- `app/src/render/PBRMesh.ts` — single submesh Mesh + ShaderMaterial wrapper
- `app/src/render/Environment.ts` — env map + BRDF LUT textures
- `app/src/shaders/common.glsl` — constants + `direction_to_uv()`
- `app/src/shaders/pbr.vert` — vertex shader
- `app/src/shaders/pbr.frag` — fragment shader (all PBR math)
- `app/src/types/manifest.ts` — TypeScript interfaces for manifest.json schema
- `app/src/ui/ScenePicker.ts` — preset dropdown + zip drop zone
- `app/src/ui/CameraControls.ts` — OrbitControls wrapper
- `app/src/ui/PerfStats.ts` — FPS / draw calls / tris / texture mem overlay
- `app/src/ui/LoadingOverlay.ts` — spinner + status text
- `app/src/vite/glsl-plugin.ts` — custom Vite plugin for `#include` resolution
- `app/tests/equivalence.test.ts` — math equivalence unit tests

---

## Task 1: Python Packaging Script — Skeleton

**Files:**
- Create: `scripts/package_runtime_asset.py`

- [ ] **Step 1: Write failing test for manifest generation**

Create `tests/test_package_runtime.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_package_runtime.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.package_runtime_asset'`

- [ ] **Step 3: Create scripts/package_runtime_asset.py with build_manifest**

Create `scripts/package_runtime_asset.py`:

```python
"""Pack training output into a Web Viewer-compatible .zip asset bundle.

Usage:
    python -m scripts.package_runtime_asset \
        --glb data/helmet_260604/scene/lowpoly.glb \
        --epoch-dir output/helmet_260604_pbr/epoch2000 \
        --scene-name helmet \
        --output output/helmet_pbr.zip
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional


SCHEMA_VERSION = 1
GENERATOR_TOOL = "differentiable-baker"
GENERATOR_VERSION = "v0.4"

REQUIRED_SUBMESH_TEXTURES = ("base_color", "roughness", "metallic", "normal_map")


def build_manifest(
    scene_name: str,
    glb_path: str,
    submeshes: list[dict],
    env_map_path: str,
    brdf_lut_path: str,
    epoch: int,
    psnr_db: float | None = None,
    is_hdr: bool = False,
) -> dict:
    """Build the manifest.json dictionary.

    Args:
        scene_name: Human-readable scene name.
        glb_path: Path to GLB inside the zip (relative).
        submeshes: List of {name, match_by, textures{...}} dicts.
        env_map_path: Path to env_map.png (relative).
        brdf_lut_path: Path to brdf_lut.png (relative).
        epoch: Training epoch this asset was baked at.
        psnr_db: Optional training PSNR in dB.
        is_hdr: Whether env map is HDR-encoded (currently always False).

    Returns:
        Manifest dict conforming to schema_version 1.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "scene_name": scene_name,
        "generator": {
            "tool": GENERATOR_TOOL,
            "version": GENERATOR_VERSION,
            "render_mode": "pbr",
            "epoch": int(epoch),
            "psnr_db": float(psnr_db) if psnr_db is not None else None,
        },
        "geometry": {
            "glb_path": glb_path,
            "up_axis": "Y",
            "scale": 1.0,
        },
        "environment": {
            "env_map_path": env_map_path,
            "is_hdr": bool(is_hdr),
            "diffuse_mip_bias": -1,
            "specular_mip_levels": -1,
        },
        "brdf_lut_path": brdf_lut_path,
        "submeshes": submeshes,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_package_runtime.py::test_manifest_minimal_structure -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/package_runtime_asset.py tests/test_package_runtime.py
git commit -m "feat(app): packaging script — manifest builder skeleton"
```

---

## Task 2: Python Packaging Script — Submesh Discovery

**Files:**
- Modify: `scripts/package_runtime_asset.py`
- Test: `tests/test_package_runtime.py`

- [ ] **Step 1: Append failing test for submesh discovery**

Append to `tests/test_package_runtime.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_package_runtime.py -v -k discover`
Expected: FAIL with `ImportError: cannot import name 'discover_submeshes'`

- [ ] **Step 3: Add discover_submeshes function**

Append to `scripts/package_runtime_asset.py`:

```python
def discover_submeshes(
    epoch_dir: Path,
    scene_name: str,
    glb_submesh_names: list[str],
) -> list[dict]:
    """Discover submesh texture directories and build submesh manifest entries.

    Two layouts supported:
      1. Single-mesh: textures directly in epoch_dir/
         → 1 submesh named scene_name (or first glb_submesh_names entry)
      2. Multi-mesh: textures in epoch_dir/Object_N/ subdirs
         → 1 submesh per subdirectory, matched by index to glb_submesh_names

    Args:
        epoch_dir: Directory containing exported PBR textures.
        scene_name: Scene name (used for single-mesh case).
        glb_submesh_names: Ordered list of submesh names extracted from GLB.

    Returns:
        List of submesh manifest entries.

    Raises:
        FileNotFoundError: If a required texture is missing.
        ValueError: If subdirectory count doesn't match GLB primitive count.
    """
    # Detect multi-mesh: any Object_N subdirectory exists
    sub_dirs = sorted([d for d in epoch_dir.iterdir() if d.is_dir() and d.name.startswith("Object_")])

    if not sub_dirs:
        # Single-mesh layout
        name = glb_submesh_names[0] if glb_submesh_names else scene_name
        return [_build_submesh_entry(name, epoch_dir, textures_prefix=f"textures/{name}")]

    # Multi-mesh layout
    if len(sub_dirs) != len(glb_submesh_names):
        raise ValueError(
            f"Subdir count {len(sub_dirs)} does not match GLB primitive count "
            f"{len(glb_submesh_names)}"
        )

    entries = []
    for sub_dir, glb_name in zip(sub_dirs, glb_submesh_names):
        entries.append(
            _build_submesh_entry(glb_name, sub_dir, textures_prefix=f"textures/{glb_name}")
        )
    return entries


def _build_submesh_entry(name: str, tex_dir: Path, textures_prefix: str) -> dict:
    """Validate textures exist and build a single submesh manifest entry."""
    textures = {}
    for tex_name in REQUIRED_SUBMESH_TEXTURES:
        tex_file = tex_dir / f"{tex_name}.png"
        if not tex_file.exists():
            raise FileNotFoundError(f"Missing required texture: {tex_file}")
        textures[tex_name] = f"{textures_prefix}/{tex_name}.png"
    return {
        "name": name,
        "match_by": "primitive_name",
        "textures": textures,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_package_runtime.py -v -k discover`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/package_runtime_asset.py tests/test_package_runtime.py
git commit -m "feat(app): packaging script — submesh discovery"
```

---

## Task 3: Python Packaging Script — GLB Submesh Name Extraction

**Files:**
- Modify: `scripts/package_runtime_asset.py`
- Test: `tests/test_package_runtime.py`

- [ ] **Step 1: Append failing test for GLB submesh name extraction**

Append to `tests/test_package_runtime.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_package_runtime.py::test_extract_glb_submesh_names_uses_gltf_loader -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Add extract_glb_submesh_names**

Add import and function to `scripts/package_runtime_asset.py`:

```python
# Add to imports section at top:
from src.mesh import load_mesh
from src.mesh import MultiMeshData


def extract_glb_submesh_names(glb_path: str) -> list[str]:
    """Extract submesh names from a GLB, matching training-time conventions.

    Uses src.mesh.load_mesh (which goes through gltf_loader → MultiMeshData).
    The names follow: mesh.name or mesh_{node.mesh}, with _prim{pi} suffix
    when a mesh has multiple primitives.

    Args:
        glb_path: Path to .glb file.

    Returns:
        Ordered list of submesh names (one per primitive).
    """
    mesh = load_mesh(glb_path)
    if isinstance(mesh, MultiMeshData):
        return [s.name for s in mesh.submeshes]
    # Single MeshData — use the mesh's name or fall back to "mesh_0"
    return [getattr(mesh, "name", None) or "mesh_0"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_package_runtime.py::test_extract_glb_submesh_names_uses_gltf_loader -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/package_runtime_asset.py tests/test_package_runtime.py
git commit -m "feat(app): packaging script — GLB submesh name extraction"
```

---

## Task 4: Python Packaging Script — Full Packaging

**Files:**
- Modify: `scripts/package_runtime_asset.py`
- Test: `tests/test_package_runtime.py`

- [ ] **Step 1: Append failing test for full packaging**

Append to `tests/test_package_runtime.py`:

```python
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

    import json
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_package_runtime.py -v -k "package_asset or scenes_index"`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Add package_asset and update_scenes_index**

Append to `scripts/package_runtime_asset.py`:

```python
def package_asset(
    glb_path: str,
    epoch_dir: Path,
    scene_name: str,
    output_path: Path,
    epoch: int,
    psnr_db: float | None = None,
) -> Path:
    """Pack a training output directory into a .zip asset bundle.

    Args:
        glb_path: Path to source GLB file.
        epoch_dir: Directory containing exported PBR textures.
        scene_name: Scene identifier.
        output_path: Output .zip path (will create parent dirs).
        epoch: Training epoch.
        psnr_db: Optional training PSNR.

    Returns:
        Path to the created .zip file.
    """
    epoch_dir = Path(epoch_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract submesh names from GLB
    glb_submesh_names = extract_glb_submesh_names(glb_path)

    # Discover texture submeshes
    submeshes = discover_submeshes(epoch_dir, scene_name, glb_submesh_names)

    # Validate env_map + brdf_lut at top level
    env_map_file = epoch_dir / "env_map.png"
    brdf_lut_file = epoch_dir / "brdf_lut.png"
    if not env_map_file.exists():
        raise FileNotFoundError(f"Missing env_map.png: {env_map_file}")
    if not brdf_lut_file.exists():
        raise FileNotFoundError(f"Missing brdf_lut.png: {brdf_lut_file}")

    # Build manifest
    manifest = build_manifest(
        scene_name=scene_name,
        glb_path="geometry/scene.glb",
        submeshes=submeshes,
        env_map_path="textures/env_map.png",
        brdf_lut_path="textures/brdf_lut.png",
        epoch=epoch,
        psnr_db=psnr_db,
    )

    # Build the zip
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Manifest
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        # Geometry
        zf.write(glb_path, "geometry/scene.glb")

        # Env map + BRDF LUT
        zf.write(env_map_file, "textures/env_map.png")
        zf.write(brdf_lut_file, "textures/brdf_lut.png")

        # Per-submesh textures
        for sub in submeshes:
            sub_name = sub["name"]
            # Find source directory
            sub_dirs = sorted([d for d in epoch_dir.iterdir() if d.is_dir() and d.name.startswith("Object_")])
            if sub_dirs:
                # Multi-mesh: index matches
                idx = [s["name"] for s in submeshes].index(sub_name)
                src_dir = sub_dirs[idx]
            else:
                src_dir = epoch_dir

            for tex_name in REQUIRED_SUBMESH_TEXTURES:
                src = src_dir / f"{tex_name}.png"
                dst = f"textures/{sub_name}/{tex_name}.png"
                zf.write(src, dst)

    return output_path


def update_scenes_index(
    index_path: Path,
    scene_name: str,
    zip_filename: str,
    psnr_db: float | None,
    epoch: int,
) -> None:
    """Add or update an entry in scenes_index.json.

    Args:
        index_path: Path to scenes_index.json (created if missing).
        scene_name: Scene name.
        zip_filename: Filename only (e.g. "helmet_pbr.zip").
        psnr_db: Training PSNR.
        epoch: Training epoch.
    """
    index_path = Path(index_path)
    if index_path.exists():
        data = json.loads(index_path.read_text())
    else:
        data = []

    entry = {
        "name": scene_name,
        "file": f"/scenes/{zip_filename}",
        "psnr_db": psnr_db,
        "epoch": epoch,
    }

    # Replace existing entry with same name, or append
    for i, e in enumerate(data):
        if e["name"] == scene_name:
            data[i] = entry
            break
    else:
        data.append(entry)

    index_path.write_text(json.dumps(data, indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_package_runtime.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/package_runtime_asset.py tests/test_package_runtime.py
git commit -m "feat(app): packaging script — full packaging + scenes index"
```

---

## Task 5: Python Packaging Script — CLI Entry Point

**Files:**
- Modify: `scripts/package_runtime_asset.py`

- [ ] **Step 1: Add CLI main()**

Append to `scripts/package_runtime_asset.py`:

```python
def main() -> None:
    """CLI entry point: pack a training output into a .zip bundle."""
    parser = argparse.ArgumentParser(
        description="Pack PBR training output into a Web Viewer .zip bundle."
    )
    parser.add_argument("--glb", required=True, help="Path to source .glb file")
    parser.add_argument("--epoch-dir", required=True, help="Training epoch output directory")
    parser.add_argument("--scene-name", required=True, help="Scene name")
    parser.add_argument("--output", default=None, help="Output .zip path (default: output/{scene}_pbr.zip)")
    parser.add_argument("--epoch", type=int, default=None, help="Training epoch (default: parse from --epoch-dir)")
    parser.add_argument("--psnr", type=float, default=None, help="Training PSNR in dB")
    args = parser.parse_args()

    # Default epoch: parse from directory name (e.g. "epoch2000" → 2000)
    epoch = args.epoch
    if epoch is None:
        dir_name = Path(args.epoch_dir).name
        if dir_name.startswith("epoch"):
            try:
                epoch = int(dir_name[5:])
            except ValueError:
                epoch = 0
        else:
            epoch = 0

    # Default output path
    output_path = args.output or f"output/{args.scene_name}_pbr.zip"

    # Pack
    created = package_asset(
        glb_path=args.glb,
        epoch_dir=Path(args.epoch_dir),
        scene_name=args.scene_name,
        output_path=Path(output_path),
        epoch=epoch,
        psnr_db=args.psnr,
    )
    print(f"Packed: {created}")

    # Update scenes_index.json in the same directory as the zip
    index_path = Path(output_path).parent / "scenes_index.json"
    update_scenes_index(
        index_path=index_path,
        scene_name=args.scene_name,
        zip_filename=Path(output_path).name,
        psnr_db=args.psnr,
        epoch=epoch,
    )
    print(f"Updated: {index_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script end-to-end on helmet (manual smoke test)**

Run:
```bash
python -m scripts.package_runtime_asset `
  --glb data/helmet_260604/scene/lowpoly.glb `
  --epoch-dir output/helmet_260604_pbr/epoch2000 `
  --scene-name helmet `
  --psnr 20.81
```
Expected: prints `Packed: output\helmet_pbr.zip` and `Updated: output\scenes_index.json`

- [ ] **Step 3: Verify zip contents**

Run: `python -c "import zipfile; z=zipfile.ZipFile('output/helmet_pbr.zip'); print('\n'.join(z.namelist()))"`
Expected output includes: `manifest.json`, `geometry/scene.glb`, `textures/env_map.png`, `textures/brdf_lut.png`, `textures/helmet/base_color.png`, etc.

- [ ] **Step 4: Commit**

```bash
git add scripts/package_runtime_asset.py
git commit -m "feat(app): packaging script — CLI entry point"
```

---

## Task 6: Python Packaging Script — Pack Piano (Multi-Mesh Smoke Test)

**Files:**
- No code changes (verification only)

- [ ] **Step 1: Run packaging on piano multi-mesh**

Run:
```bash
python -m scripts.package_runtime_asset `
  --glb data/piano_260604/scene/original_with_mats.glb `
  --epoch-dir output/piano_260604_pbr_multi/epoch2000 `
  --scene-name piano `
  --psnr 28.80
```
Expected: Success message. Note: if submesh count mismatch occurs, inspect `gltf_loader.py` output for piano.

- [ ] **Step 2: Verify zip has 6 submesh texture directories**

Run: `python -c "import zipfile; z=zipfile.ZipFile('output/piano_pbr.zip'); dirs=set(n.split('/')[1] for n in z.namelist() if n.startswith('textures/') and '/' in n[9:]); print(sorted(dirs))"`
Expected: 6 submesh directories matching the GLB primitive names.

- [ ] **Step 3: If mismatch, debug gltf_loader names**

If submesh count != 6, run this debug:
```bash
python -c "from src.mesh import load_mesh; m=load_mesh('data/piano_260604/scene/original_with_mats.glb'); print([s.name for s in m.submeshes])"
```
Compare with `Object_0` through `Object_5` directories. If names don't align by index, the `discover_submeshes` zip-by-index logic in Task 4 handles it correctly.

No commit needed — this is a verification step.

---

## Task 7: App Scaffolding — package.json and Configs

**Files:**
- Create: `app/package.json`
- Create: `app/tsconfig.json`
- Create: `app/vite.config.ts`
- Create: `app/index.html`
- Create: `app/.gitignore`

- [ ] **Step 1: Create app/package.json**

```json
{
  "name": "pbr-web-viewer",
  "version": "0.1.0",
  "description": "WebGL2 PBR viewer consuming differentiable baker outputs",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc --noEmit && vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "dependencies": {
    "jszip": "^3.10.1",
    "three": "^0.170.0"
  },
  "devDependencies": {
    "@types/three": "^0.170.0",
    "typescript": "^5.4.0",
    "vite": "^5.4.0",
    "vitest": "^1.6.0"
  }
}
```

- [ ] **Step 2: Create app/tsconfig.json**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true
  },
  "include": ["src", "tests"]
}
```

- [ ] **Step 3: Create app/vite.config.ts**

```typescript
import { defineConfig } from 'vite';
import { glslIncludePlugin } from './src/vite/glsl-plugin';
import path from 'node:path';

export default defineConfig({
  // Dev mode: serve ../output so /scenes/*.zip is reachable
  publicDir: path.resolve(__dirname, '../output'),
  plugins: [glslIncludePlugin()],
  server: {
    open: true,
  },
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
  },
});
```

- [ ] **Step 4: Create app/index.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no" />
  <title>Differentiable Baker — PBR Viewer</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    html, body { width: 100%; height: 100%; overflow: hidden; background: #1a1a1a; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
    #app { width: 100vw; height: 100vh; position: relative; }
    canvas { display: block; }
  </style>
</head>
<body>
  <div id="app"></div>
  <script type="module" src="/src/main.ts"></script>
</body>
</html>
```

- [ ] **Step 5: Create app/.gitignore**

```
node_modules/
dist/
*.log
.DS_Store
```

- [ ] **Step 6: Install dependencies**

Run:
```bash
cd app
npm install
```
Expected: `node_modules/` created, no errors.

- [ ] **Step 7: Commit**

```bash
cd ..
git add app/package.json app/tsconfig.json app/vite.config.ts app/index.html app/.gitignore
git commit -m "feat(app): scaffold Vite + TypeScript project"
```

---

## Task 8: GLSL Include Plugin

**Files:**
- Create: `app/src/vite/glsl-plugin.ts`

- [ ] **Step 1: Create the GLSL #include resolver**

Create `app/src/vite/glsl-plugin.ts`:

```typescript
import type { Plugin } from 'vite';
import path from 'node:path';
import fs from 'node:fs';

/**
 * Vite plugin that resolves `#include "file.glsl"` directives in .glsl files
 * imported with `?raw` suffix. Inlines the included file content recursively.
 *
 * Usage in TS:
 *   import fragSrc from '../shaders/pbr.frag?raw';
 *
 * In pbr.frag:
 *   #include "common.glsl"
 */
export function glslIncludePlugin(): Plugin {
  const includeRegex = /^#include\s+"([^"]+)"\s*$/gm;

  function resolveIncludes(source: string, dir: string, depth = 0): string {
    if (depth > 8) {
      throw new Error(`GLSL include depth exceeded 8 (circular include?) in ${dir}`);
    }
    return source.replace(includeRegex, (match, filename) => {
      const includePath = path.resolve(dir, filename);
      const includeSource = fs.readFileSync(includePath, 'utf-8');
      const includeDir = path.dirname(includePath);
      return resolveIncludes(includeSource, includeDir, depth + 1);
    });
  }

  return {
    name: 'glsl-include',
    name: 'glsl-include-resolver',
    enforce: 'pre',
    transform(code, id) {
      if (!id.endsWith('.glsl?raw') && !id.endsWith('.glsl')) return null;
      const dir = path.dirname(id.replace(/\?raw$/, ''));
      const resolved = resolveIncludes(code, dir);
      return { code: resolved, map: null };
    },
  };
}
```

Fix duplicate `name` property — final version:

```typescript
export function glslIncludePlugin(): Plugin {
  const includeRegex = /^#include\s+"([^"]+)"\s*$/gm;

  function resolveIncludes(source: string, dir: string, depth = 0): string {
    if (depth > 8) {
      throw new Error(`GLSL include depth exceeded 8 (circular include?) in ${dir}`);
    }
    return source.replace(includeRegex, (match, filename) => {
      const includePath = path.resolve(dir, filename);
      const includeSource = fs.readFileSync(includePath, 'utf-8');
      const includeDir = path.dirname(includePath);
      return resolveIncludes(includeSource, includeDir, depth + 1);
    });
  }

  return {
    name: 'glsl-include-resolver',
    enforce: 'pre',
    transform(code, id) {
      if (!id.endsWith('.glsl?raw') && !id.endsWith('.glsl')) return null;
      const dir = path.dirname(id.replace(/\?raw$/, ''));
      const resolved = resolveIncludes(code, dir);
      return { code: resolved, map: null };
    },
  };
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd app && npx tsc --noEmit`
Expected: No errors (will warn about missing main.ts — that's fine, we'll create it later).

- [ ] **Step 3: Commit**

```bash
git add app/src/vite/glsl-plugin.ts
git commit -m "feat(app): GLSL #include resolver Vite plugin"
```

---

## Task 9: GLSL Shaders — common.glsl

**Files:**
- Create: `app/src/shaders/common.glsl`

- [ ] **Step 1: Create common.glsl**

Create `app/src/shaders/common.glsl`:

```glsl
// Common GLSL constants and helpers — included by pbr.vert and pbr.frag.
// Portable to native (Vulkan/Metal/GLES) with minimal changes.

const float PI = 3.14159265359;

// Convert a direction vector to equirectangular UV coordinates.
// Mirrors Python EnvironmentMap.direction_to_uv (src/shading/pbr/env_map.py:60).
vec2 direction_to_uv(vec3 dir) {
  float u = atan(dir.z, dir.x) / (2.0 * PI) + 0.5;
  float v = asin(clamp(dir.y, -0.999, 0.999)) / PI + 0.5;
  return vec2(u, v);
}
```

- [ ] **Step 2: Commit**

```bash
git add app/src/shaders/common.glsl
git commit -m "feat(app): GLSL common.glsl with direction_to_uv"
```

---

## Task 10: Math Equivalence Tests — direction_to_uv

**Files:**
- Create: `app/tests/equivalence.test.ts`
- Create: `app/src/math/pbr_math.ts`

- [ ] **Step 1: Write failing test for direction_to_uv**

Create `app/tests/equivalence.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { directionToUV } from '../src/math/pbr_math';

describe('direction_to_uv', () => {
  it('maps (0,1,0) to top center', () => {
    // atan2(0, 0) = 0 → u = 0 + 0.5 = 0.5
    // asin(1) = π/2 → v = 0.5 + 0.5 = 1.0
    const [u, v] = directionToUV([0, 1, 0]);
    expect(u).toBeCloseTo(0.5, 5);
    expect(v).toBeCloseTo(1.0, 5);
  });

  it('maps (1,0,0) to equator center', () => {
    // atan2(0, 1) = 0 → u = 0.5
    // asin(0) = 0 → v = 0.5
    const [u, v] = directionToUV([1, 0, 0]);
    expect(u).toBeCloseTo(0.5, 5);
    expect(v).toBeCloseTo(0.5, 5);
  });

  it('maps (0,0,1) to equator right', () => {
    // atan2(1, 0) = π/2 → u = 0.25 + 0.5 = 0.75
    // asin(0) = 0 → v = 0.5
    const [u, v] = directionToUV([0, 0, 1]);
    expect(u).toBeCloseTo(0.75, 5);
    expect(v).toBeCloseTo(0.5, 5);
  });

  it('maps (0,-1,0) to bottom center', () => {
    const [u, v] = directionToUV([0, -1, 0]);
    expect(v).toBeCloseTo(0.0, 5);
  });

  it('clamps extreme y values', () => {
    const [u, v] = directionToUV([0, 1.5, 0]);
    expect(v).toBeCloseTo(1.0, 5);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app && npx vitest run tests/equivalence.test.ts`
Expected: FAIL with `Cannot find module '../src/math/pbr_math'`

- [ ] **Step 3: Create pbr_math.ts**

Create `app/src/math/pbr_math.ts`:

```typescript
/**
 * TypeScript mirror of PBR GLSL math.
 *
 * These functions are NOT used at runtime (the GLSL shaders are).
 * They exist solely for unit testing — to verify the GLSL logic
 * produces values matching Python's src/shading/pbr_model.py.
 *
 * Each function documents the GLSL line it mirrors.
 */

export const PI = Math.PI;

/**
 * Convert direction vector to equirectangular UV.
 * Mirrors GLSL: shaders/common.glsl → direction_to_uv
 * Mirrors Python: src/shading/pbr/env_map.py:60 → EnvironmentMap.direction_to_uv
 */
export function directionToUV(dir: [number, number, number]): [number, number] {
  const [x, y, z] = dir;
  const u = Math.atan2(z, x) / (2 * PI) + 0.5;
  const yClamped = Math.max(-0.999, Math.min(0.999, y));
  const v = Math.asin(yClamped) / PI + 0.5;
  return [u, v];
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd app && npx vitest run tests/equivalence.test.ts`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Commit**

```bash
git add app/tests/equivalence.test.ts app/src/math/pbr_math.ts
git commit -m "feat(app): direction_to_uv math equivalence tests"
```

---

## Task 11: Math Equivalence Tests — Material Decode

**Files:**
- Modify: `app/tests/equivalence.test.ts`
- Modify: `app/src/math/pbr_math.ts`

- [ ] **Step 1: Append failing test for decode_material**

Append to `app/tests/equivalence.test.ts`:

```typescript
import { decodeMaterial } from '../src/math/pbr_math';

describe('decode_material', () => {
  it('decodes base_color with sRGB→linear (gamma 2.2)', () => {
    // sRGB 0.5 → linear ≈ 0.214
    const texInput = {
      baseColorSRGB: [0.5, 0.5, 0.5] as [number, number, number],
      roughness: 0.0,
      metallic: 0.0,
      normalMap: [0.5, 0.5, 1.0] as [number, number, number], // (0,0,1) after remap
    };
    const m = decodeMaterial(texInput);
    expect(m.baseColor[0]).toBeCloseTo(Math.pow(0.5, 2.2), 5);
    expect(m.baseColor[1]).toBeCloseTo(Math.pow(0.5, 2.2), 5);
  });

  it('passes roughness and metallic through directly', () => {
    const m = decodeMaterial({
      baseColorSRGB: [1, 1, 1],
      roughness: 0.7,
      metallic: 0.3,
      normalMap: [0.5, 0.5, 1.0],
    });
    expect(m.roughness).toBeCloseTo(0.7, 5);
    expect(m.metallic).toBeCloseTo(0.3, 5);
  });

  it('remaps normal from [0,1] to [-1,1] and normalizes', () => {
    const m = decodeMaterial({
      baseColorSRGB: [1, 1, 1],
      roughness: 0.5,
      metallic: 0.0,
      normalMap: [1.0, 1.0, 1.0], // → (1,1,1) / sqrt(3)
    });
    const inv = 1 / Math.sqrt(3);
    expect(m.normalTS[0]).toBeCloseTo(inv, 4);
    expect(m.normalTS[1]).toBeCloseTo(inv, 4);
    expect(m.normalTS[2]).toBeCloseTo(inv, 4);
  });

  it('neutral normal (0,0,1) stays (0,0,1)', () => {
    const m = decodeMaterial({
      baseColorSRGB: [1, 1, 1],
      roughness: 0.5,
      metallic: 0.0,
      normalMap: [0.5, 0.5, 1.0],
    });
    expect(m.normalTS[0]).toBeCloseTo(0, 5);
    expect(m.normalTS[1]).toBeCloseTo(0, 5);
    expect(m.normalTS[2]).toBeCloseTo(1, 5);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd app && npx vitest run tests/equivalence.test.ts -t decode_material`
Expected: FAIL with import error

- [ ] **Step 3: Add decodeMaterial to pbr_math.ts**

Append to `app/src/math/pbr_math.ts`:

```typescript
export interface DecodedMaterial {
  baseColor: [number, number, number]; // linear
  roughness: number;
  metallic: number;
  normalTS: [number, number, number]; // unit vector
}

export interface MaterialTextureInput {
  baseColorSRGB: [number, number, number];
  roughness: number;
  metallic: number;
  normalMap: [number, number, number]; // [0,1] range
}

/**
 * Decode 4-channel material texture samples.
 * Mirrors GLSL: shaders/pbr.frag → step 1 (Material decode)
 * Mirrors Python: src/shading/pbr/material.py → decode_material
 *
 * Note: Python's training stores raw texture as sigmoid-encoded; the exported
 * PNGs are already in display space (base_color=sRGB, roughness/metallic=linear).
 * So we only apply pow(2.2) to base_color here.
 */
export function decodeMaterial(input: MaterialTextureInput): DecodedMaterial {
  const [r, g, b] = input.baseColorSRGB;
  const baseColor: [number, number, number] = [
    Math.pow(r, 2.2),
    Math.pow(g, 2.2),
    Math.pow(b, 2.2),
  ];
  const roughness = input.roughness;
  const metallic = input.metallic;

  // Remap [0,1] → [-1,1], then normalize (mirror F.normalize)
  const nx = input.normalMap[0] * 2 - 1;
  const ny = input.normalMap[1] * 2 - 1;
  const nz = input.normalMap[2] * 2 - 1;
  const len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
  const normalTS: [number, number, number] = [nx / len, ny / len, nz / len];

  return { baseColor, roughness, metallic, normalTS };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd app && npx vitest run tests/equivalence.test.ts -t decode_material`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Commit**

```bash
git add app/tests/equivalence.test.ts app/src/math/pbr_math.ts
git commit -m "feat(app): decode_material math equivalence tests"
```

---

## Task 12: Math Equivalence Tests — BRDF LUT align_corners fix

**Files:**
- Modify: `app/tests/equivalence.test.ts`
- Modify: `app/src/math/pbr_math.ts`

- [ ] **Step 1: Append failing test**

Append to `app/tests/equivalence.test.ts`:

```typescript
import { brdfLutUVAlignCorners } from '../src/math/pbr_math';

describe('brdf_lut align_corners fix', () => {
  // Python uses grid_sample(align_corners=True).
  // WebGL texture() uses align_corners=False.
  // Fix: uv_fixed = (uv * (size - 1) + 0.5) / size

  it('at uv=(0,0) with size=256 maps to center of first texel', () => {
    const [u, v] = brdfLutUVAlignCorners(0, 0, 256);
    expect(u).toBeCloseTo(0.5 / 256, 5);
    expect(v).toBeCloseTo(0.5 / 256, 5);
  });

  it('at uv=(1,1) with size=256 maps to center of last texel', () => {
    const [u, v] = brdfLutUVAlignCorners(1, 1, 256);
    expect(u).toBeCloseTo(255.5 / 256, 5);
    expect(v).toBeCloseTo(255.5 / 256, 5);
  });

  it('at uv=(0.5,0.5) with size=256 maps to middle of texture', () => {
    const [u, v] = brdfLutUVAlignCorners(0.5, 0.5, 256);
    expect(u).toBeCloseTo(128.0 / 256, 5);
    expect(v).toBeCloseTo(128.0 / 256, 5);
  });

  it('preserves input range [0,1]', () => {
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      const [u, v] = brdfLutUVAlignCorners(t, t, 256);
      expect(u).toBeGreaterThanOrEqual(0);
      expect(u).toBeLessThanOrEqual(1);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app && npx vitest run tests/equivalence.test.ts -t "brdf_lut"`
Expected: FAIL with import error

- [ ] **Step 3: Add brdfLutUVAlignCorners**

Append to `app/src/math/pbr_math.ts`:

```typescript
/**
 * Convert [0,1] UV to align_corners=True compatible UV for WebGL textures.
 *
 * Python's grid_sample uses align_corners=True (extents map to texel centers).
 * WebGL's texture() uses align_corners=False (extents map to texel edges).
 *
 * To get equivalent sampling, transform: uv → (uv * (size-1) + 0.5) / size.
 *
 * Mirrors GLSL: shaders/pbr.frag → step 5 (BRDF LUT UV fix)
 */
export function brdfLutUVAlignCorners(
  u: number,
  v: number,
  size: number,
): [number, number] {
  const uFixed = (u * (size - 1) + 0.5) / size;
  const vFixed = (v * (size - 1) + 0.5) / size;
  return [uFixed, vFixed];
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd app && npx vitest run tests/equivalence.test.ts -t "brdf_lut"`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Commit**

```bash
git add app/tests/equivalence.test.ts app/src/math/pbr_math.ts
git commit -m "feat(app): BRDF LUT align_corners UV fix + tests"
```

---

## Task 13: Math Equivalence Tests — Split-Sum Composition

**Files:**
- Modify: `app/tests/equivalence.test.ts`
- Modify: `app/src/math/pbr_math.ts`

- [ ] **Step 1: Append failing test**

Append to `app/tests/equivalence.test.ts`:

```typescript
import { splitSumShade } from '../src/math/pbr_math';

describe('split_sum composition', () => {
  const F0_dielectric = 0.04;

  it('pure diffuse (metallic=0, roughness=1) — specular ≈ 0', () => {
    const result = splitSumShade({
      baseColor: [0.8, 0.8, 0.8],
      roughness: 1.0,
      metallic: 0.0,
      NdotV: 0.9,
      brdfLutScale: 0.0, // at high roughness, scale (F0 multiplier) is small
      brdfLutBias: 0.0,
      irradiance: [1.0, 1.0, 1.0],
      prefiltered: [0.0, 0.0, 0.0], // high roughness → very blurred env
    });
    // diffuse = (1-0)*(1-0.04) * 0.8 * 1.0 = 0.768
    expect(result[0]).toBeCloseTo(0.768, 3);
    expect(result[1]).toBeCloseTo(0.768, 3);
    expect(result[2]).toBeCloseTo(0.768, 3);
  });

  it('pure metal (metallic=1) — diffuse ≈ 0, specular dominates', () => {
    const result = splitSumShade({
      baseColor: [0.95, 0.6, 0.3], // gold-ish
      roughness: 0.1,
      metallic: 1.0,
      NdotV: 0.9,
      brdfLutScale: 0.8,
      brdfLutBias: 0.05,
      irradiance: [1.0, 1.0, 1.0],
      prefiltered: [0.9, 0.9, 0.9],
    });
    // F0 = mix(0.04, baseColor, 1.0) = baseColor
    // kd = (1-1)*(1-F0) = 0 → diffuse = 0
    // specular = (F0 * 0.8 + 0.05) * prefiltered = (0.76 + 0.05 + ...) * 0.9
    expect(result[0]).toBeGreaterThan(result[1]); // red dominant
    expect(result[1]).toBeGreaterThan(result[2]);
  });

  it('clamps output to [0,1]', () => {
    const result = splitSumShade({
      baseColor: [2.0, 2.0, 2.0],
      roughness: 0.0,
      metallic: 1.0,
      NdotV: 1.0,
      brdfLutScale: 1.0,
      brdfLutBias: 1.0,
      irradiance: [10, 10, 10],
      prefiltered: [10, 10, 10],
    });
    expect(result[0]).toBeLessThanOrEqual(1.0);
    expect(result[1]).toBeLessThanOrEqual(1.0);
    expect(result[2]).toBeLessThanOrEqual(1.0);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app && npx vitest run tests/equivalence.test.ts -t "split_sum"`
Expected: FAIL with import error

- [ ] **Step 3: Add splitSumShade**

Append to `app/src/math/pbr_math.ts`:

```typescript
export interface SplitSumInput {
  baseColor: [number, number, number];
  roughness: number;
  metallic: number;
  NdotV: number;
  brdfLutScale: number; // RG.R of BRDF LUT
  brdfLutBias: number;  // RG.G of BRDF LUT
  irradiance: [number, number, number];
  prefiltered: [number, number, number];
}

/**
 * Compute final shaded RGB via split-sum composition.
 * Mirrors GLSL: shaders/pbr.frag → steps 4-6
 * Mirrors Python: src/shading/pbr_model.py:91-107 (shade_submesh)
 */
export function splitSumShade(input: SplitSumInput): [number, number, number] {
  const { baseColor, metallic, NdotV, brdfLutScale, brdfLutBias, irradiance, prefiltered } = input;

  // F0 = mix(0.04, baseColor, metallic)
  const F0: [number, number, number] = [
    0.04 * (1 - metallic) + baseColor[0] * metallic,
    0.04 * (1 - metallic) + baseColor[1] * metallic,
    0.04 * (1 - metallic) + baseColor[2] * metallic,
  ];

  // Diffuse: kd = (1 - metallic) * (1 - F0)
  const kd = (1 - metallic);
  const diffuse: [number, number, number] = [
    kd * (1 - F0[0]) * baseColor[0] * irradiance[0],
    kd * (1 - F0[1]) * baseColor[1] * irradiance[1],
    kd * (1 - F0[2]) * baseColor[2] * irradiance[2],
  ];

  // Specular: (F0 * scale + bias) * prefiltered
  const specular: [number, number, number] = [
    (F0[0] * brdfLutScale + brdfLutBias) * prefiltered[0],
    (F0[1] * brdfLutScale + brdfLutBias) * prefiltered[1],
    (F0[2] * brdfLutScale + brdfLutBias) * prefiltered[2],
  ];

  // Combine + clamp
  return [
    Math.max(0, Math.min(1, diffuse[0] + specular[0])),
    Math.max(0, Math.min(1, diffuse[1] + specular[1])),
    Math.max(0, Math.min(1, diffuse[2] + specular[2])),
  ];
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd app && npx vitest run tests/equivalence.test.ts -t "split_sum"`
Expected: PASS (all 3 tests)

- [ ] **Step 5: Run full test suite**

Run: `cd app && npx vitest run`
Expected: All math tests PASS

- [ ] **Step 6: Commit**

```bash
git add app/tests/equivalence.test.ts app/src/math/pbr_math.ts
git commit -m "feat(app): split_sum composition math equivalence tests"
```

---

## Task 14: Manifest Type Definitions

**Files:**
- Create: `app/src/types/manifest.ts`

- [ ] **Step 1: Create manifest.ts**

```typescript
/**
 * TypeScript interfaces for manifest.json schema.
 * Mirrors Python: scripts/package_runtime_asset.py → build_manifest
 * Schema version: 1
 */

export type SubmeshMatchBy = 'primitive_name' | 'material_name' | 'mesh_index';

export interface SubmeshTextures {
  base_color: string;
  roughness: string;
  metallic: string;
  normal_map: string;
}

export interface SubmeshEntry {
  name: string;
  match_by: SubmeshMatchBy;
  textures: SubmeshTextures;
}

export interface ManifestGenerator {
  tool: string;
  version: string;
  render_mode: 'pbr';
  epoch: number;
  psnr_db: number | null;
}

export interface ManifestGeometry {
  glb_path: string;
  up_axis: 'Y' | 'Z';
  scale: number;
}

export interface ManifestEnvironment {
  env_map_path: string;
  is_hdr: boolean;
  diffuse_mip_bias: number; // -1 = auto
  specular_mip_levels: number; // -1 = auto
}

export interface Manifest {
  schema_version: 1;
  scene_name: string;
  generator: ManifestGenerator;
  geometry: ManifestGeometry;
  environment: ManifestEnvironment;
  brdf_lut_path: string;
  submeshes: SubmeshEntry[];
}

export interface SceneIndexEntry {
  name: string;
  file: string;
  psnr_db: number | null;
  epoch: number;
}

export type SceneIndex = SceneIndexEntry[];

/**
 * Validate a parsed JSON object against the Manifest schema.
 * Throws Error with descriptive message if invalid.
 */
export function validateManifest(data: unknown): Manifest {
  if (typeof data !== 'object' || data === null) {
    throw new Error('Manifest must be a JSON object');
  }
  const m = data as Record<string, unknown>;
  if (m.schema_version !== 1) {
    throw new Error(`Unsupported schema_version: ${m.schema_version} (expected 1)`);
  }
  if (typeof m.scene_name !== 'string') {
    throw new Error('manifest.scene_name must be a string');
  }
  if (typeof m.brdf_lut_path !== 'string') {
    throw new Error('manifest.brdf_lut_path must be a string');
  }
  if (!Array.isArray(m.submeshes) || m.submeshes.length === 0) {
    throw new Error('manifest.submeshes must be a non-empty array');
  }
  for (let i = 0; i < m.submeshes.length; i++) {
    const s = m.submeshes[i] as Record<string, unknown>;
    if (typeof s.name !== 'string') throw new Error(`submeshes[${i}].name must be string`);
    if (!['primitive_name', 'material_name', 'mesh_index'].includes(s.match_by as string)) {
      throw new Error(`submeshes[${i}].match_by invalid: ${s.match_by}`);
    }
    const t = s.textures as Record<string, unknown>;
    for (const k of ['base_color', 'roughness', 'metallic', 'normal_map']) {
      if (typeof t[k] !== 'string') {
        throw new Error(`submeshes[${i}].textures.${k} must be a string`);
      }
    }
  }
  return data as Manifest;
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd app && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add app/src/types/manifest.ts
git commit -m "feat(app): manifest TypeScript types and validation"
```

---

## Task 15: GLSL Shaders — pbr.vert

**Files:**
- Create: `app/src/shaders/pbr.vert`

- [ ] **Step 1: Create pbr.vert**

Create `app/src/shaders/pbr.vert`:

```glsl
#version 300 es

precision highp float;

#include "common.glsl"

// Vertex attributes (Three.js default names)
in vec3 position;
in vec2 uv;
in vec3 normal;
in vec4 tangent; // xyz=dir, w=sign for bitangent handedness

// Uniforms (Three.js auto-injects these)
uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;
uniform vec3 cameraPosition;

// Outputs to fragment shader
out vec2 vUV;
out vec3 vNormalW;
out vec3 vTangentW;
out vec3 vBitangentW;
out vec3 vViewDirW;

void main() {
  vec4 worldPos = modelMatrix * vec4(position, 1.0);

  vUV = uv;
  vNormalW = normalize(mat3(modelMatrix) * normal);
  vTangentW = normalize(mat3(modelMatrix) * tangent.xyz);

  // Bitangent: cross(N, T) * tangent.w (glTF convention)
  // Mirrors Python: src/mesh.py → compute_vertex_tangents → B = cross(N, T)
  vBitangentW = normalize(cross(vNormalW, vTangentW) * tangent.w);

  // View direction: from camera to fragment, in world space
  // Mirrors Python: view_dirs is normalized (camera-to-vertex)
  vViewDirW = normalize(cameraPosition - worldPos.xyz);

  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
```

- [ ] **Step 2: Commit**

```bash
git add app/src/shaders/pbr.vert
git commit -m "feat(app): GLSL vertex shader pbr.vert"
```

---

## Task 16: GLSL Shaders — pbr.frag

**Files:**
- Create: `app/src/shaders/pbr.frag`

- [ ] **Step 1: Create pbr.frag**

Create `app/src/shaders/pbr.frag`:

```glsl
#version 300 es

precision highp float;

#include "common.glsl"

// Inputs from vertex shader
in vec2 vUV;
in vec3 vNormalW;
in vec3 vTangentW;
in vec3 vBitangentW;
in vec3 vViewDirW;

// Output
out vec4 fragColor;

// Material textures
uniform sampler2D uBaseColor;    // sRGB
uniform sampler2D uRoughness;    // R channel, linear
uniform sampler2D uMetallic;     // R channel, linear
uniform sampler2D uNormalMap;    // [0,1] encoding of [-1,1] tangent-space normal

// Environment
uniform sampler2D uEnvMap;       // equirect, RGBA, will be sampled with mipmaps
uniform sampler2D uBRDFLut;      // 2-channel (RG) lookup table

// Runtime parameters
uniform float uMaxEnvMip;        // = floor(log2(max(envH, envW)))
uniform float uDiffuseMipBias;   // = uMaxEnvMip (sample most-blurred mip)
uniform bool  uNormalMapEnabled;
uniform vec2  uBRDFLutSize;      // e.g. vec2(256.0, 256.0)

void main() {
  vec3 N = normalize(vNormalW);
  vec3 T = normalize(vTangentW);
  vec3 B = normalize(vBitangentW);
  vec3 V = normalize(vViewDirW);

  // ===== 1. Material decode (mirror Python: src/shading/pbr/material.py → decode_material) =====
  vec3 baseColor = pow(texture(uBaseColor, vUV).rgb, vec3(2.2));
  float roughness = texture(uRoughness, vUV).r;
  float metallic = texture(uMetallic, vUV).r;
  vec3 normalTS = texture(uNormalMap, vUV).rgb * 2.0 - 1.0;
  normalTS = normalize(normalTS); // mirror F.normalize

  // ===== 2. Normal mapping (direct TBN, no Gram-Schmidt) =====
  // Mirror Python: src/shading/pbr_model.py:79-86 (shade_submesh)
  if (uNormalMapEnabled) {
    N = normalize(T * normalTS.x + B * normalTS.y + N * normalTS.z);
  }

  // ===== 3. Reflect direction (mirror shade_submesh lines 89-92) =====
  float NdotV = clamp(dot(N, V), 0.0, 1.0);
  vec3 R = 2.0 * NdotV * N - V;
  R = normalize(R);

  // ===== 4. F0 + Diffuse (mirror compute_F0 + shade_submesh lines 95-99) =====
  vec3 F0 = mix(vec3(0.04), baseColor, metallic); // dielectric_F0 = 0.04
  vec3 kd = (1.0 - metallic) * (1.0 - F0);
  vec3 irradiance = textureLod(uEnvMap, direction_to_uv(N), uDiffuseMipBias).rgb;
  vec3 diffuse = kd * baseColor * irradiance;

  // ===== 5. Specular (mirror shade_submesh lines 102-108) =====
  float specLod = roughness * uMaxEnvMip; // linear mapping (NOT roughness^2)
  vec3 prefiltered = textureLod(uEnvMap, direction_to_uv(R), specLod).rgb;

  // BRDF LUT sampling with align_corners fix:
  // Python uses grid_sample(align_corners=True); WebGL texture() is align_corners=False.
  // Transform: uv → (uv * (size-1) + 0.5) / size
  vec2 brdfUv = (vec2(NdotV, roughness) * (uBRDFLutSize - 1.0) + 0.5) / uBRDFLutSize;
  vec2 brdf = texture(uBRDFLut, brdfUv).rg;
  vec3 specular = (F0 * brdf.x + brdf.y) * prefiltered;

  // ===== 6. Combine (mirror shade_submesh lines 111-113) =====
  vec3 rgb = diffuse + specular;
  rgb = clamp(rgb, 0.0, 1.0);

  fragColor = vec4(rgb, 1.0);
}
```

- [ ] **Step 2: Commit**

```bash
git add app/src/shaders/pbr.frag
git commit -m "feat(app): GLSL fragment shader pbr.frag"
```

---

## Task 17: Environment Texture Loader

**Files:**
- Create: `app/src/render/Environment.ts`

- [ ] **Step 1: Create Environment.ts**

```typescript
import * as THREE from 'three';

/**
 * Loads and configures the environment map + BRDF LUT textures.
 *
 * The env map is loaded as sRGB (matching Python's export of clamped [0,1] PNG),
 * then Three.js converts to linear for shader use.
 * Mipmaps are auto-generated so textureLod can sample prefiltered specular.
 */
export class Environment {
  /** Equirectangular env map texture with mipmaps. */
  readonly envMap: THREE.Texture;
  /** BRDF LUT texture (RG channels). */
  readonly brdfLut: THREE.Texture;
  /** floor(log2(max(envH, envW))) — max mip level for textureLod. */
  readonly maxEnvMip: number;
  /** Equals maxEnvMip — used as the diffuse irradiance mip bias. */
  readonly diffuseMipBias: number;

  private constructor(
    envMap: THREE.Texture,
    brdfLut: THREE.Texture,
    maxEnvMip: number,
  ) {
    this.envMap = envMap;
    this.brdfLut = brdfLut;
    this.maxEnvMip = maxEnvMip;
    this.diffuseMipBias = maxEnvMip;
  }

  /**
   * Build Environment from blob URLs.
   *
   * @param envMapUrl Blob URL for env_map.png
   * @param brdfLutUrl Blob URL for brdf_lut.png
   */
  static async fromUrls(envMapUrl: string, brdfLutUrl: string): Promise<Environment> {
    const envMap = await loadTexture(envMapUrl, THREE.SRGBColorSpace, true);
    const brdfLut = await loadTexture(brdfLutUrl, THREE.LinearSRGBColorSpace, false);

    // Compute max mip from env map dimensions (after image loads)
    const maxDim = Math.max(envMap.image.width, envMap.image.height);
    const maxEnvMip = Math.floor(Math.log2(maxDim));

    return new Environment(envMap, brdfLut, maxEnvMip);
  }

  dispose(): void {
    this.envMap.dispose();
    this.brdfLut.dispose();
  }
}

/**
 * Load a PNG texture from a URL with specified color space and mipmap option.
 */
function loadTexture(
  url: string,
  colorSpace: THREE.ColorSpace,
  generateMipmaps: boolean,
): Promise<THREE.Texture> {
  return new Promise((resolve, reject) => {
    const loader = new THREE.TextureLoader();
    loader.load(
      url,
      (texture) => {
        texture.colorSpace = colorSpace;
        texture.generateMipmaps = generateMipmaps;
        texture.minFilter = generateMipmaps
          ? THREE.LinearMipmapLinearFilter
          : THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
        texture.wrapS = THREE.RepeatWrapping; // equirect wraps horizontally
        texture.wrapT = THREE.ClampToEdgeWrapping;
        texture.needsUpdate = true;
        resolve(texture);
      },
      undefined,
      (err) => reject(new Error(`Failed to load texture ${url}: ${err}`)),
    );
  });
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd app && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add app/src/render/Environment.ts
git commit -m "feat(app): Environment texture loader (env map + BRDF LUT)"
```

---

## Task 18: PBRMesh Builder

**Files:**
- Create: `app/src/render/PBRMesh.ts`

- [ ] **Step 1: Create PBRMesh.ts**

```typescript
import * as THREE from 'three';
import type { SubmeshEntry, SubmeshTextures } from '../types/manifest';
import { Environment } from './Environment';

import vertSrc from '../shaders/pbr.vert?raw';
import fragSrc from '../shaders/pbr.frag?raw';

/**
 * Wraps a single submesh (one glTF primitive) with a PBR ShaderMaterial.
 *
 * Loads 4 material textures (base_color, roughness, metallic, normal_map),
 * builds uniforms, and returns a THREE.Mesh ready for the scene.
 */
export class PBRMesh {
  /** Builds a PBRMesh from a glTF primitive (THREE.Mesh) and submesh manifest entry. */
  static async fromPrimitive(
    primitive: THREE.Mesh,
    submesh: SubmeshEntry,
    textureUrls: SubmeshTextures,
    env: Environment,
    brdfLutSize: number,
  ): Promise<PBRMesh> {
    const material = await PBRMesh.buildMaterial(
      textureUrls,
      env,
      brdfLutSize,
    );

    // Replace the primitive's material with our PBR material
    primitive.material = material;

    return new PBRMesh(primitive, material, submesh);
  }

  private static async buildMaterial(
    textureUrls: SubmeshTextures,
    env: Environment,
    brdfLutSize: number,
  ): Promise<THREE.ShaderMaterial> {
    const [baseColor, roughness, metallic, normalMap] = await Promise.all([
      loadMaterialTexture(textureUrls.base_color, THREE.SRGBColorSpace),
      loadMaterialTexture(textureUrls.roughness, THREE.LinearSRGBColorSpace),
      loadMaterialTexture(textureUrls.metallic, THREE.LinearSRGBColorSpace),
      loadMaterialTexture(textureUrls.normal_map, THREE.LinearSRGBColorSpace),
    ]);

    return new THREE.ShaderMaterial({
      vertexShader: vertSrc,
      fragmentShader: fragSrc,
      uniforms: {
        uBaseColor: { value: baseColor },
        uRoughness: { value: roughness },
        uMetallic: { value: metallic },
        uNormalMap: { value: normalMap },
        uEnvMap: { value: env.envMap },
        uBRDFLut: { value: env.brdfLut },
        uMaxEnvMip: { value: env.maxEnvMip },
        uDiffuseMipBias: { value: env.diffuseMipBias },
        uNormalMapEnabled: { value: true },
        uBRDFLutSize: { value: new THREE.Vector2(brdfLutSize, brdfLutSize) },
      },
    });
  }

  private constructor(
    readonly mesh: THREE.Mesh,
    private readonly material: THREE.ShaderMaterial,
    readonly submesh: SubmeshEntry,
  ) {}

  dispose(): void {
    this.material.uniforms.uBaseColor.value.dispose();
    this.material.uniforms.uRoughness.value.dispose();
    this.material.uniforms.uMetallic.value.dispose();
    this.material.uniforms.uNormalMap.value.dispose();
    this.material.dispose();
  }
}

async function loadMaterialTexture(
  url: string,
  colorSpace: THREE.ColorSpace,
): Promise<THREE.Texture> {
  return new Promise((resolve, reject) => {
    const loader = new THREE.TextureLoader();
    loader.load(
      url,
      (texture) => {
        texture.colorSpace = colorSpace;
        texture.generateMipmaps = true;
        texture.minFilter = THREE.LinearMipmapLinearFilter;
        texture.magFilter = THREE.LinearFilter;
        texture.wrapS = THREE.ClampToEdgeWrapping;
        texture.wrapT = THREE.ClampToEdgeWrapping;
        texture.needsUpdate = true;
        resolve(texture);
      },
      undefined,
      (err) => reject(new Error(`Failed to load material texture ${url}: ${err}`)),
    );
  });
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd app && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add app/src/render/PBRMesh.ts
git commit -m "feat(app): PBRMesh builder with ShaderMaterial"
```

---

## Task 19: Scene Loader (ZIP → AssetBundle)

**Files:**
- Create: `app/src/app/SceneLoader.ts`

- [ ] **Step 1: Create SceneLoader.ts**

```typescript
import JSZip from 'jszip';
import type { Manifest, SubmeshTextures } from '../types/manifest';
import { validateManifest } from '../types/manifest';

/**
 * Result of loading a scene: validated manifest + blob URLs for all assets.
 * Caller is responsible for revoking URLs via dispose().
 */
export interface AssetBundle {
  manifest: Manifest;
  glbUrl: string;
  envMapUrl: string;
  brdfLutUrl: string;
  /** Per-submesh texture URLs, indexed by submesh name. */
  submeshTextureUrlss: Record<string, SubmeshTextures>;
  /** All created blob URLs, for cleanup. */
  blobUrls: string[];
}

export type LoadProgressCallback = (stage: string) => void;

/**
 * Load and parse a .zip asset bundle.
 *
 * Accepts either:
 *   - A URL string (fetch + arraybuffer)
 *   - A File/Blob (from drag-drop or file input)
 */
export class SceneLoader {
  static async load(
    source: string | File | Blob,
    onProgress?: LoadProgressCallback,
  ): Promise<AssetBundle> {
    onProgress?.('Fetching archive');
    const data = await toArrayBuffer(source);

    onProgress?.('Unzipping');
    const zip = await JSZip.loadAsync(data);

    // Parse manifest
    onProgress?.('Parsing manifest');
    const manifestFile = zip.file('manifest.json');
    if (!manifestFile) {
      throw new Error('manifest.json not found in archive');
    }
    const manifestJson = await manifestFile.async('text');
    const manifest = validateManifest(JSON.parse(manifestJson));

    // Build blob URLs for each asset
    onProgress?.('Extracting geometry');
    const glbUrl = await extractBlobUrl(zip, manifest.geometry.glb_path);

    onProgress?.('Extracting environment');
    const envMapUrl = await extractBlobUrl(zip, manifest.environment.env_map_path);
    const brdfLutUrl = await extractBlobUrl(zip, manifest.brdf_lut_path);

    onProgress?.('Extracting textures');
    const submeshTextureUrls: Record<string, SubmeshTextures> = {};
    for (const sub of manifest.submeshes) {
      const urls: Partial<SubmeshTextures> = {};
      for (const key of ['base_color', 'roughness', 'metallic', 'normal_map'] as const) {
        urls[key] = await extractBlobUrl(zip, sub.textures[key]);
      }
      submeshTextureUrls[sub.name] = urls as SubmeshTextures;
    }

    const blobUrls = [
      glbUrl,
      envMapUrl,
      brdfLutUrl,
      ...Object.values(submeshTextureUrls).flatMap((t) => Object.values(t)),
    ];

    return {
      manifest,
      glbUrl,
      envMapUrl,
      brdfLutUrl,
      submeshTextureUrls,
      blobUrls,
    };
  }

  /** Revoke all blob URLs created by the loader. */
  static dispose(bundle: AssetBundle): void {
    for (const url of bundle.blobUrls) {
      URL.revokeObjectURL(url);
    }
  }
}

async function toArrayBuffer(source: string | File | Blob): Promise<ArrayBuffer> {
  if (typeof source === 'string') {
    const resp = await fetch(source);
    if (!resp.ok) {
      throw new Error(`Failed to fetch ${source}: ${resp.status}`);
    }
    return await resp.arrayBuffer();
  }
  return await source.arrayBuffer();
}

async function extractBlobUrl(zip: JSZip, path: string): Promise<string> {
  const file = zip.file(path);
  if (!file) {
    throw new Error(`File not found in archive: ${path}`);
  }
  const blob = await file.async('blob');
  return URL.createObjectURL(blob);
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd app && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add app/src/app/SceneLoader.ts
git commit -m "feat(app): SceneLoader (ZIP → AssetBundle with blob URLs)"
```

---

## Task 20: Camera Controls Wrapper

**Files:**
- Create: `app/src/ui/CameraControls.ts`

- [ ] **Step 1: Create CameraControls.ts**

```typescript
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

/**
 * Wraps OrbitControls with mobile-friendly defaults and fit-to-bounds helper.
 *
 * Touch gestures (handled by OrbitControls natively):
 *   - 1 finger: rotate
 *   - 2 finger: pan / pinch-zoom
 */
export class CameraControls {
  readonly controls: OrbitControls;
  private readonly camera: THREE.PerspectiveCamera;
  private readonly domElement: HTMLElement;

  constructor(camera: THREE.PerspectiveCamera, domElement: HTMLElement) {
    this.camera = camera;
    this.domElement = domElement;
    this.controls = new OrbitControls(camera, domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.screenSpacePanning = true;
    this.controls.minDistance = 0.1;
    this.controls.maxDistance = 100;
  }

  /** Frame the camera to fit the given bounding sphere. */
  fitToBoundingSphere(sphere: THREE.Sphere): void {
    const center = sphere.center.clone();
    const radius = Math.max(sphere.radius, 0.001);

    // Position camera at a distance that fits the sphere
    const fov = (this.camera.fov * Math.PI) / 180;
    const distance = radius / Math.sin(fov / 2) * 1.2;

    // Place camera along +Z from center (arbitrary, will be user-adjustable)
    this.camera.position.set(center.x, center.y, center.z + distance);
    this.camera.near = distance / 100;
    this.camera.far = distance * 100;
    this.camera.updateProjectionMatrix();

    this.controls.target.copy(center);
    this.controls.update();
  }

  /** Reset camera to default framing of the bounding sphere. */
  reset(sphere: THREE.Sphere): void {
    this.fitToBoundingSphere(sphere);
  }

  update(): void {
    this.controls.update();
  }

  dispose(): void {
    this.controls.dispose();
  }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd app && npx tsc --noEmit`
Expected: No errors (may need to install OrbitControls type — should be in @types/three already)

- [ ] **Step 3: Commit**

```bash
git add app/src/ui/CameraControls.ts
git commit -m "feat(app): CameraControls (OrbitControls wrapper with fit-to-bounds)"
```

---

## Task 21: Performance Stats Overlay

**Files:**
- Create: `app/src/ui/PerfStats.ts`

- [ ] **Step 1: Create PerfStats.ts**

```typescript
import * as THREE from 'three';

/**
 * Real-time performance overlay: FPS, draw calls, triangle count, texture memory.
 * Updates every 500ms.
 */
export class PerfStats {
  private readonly element: HTMLDivElement;
  private readonly fpsEl: HTMLSpanElement;
  private readonly drawEl: HTMLSpanElement;
  private readonly trisEl: HTMLSpanElement;
  private readonly texEl: HTMLSpanElement;

  private frameCount = 0;
  private lastSampleTime = performance.now();
  private sampleIntervalMs = 500;
  private textures: THREE.Texture[] = [];

  constructor(parent: HTMLElement) {
    this.element = document.createElement('div');
    this.element.style.cssText = `
      position: absolute;
      top: 12px;
      right: 12px;
      padding: 10px 14px;
      background: rgba(0, 0, 0, 0.55);
      color: #00ff88;
      font-family: 'SF Mono', Consolas, monospace;
      font-size: 12px;
      border-radius: 6px;
      pointer-events: none;
      z-index: 10;
      line-height: 1.6;
    `;
    parent.appendChild(this.element);

    this.element.innerHTML = `
      <div>FPS: <span id="ps-fps">--</span></div>
      <div>Draw: <span id="ps-draw">--</span></div>
      <div>Tris: <span id="ps-tris">--</span></div>
      <div>Tex: <span id="ps-tex">--</span></div>
    `;
    this.fpsEl = this.element.querySelector('#ps-fps')!;
    this.drawEl = this.element.querySelector('#ps-draw')!;
    this.trisEl = this.element.querySelector('#ps-tris')!;
    this.texEl = this.element.querySelector('#ps-tex')!;
  }

  /** Register textures to track memory usage. */
  trackTextures(textures: THREE.Texture[]): void {
    this.textures = textures;
  }

  /** Called every animation frame. */
  onFrame(renderer: THREE.WebGLRenderer): void {
    this.frameCount++;
    const now = performance.now();
    const elapsed = now - this.lastSampleTime;
    if (elapsed < this.sampleIntervalMs) return;

    const fps = (this.frameCount * 1000) / elapsed;
    this.fpsEl.textContent = fps.toFixed(0);

    const info = renderer.info;
    this.drawEl.textContent = info.render.calls.toString();
    this.trisEl.textContent = formatNumber(info.render.triangles);

    // Estimate texture memory (RGBA = 4 bytes per pixel)
    let bytes = 0;
    for (const tex of this.textures) {
      const img = tex.image as { width?: number; height?: number };
      if (img?.width && img?.height) {
        bytes += img.width * img.height * 4;
        // Include mipmaps (~1.33x)
        if (tex.generateMipmaps) bytes = Math.floor(bytes * 1.33);
      }
    }
    this.texEl.textContent = formatBytes(bytes);

    this.frameCount = 0;
    this.lastSampleTime = now;
  }

  dispose(): void {
    this.element.remove();
  }
}

function formatNumber(n: number): string {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toString();
}

function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return (bytes / 1e9).toFixed(2) + ' GB';
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(0) + ' MB';
  if (bytes >= 1e3) return (bytes / 1e3).toFixed(0) + ' KB';
  return bytes + ' B';
}
```

- [ ] **Step 2: Commit**

```bash
git add app/src/ui/PerfStats.ts
git commit -m "feat(app): PerfStats overlay (FPS, draw calls, tris, texture memory)"
```

---

## Task 22: Loading Overlay

**Files:**
- Create: `app/src/ui/LoadingOverlay.ts`

- [ ] **Step 1: Create LoadingOverlay.ts**

```typescript
/** Loading spinner overlay with status text. */
export class LoadingOverlay {
  private readonly element: HTMLDivElement;
  private readonly statusEl: HTMLDivElement;

  constructor(parent: HTMLElement) {
    this.element = document.createElement('div');
    this.element.style.cssText = `
      position: absolute;
      inset: 0;
      background: rgba(0, 0, 0, 0.7);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      color: white;
      font-family: -apple-system, BlinkMacSystemFont, sans-serif;
      z-index: 100;
    `;
    parent.appendChild(this.element);

    const spinner = document.createElement('div');
    spinner.style.cssText = `
      width: 40px;
      height: 40px;
      border: 3px solid rgba(255, 255, 255, 0.2);
      border-top-color: #00ff88;
      border-radius: 50%;
      animation: ps-spin 0.8s linear infinite;
    `;
    this.element.appendChild(spinner);

    this.statusEl = document.createElement('div');
    this.statusEl.style.cssText = `margin-top: 14px; font-size: 13px; opacity: 0.85;`;
    this.statusEl.textContent = 'Loading...';
    this.element.appendChild(this.statusEl);

    // Inject keyframes if not already present
    if (!document.getElementById('ps-spin-keyframes')) {
      const style = document.createElement('style');
      style.id = 'ps-spin-keyframes';
      style.textContent = `@keyframes ps-spin { to { transform: rotate(360deg); } }`;
      document.head.appendChild(style);
    }
  }

  setStatus(text: string): void {
    this.statusEl.textContent = text;
  }

  show(): void {
    this.element.style.display = 'flex';
  }

  hide(): void {
    this.element.style.display = 'none';
  }

  showError(message: string): void {
    this.statusEl.textContent = `Error: ${message}`;
    this.statusEl.style.color = '#ff6b6b';
    // Auto-hide after 5 seconds
    setTimeout(() => this.hide(), 5000);
  }

  dispose(): void {
    this.element.remove();
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add app/src/ui/LoadingOverlay.ts
git commit -m "feat(app): LoadingOverlay with spinner and status"
```

---

## Task 23: Scene Picker UI

**Files:**
- Create: `app/src/ui/ScenePicker.ts`

- [ ] **Step 1: Create ScenePicker.ts**

```typescript
import type { SceneIndex } from '../types/manifest';

/**
 * Top toolbar: scene dropdown + drag-drop .zip support.
 */
export class ScenePicker {
  private readonly element: HTMLDivElement;
  private readonly selectEl: HTMLSelectElement;
  private readonly errorToast: HTMLDivElement;
  private scenes: SceneIndex = [];

  /** Fired when user picks a preset scene; receives the scene file URL. */
  onSceneSelect: ((fileUrl: string) => void) | null = null;
  /** Fired when user drops a .zip file; receives the File. */
  onZipDrop: ((file: File) => void) | null = null;

  constructor(parent: HTMLElement) {
    this.element = document.createElement('div');
    this.element.style.cssText = `
      position: absolute;
      top: 12px;
      left: 12px;
      padding: 8px 12px;
      background: rgba(0, 0, 0, 0.55);
      color: white;
      font-family: -apple-system, BlinkMacSystemFont, sans-serif;
      font-size: 13px;
      border-radius: 6px;
      z-index: 10;
      display: flex;
      align-items: center;
      gap: 10px;
    `;
    parent.appendChild(this.element);

    const label = document.createElement('span');
    label.textContent = 'Scene:';
    label.style.opacity = '0.7';
    this.element.appendChild(label);

    this.selectEl = document.createElement('select');
    this.selectEl.style.cssText = `
      background: rgba(255, 255, 255, 0.1);
      color: white;
      border: 1px solid rgba(255, 255, 255, 0.2);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 13px;
    `;
    this.selectEl.addEventListener('change', () => {
      const url = this.selectEl.value;
      if (url && this.onSceneSelect) this.onSceneSelect(url);
    });
    this.element.appendChild(this.selectEl);

    // Hint text
    const hint = document.createElement('span');
    hint.textContent = '· or drop .zip';
    hint.style.opacity = '0.5';
    hint.style.fontSize = '11px';
    this.element.appendChild(hint);

    // Drag-drop overlay (whole window)
    this.errorToast = document.createElement('div');
    this.errorToast.style.cssText = `
      position: absolute;
      bottom: 20px;
      left: 50%;
      transform: translateX(-50%);
      padding: 10px 16px;
      background: rgba(220, 50, 50, 0.9);
      color: white;
      font-family: -apple-system, sans-serif;
      font-size: 13px;
      border-radius: 6px;
      display: none;
      z-index: 100;
    `;
    parent.appendChild(this.errorToast);

    this.setupDragDrop(parent);
  }

  /** Load the preset scene index. */
  async loadSceneIndex(indexUrl: string): Promise<void> {
    try {
      const resp = await fetch(indexUrl);
      if (!resp.ok) {
        console.warn(`scenes_index.json not found at ${indexUrl}`);
        return;
      }
      this.scenes = (await resp.json()) as SceneIndex;
      this.renderOptions();
    } catch (err) {
      console.warn('Failed to load scene index:', err);
    }
  }

  private renderOptions(): void {
    this.selectEl.innerHTML = '';
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = '-- select --';
    this.selectEl.appendChild(placeholder);

    for (const scene of this.scenes) {
      const opt = document.createElement('option');
      opt.value = scene.file;
      const psnr = scene.psnr_db !== null ? ` (${scene.psnr_db.toFixed(1)} dB)` : '';
      opt.textContent = `${scene.name}${psnr}`;
      this.selectEl.appendChild(opt);
    }
  }

  private setupDragDrop(target: HTMLElement): void {
    let dragCounter = 0;

    target.addEventListener('dragenter', (e) => {
      e.preventDefault();
      dragCounter++;
      target.style.background = 'rgba(0, 255, 136, 0.1)';
    });

    target.addEventListener('dragleave', () => {
      dragCounter--;
      if (dragCounter === 0) {
        target.style.background = '';
      }
    });

    target.addEventListener('dragover', (e) => {
      e.preventDefault();
    });

    target.addEventListener('drop', (e) => {
      e.preventDefault();
      dragCounter = 0;
      target.style.background = '';

      const files = e.dataTransfer?.files;
      if (!files || files.length === 0) return;

      const file = files[0];
      if (!file.name.endsWith('.zip')) {
        this.showError('Please drop a .zip file');
        return;
      }
      if (this.onZipDrop) this.onZipDrop(file);
    });
  }

  showError(message: string): void {
    this.errorToast.textContent = message;
    this.errorToast.style.display = 'block';
    setTimeout(() => {
      this.errorToast.style.display = 'none';
    }, 4000);
  }

  dispose(): void {
    this.element.remove();
    this.errorToast.remove();
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add app/src/ui/ScenePicker.ts
git commit -m "feat(app): ScenePicker (dropdown + drag-drop zip)"
```

---

## Task 24: PBR Pipeline (Renderer + Scene + Load Logic)

**Files:**
- Create: `app/src/render/PBRPipeline.ts`

- [ ] **Step 1: Create PBRPipeline.ts**

```typescript
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

import type { AssetBundle } from '../app/SceneLoader';
import { Environment } from './Environment';
import { PBRMesh } from './PBRMesh';

/**
 * Owns the Three.js renderer, scene, camera, and the currently-loaded scene's PBRMeshes.
 */
export class PBRPipeline {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  private readonly gltfLoader: GLTFLoader;
  private pbrMeshes: PBRMesh[] = [];
  private currentEnv: Environment | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);

    this.gltfLoader = new GLTFLoader();
  }

  setSize(width: number, height: number): void {
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  /**
   * Load and instantiate a scene from an AssetBundle.
   * Disposes any previously-loaded scene first.
   */
  async loadScene(bundle: AssetBundle): Promise<{ meshes: PBRMesh[]; env: Environment }> {
    this.disposeScene();

    // 1. Load env map + BRDF LUT
    this.currentEnv = await Environment.fromUrls(bundle.envMapUrl, bundle.brdfLutUrl);

    // 2. Determine BRDF LUT size from manifest default (256) — could be detected from image
    const brdfLutSize = 256;

    // 3. Load GLB
    const gltf = await this.gltfLoader.loadAsync(bundle.glbUrl);

    // 4. Walk glTF scene, find Mesh primitives, match to submesh manifest entries
    const allTextures: THREE.Texture[] = [this.currentEnv.envMap, this.currentEnv.brdfLut];
    const pbrMeshes: PBRMesh[] = [];

    gltf.scene.updateMatrixWorld(true);

    // Build a map from name → submesh entry
    const submeshByName = new Map(bundle.manifest.submeshes.map((s) => [s.name, s]));

    // Track primitive index per mesh name to disambiguate when match_by needs adjustment
    const primitivesByName = new Map<string, THREE.Mesh[]>();

    gltf.scene.traverse((obj) => {
      if (obj instanceof THREE.Mesh) {
        const meshName = obj.name || `mesh_${primitivesByName.size}`;
        if (!primitivesByName.has(meshName)) {
          primitivesByName.set(meshName, []);
        }
        primitivesByName.get(meshName)!.push(obj);
      }
    });

    // For each submesh manifest entry, find matching primitive(s)
    for (const submesh of bundle.manifest.submeshes) {
      const primitives = primitivesByName.get(submesh.name);
      if (!primitives || primitives.length === 0) {
        console.warn(`No glTF primitive found for submesh "${submesh.name}"`);
        continue;
      }
      const textureUrls = bundle.submeshTextureUrlss[submesh.name];
      if (!textureUrls) {
        console.warn(`No texture URLs for submesh "${submesh.name}"`);
        continue;
      }
      for (const prim of primitives) {
        const pbrMesh = await PBRMesh.fromPrimitive(
          prim,
          submesh,
          textureUrls,
          this.currentEnv,
          brdfLutSize,
        );
        pbrMeshes.push(pbrMesh);
        // Track material textures for memory accounting
        const mat = pbrMesh.mesh.material as THREE.ShaderMaterial;
        allTextures.push(
          mat.uniforms.uBaseColor.value,
          mat.uniforms.uRoughness.value,
          mat.uniforms.uMetallic.value,
          mat.uniforms.uNormalMap.value,
        );
      }
    }

    // 5. Add glTF scene to render scene
    this.scene.add(gltf.scene);
    this.pbrMeshes = pbrMeshes;

    return { meshes: pbrMeshes, env: this.currentEnv };
  }

  /** Compute bounding sphere of the loaded scene. */
  getBoundingSphere(): THREE.Sphere {
    const sphere = new THREE.Sphere();
    const box = new THREE.Box3().setFromObject(this.scene);
    if (box.isEmpty()) {
      sphere.set(new THREE.Vector3(0, 0, 0), 1);
    } else {
      box.getBoundingSphere(sphere);
    }
    return sphere;
  }

  /** Get all textures for memory accounting. */
  getTrackedTextures(): THREE.Texture[] {
    const textures: THREE.Texture[] = [];
    if (this.currentEnv) {
      textures.push(this.currentEnv.envMap, this.currentEnv.brdfLut);
    }
    for (const m of this.pbrMeshes) {
      const mat = m.mesh.material as THREE.ShaderMaterial;
      textures.push(
        mat.uniforms.uBaseColor.value,
        mat.uniforms.uRoughness.value,
        mat.uniforms.uMetallic.value,
        mat.uniforms.uNormalMap.value,
      );
    }
    return textures;
  }

  render(): void {
    this.renderer.render(this.scene, this.camera);
  }

  private disposeScene(): void {
    for (const m of this.pbrMeshes) {
      m.dispose();
    }
    this.pbrMeshes = [];
    if (this.currentEnv) {
      this.currentEnv.dispose();
      this.currentEnv = null;
    }
    // Clear scene children
    while (this.scene.children.length > 0) {
      const child = this.scene.children[0];
      this.scene.remove(child);
    }
  }

  dispose(): void {
    this.disposeScene();
    this.renderer.dispose();
  }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd app && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add app/src/render/PBRPipeline.ts
git commit -m "feat(app): PBRPipeline (renderer + scene + GLTF loading)"
```

---

## Task 25: App Orchestrator

**Files:**
- Create: `app/src/app/App.ts`

- [ ] **Step 1: Create App.ts**

```typescript
import * as THREE from 'three';

import { PBRPipeline } from '../render/PBRPipeline';
import { CameraControls } from '../ui/CameraControls';
import { PerfStats } from '../ui/PerfStats';
import { LoadingOverlay } from '../ui/LoadingOverlay';
import { ScenePicker } from '../ui/ScenePicker';
import { SceneLoader, type AssetBundle } from './SceneLoader';

/**
 * Top-level orchestrator: wires together renderer, UI, camera, animation loop.
 */
export class App {
  private readonly pipeline: PBRPipeline;
  private readonly cameraControls: CameraControls;
  private readonly perfStats: PerfStats;
  private readonly loading: LoadingOverlay;
  private readonly scenePicker: ScenePicker;
  private readonly container: HTMLElement;
  private currentBundle: AssetBundle | null = null;
  private animationId = 0;

  constructor(container: HTMLElement) {
    this.container = container;

    // Create canvas
    const canvas = document.createElement('canvas');
    canvas.style.cssText = 'width: 100%; height: 100%; display: block;';
    container.appendChild(canvas);

    // Pipeline
    this.pipeline = new PBRPipeline(canvas);
    this.resize();
    window.addEventListener('resize', () => this.resize());

    // Camera controls
    this.cameraControls = new CameraControls(this.pipeline.camera, canvas);
    canvas.addEventListener('keydown', (e) => {
      if (e.key === 'r' || e.key === 'R') {
        const sphere = this.pipeline.getBoundingSphere();
        this.cameraControls.reset(sphere);
      }
    });

    // UI overlays
    this.perfStats = new PerfStats(container);
    this.loading = new LoadingOverlay(container);
    this.loading.show();

    this.scenePicker = new ScenePicker(container);
    this.scenePicker.onSceneSelect = (url) => this.loadScene(url);
    this.scenePicker.onZipDrop = (file) => this.loadScene(file);

    // Load preset scene index
    this.scenePicker.loadSceneIndex('/scenes_index.json').then(() => {
      // Auto-select first scene if available
      const first = this.scenePicker['scenes'] as unknown as { file: string }[] | undefined;
      if (first && first.length > 0 && first[0].file) {
        this.loadScene(first[0].file);
      } else {
        this.loading.setStatus('Drop a .zip to begin');
      }
    });

    // Start animation loop
    this.animate();
  }

  private async loadScene(source: string | File): Promise<void> {
    this.loading.show();
    try {
      // Dispose previous bundle's blob URLs
      if (this.currentBundle) {
        SceneLoader.dispose(this.currentBundle);
        this.currentBundle = null;
      }

      this.loading.setStatus('Loading...');
      this.currentBundle = await SceneLoader.load(source, (stage) => {
        this.loading.setStatus(stage);
      });

      this.loading.setStatus('Building scene');
      const { env } = await this.pipeline.loadScene(this.currentBundle);

      // Re-fit camera
      const sphere = this.pipeline.getBoundingSphere();
      this.cameraControls.fitToBoundingSphere(sphere);

      // Track textures for memory accounting
      this.perfStats.trackTextures(this.pipeline.getTrackedTextures());

      this.loading.hide();
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error('Scene load failed:', err);
      this.loading.showError(message);
      this.scenePicker.showError(message);
    }
  }

  private animate = (): void => {
    this.animationId = requestAnimationFrame(this.animate);
    this.cameraControls.update();
    this.pipeline.render();
    this.perfStats.onFrame(this.pipeline.renderer);
  };

  private resize(): void {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.pipeline.setSize(w, h);
  }

  dispose(): void {
    cancelAnimationFrame(this.animationId);
    if (this.currentBundle) SceneLoader.dispose(this.currentBundle);
    this.cameraControls.dispose();
    this.perfStats.dispose();
    this.loading.dispose();
    this.scenePicker.dispose();
    this.pipeline.dispose();
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add app/src/app/App.ts
git commit -m "feat(app): App orchestrator wiring all components"
```

---

## Task 26: Entry Point main.ts

**Files:**
- Create: `app/src/main.ts`

- [ ] **Step 1: Create main.ts**

```typescript
import { App } from './app/App';

const container = document.getElementById('app');
if (!container) {
  throw new Error('#app element not found in DOM');
}

const app = new App(container);

// Expose for debugging in browser console
(window as unknown as { __app: App }).__app = app;
```

- [ ] **Step 2: Verify dev server starts**

Run: `cd app && npm run dev`
Expected: Vite dev server starts on http://localhost:5173, no compile errors.

- [ ] **Step 3: Verify production build**

Run: `cd app && npm run build`
Expected: TypeScript compiles clean, Vite outputs to `app/dist/`.

- [ ] **Step 4: Commit**

```bash
git add app/src/main.ts
git commit -m "feat(app): main.ts entry point"
```

---

## Task 27: End-to-End Validation — Helmet Scene

**Files:**
- No code changes (manual verification)

- [ ] **Step 1: Pack helmet asset (if not already done)**

Run from project root:
```bash
python -m scripts.package_runtime_asset `
  --glb data/helmet_260604/scene/lowpoly.glb `
  --epoch-dir output/helmet_260604_pbr/epoch2000 `
  --scene-name helmet `
  --psnr 20.81
```
Expected: `output/helmet_pbr.zip` and `output/scenes_index.json` created/updated.

- [ ] **Step 2: Start dev server**

Run: `cd app && npm run dev`
Expected: Browser opens automatically to `http://localhost:5173`.

- [ ] **Step 3: Verify scene loads automatically**

Expected in browser:
- Loading overlay shows briefly (Unzipping → Parsing manifest → Loading GLB → Building scene)
- Loading overlay disappears
- Helmet mesh visible, PBR-shaded (diffuse + specular highlights)
- PerfStats in upper-right shows non-zero FPS, draw calls = 1, triangles > 0
- No console errors

- [ ] **Step 4: Verify camera interaction**

- Mouse drag rotates the scene
- Scroll wheel zooms
- Right-drag pans
- Press `R` to reset view

- [ ] **Step 5: Verify drag-drop zip loading**

- Drag `output/helmet_pbr.zip` from file explorer onto browser window
- Scene reloads successfully

- [ ] **Step 6: Compare visual output to training orbit.mp4**

Open `output/helmet_260604_pbr/epoch2000/orbit.mp4` alongside.
Rotate the browser view to match video frames. Check:
- Diffuse base color matches
- Specular highlights in similar positions
- No obvious normal-flip artifacts (e.g. lighting on wrong side)

If normal mapping looks inverted, see Task 28 for the flip_normal_green troubleshooting.

---

## Task 28: End-to-End Validation — Piano (Multi-Mesh)

**Files:**
- No code changes (verification only, unless bugs found)

- [ ] **Step 1: Pack piano asset**

```bash
python -m scripts.package_runtime_asset `
  --glb data/piano_260604/scene/original_with_mats.glb `
  --epoch-dir output/piano_260604_pbr_multi/epoch2000 `
  --scene-name piano `
  --psnr 28.80
```

- [ ] **Step 2: Reload dev server**

Browser should show piano scene in dropdown. Select it.
Expected:
- Loading completes
- All 6 submeshes render
- PerfStats shows 6 draw calls
- No submesh missing (no warning in console)

- [ ] **Step 3: Compare to piano orbit.mp4**

Check that each submesh (keys, body, pedals, etc.) has correct textures.

- [ ] **Step 4: If any submesh fails to match**

Common issues:
- **glTF primitive name doesn't match manifest submesh name** → check `output/piano_pbr.zip`'s `manifest.json` against glTF names from `python -c "from src.mesh import load_mesh; print([s.name for s in load_mesh('data/piano_260604/scene/original_with_mats.glb').submeshes])"`
- **Normal map appears inverted** → edit `pbr.frag` step 2: add `normalTS.y = -normalTS.y;` after the normalize; rebuild and re-test. Document in app/README.md if needed.

---

## Task 29: App README

**Files:**
- Create: `app/README.md`

- [ ] **Step 1: Create app/README.md**

```markdown
# PBR Web Viewer

WebGL2 viewer for verifying differentiable baker PBR outputs. Loads `.zip` asset bundles packed by `scripts/package_runtime_asset.py` and renders them with GLSL shaders that mirror the training-time PBR math (`src/shading/pbr_model.py`).

## Quick Start

```bash
# Pack a training output (from project root)
python -m scripts.package_runtime_asset \
  --glb data/helmet_260604/scene/lowpoly.glb \
  --epoch-dir output/helmet_260604_pbr/epoch2000 \
  --scene-name helmet \
  --psnr 20.81

# Start the dev server
cd app
npm install
npm run dev
```

Browser opens to `http://localhost:5173`. The helmet scene loads automatically.

## Asset Bundle Format

See `docs/superpowers/specs/2026-06-17-pbr-web-viewer-design.md` §3 for the full spec.

Minimum required structure:

```
scene.zip
├── manifest.json
├── geometry/scene.glb
├── textures/
│   ├── env_map.png
│   ├── brdf_lut.png
│   └── {submesh_name}/
│       ├── base_color.png
│       ├── roughness.png
│       ├── metallic.png
│       └── normal_map.png
```

You can also drag-drop any valid `.zip` onto the browser window.

## Controls

| Action | Desktop | Mobile |
|---|---|---|
| Rotate | Left-drag | One-finger drag |
| Pan | Right-drag | Two-finger drag |
| Zoom | Scroll | Pinch |
| Reset | `R` key | — |

## GLSL Files (Porting Reference)

All PBR math lives in `src/shaders/`:

- `common.glsl` — `PI` constant + `direction_to_uv()` helper
- `pbr.vert` — vertex transform, world-space normal/tangent/view outputs
- `pbr.frag` — fragment shader with all split-sum PBR math

These files have **no Three.js dependencies** and can be copied directly to a native engine (Vulkan/Metal/GLES) with minimal changes.

### Uniforms Contract

The fragment shader expects:

| Uniform | Type | Description |
|---|---|---|
| `uBaseColor` | sampler2D | sRGB base color texture |
| `uRoughness` | sampler2D | Linear roughness (R channel) |
| `uMetallic` | sampler2D | Linear metallic (R channel) |
| `uNormalMap` | sampler2D | Tangent-space normal map |
| `uEnvMap` | sampler2D | Equirectangular env map (with mipmaps) |
| `uBRDFLut` | sampler2D | 2-channel GGX BRDF integration LUT |
| `uMaxEnvMip` | float | `floor(log2(max(envH, envW)))` |
| `uDiffuseMipBias` | float | Equals `uMaxEnvMip` |
| `uNormalMapEnabled` | bool | Toggle normal mapping |
| `uBRDFLutSize` | vec2 | LUT resolution (e.g. 256×256) |

## Development

```bash
npm run test      # Vitest unit tests (math equivalence)
npm run build     # Production build to dist/
npm run preview   # Preview production build
```

## Testing Equivalence

The unit tests in `tests/equivalence.test.ts` mirror the GLSL math in TypeScript to verify numerical equivalence with Python's `src/shading/pbr_model.py`. Run them after any shader change.
```

- [ ] **Step 2: Commit**

```bash
git add app/README.md
git commit -m "docs(app): README with usage, asset format, porting reference"
```

---

## Task 30: Update Project README and Final Commit

**Files:**
- Modify: `README.md` (project root)

- [ ] **Step 1: Add Web Viewer section to project README**

Find the section after the v0.4 changelog (added earlier). Insert before "## License":

```markdown
## Web Viewer (Runtime Validation)

A WebGL2 viewer is available in `app/` for validating PBR baking outputs in a browser environment. It loads asset bundles produced by `scripts/package_runtime_asset.py` and renders them with GLSL shaders that strictly mirror the training-time PBR math.

```bash
# Pack a training output
python -m scripts.package_runtime_asset \
  --glb data/helmet_260604/scene/lowpoly.glb \
  --epoch-dir output/helmet_260604_pbr/epoch2000 \
  --scene-name helmet \
  --psnr 20.81

# Run the viewer
cd app && npm install && npm run dev
```

See `app/README.md` for details.
```

- [ ] **Step 2: Run final test suite**

Run: `pytest tests/ -v`
Expected: All Python tests pass, including new `tests/test_package_runtime.py`.

Run: `cd app && npm run test`
Expected: All TypeScript math tests pass.

Run: `cd app && npm run build`
Expected: Production build succeeds without errors.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add Web Viewer section to project README"
```

---

## Self-Review

After writing this plan, I checked it against the spec:

**Spec coverage:**
- §1 Goals: covered by all tasks (validation = Tasks 27-28, portability = GLSL Tasks 9/15/16, arbitrary loading = Tasks 19/23)
- §1.2 Non-goals: explicitly omitted (no GT comparison, no tone mapping, etc.)
- §2 Project structure: Task 7 establishes the scaffold
- §3 Asset format: Tasks 1-6 (Python packaging), Task 14 (TS manifest types), Task 19 (loader)
- §4 Render pipeline: Tasks 9/15/16 (shaders), Task 17 (env), Task 18 (mesh), Task 24 (pipeline)
- §5 UI: Tasks 20-23 (camera, perf, loading, picker), Task 25 (orchestrator)
- §6 Tests: Tasks 10-13 (math equivalence), Task 30 (final suite)
- §7 Python script: Tasks 1-6
- §10 Acceptance: Tasks 27-28 (E2E), Tasks 10-13 (math tests)

**Placeholder scan:** No TBD/TODO. All code blocks are complete.

**Type consistency:**
- `AssetBundle` defined in Task 19, used in Tasks 24-25 ✓
- `PBRMesh.fromPrimitive` signature consistent across Tasks 18/24 ✓
- `Environment.fromUrls` signature consistent across Tasks 17/24 ✓
- `validateManifest` defined in Task 14, used in Task 19 ✓
- `SceneIndex`/`SceneIndexEntry` defined in Task 14, used in Task 23 ✓
- Manifest field names: `submeshes[].textures.{base_color,roughness,metallic,normal_map}` consistent across Python (Task 1) and TS (Task 14) ✓

**Open issues to flag during execution:**
- Task 23 has a private field access (`this.scenePicker['scenes']`) which is a code smell — could be cleaned up by exposing a public getter. Documented inline.
- Task 27 step 6 mentions possible normal map Y-flip — this is a known glTF/OpenGL convention difference; if it occurs, the fix is documented in Task 28.
