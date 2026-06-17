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

from src.mesh import load_mesh, MultiMeshData


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
        "submeshes": [
            {**sm, "match_by": sm.get("match_by", "primitive_name")}
            for sm in submeshes
        ],
    }


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
    if hasattr(mesh, "submeshes"):
        return [s.name for s in mesh.submeshes]
    # Single MeshData — use the mesh's name or fall back to "mesh_0"
    return [getattr(mesh, "name", None) or "mesh_0"]


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
