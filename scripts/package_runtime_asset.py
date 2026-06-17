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
        "submeshes": [
            {**sm, "match_by": sm.get("match_by", "primitive_name")}
            for sm in submeshes
        ],
    }
