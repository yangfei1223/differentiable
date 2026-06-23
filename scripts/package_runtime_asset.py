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
import io
import json
import os
import zipfile
from pathlib import Path

from src.mesh import load_mesh


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
    material_textures_flip_y: bool = True,
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
        material_textures_flip_y: Whether to flip material textures on Y axis
            during GPU upload. True (default) for GLBs authored with V=0 at
            bottom (OpenGL); False for GLBs authored with V=0 at top
            (image-data, e.g., Sketchfab exports).

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
        "material_textures_flip_y": bool(material_textures_flip_y),
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
      2. Multi-mesh: textures in epoch_dir/<glb_name>/ subdirs
         → 1 submesh per GLB submesh name, matched by NAME not position.

    Args:
        epoch_dir: Directory containing exported PBR textures.
        scene_name: Scene name (used for single-mesh case).
        glb_submesh_names: Ordered list of submesh names extracted from GLB.

    Returns:
        List of submesh manifest entries.

    Raises:
        FileNotFoundError: If a required texture is missing.
        ValueError: If a GLB submesh name has no matching filesystem directory.
    """
    # Detect multi-mesh: any Object_N subdirectory exists
    sub_dirs = sorted([d for d in epoch_dir.iterdir() if d.is_dir() and d.name.startswith("Object_")])

    if not sub_dirs:
        # Single-mesh layout
        name = glb_submesh_names[0] if glb_submesh_names else scene_name
        return [_build_submesh_entry(name, epoch_dir, textures_prefix=f"textures/{name}")]

    # Multi-mesh layout: match by NAME, not by position
    entries = []
    missing_dirs = []
    for glb_name in glb_submesh_names:
        sub_dir = epoch_dir / glb_name
        if not sub_dir.exists():
            missing_dirs.append(glb_name)
            continue
        entries.append(
            _build_submesh_entry(glb_name, sub_dir, textures_prefix=f"textures/{glb_name}")
        )

    if missing_dirs:
        actual_dirs = sorted([d.name for d in epoch_dir.iterdir() if d.is_dir()])
        raise ValueError(
            f"GLB submesh names {missing_dirs} not found in epoch_dir. "
            f"Available subdirs: {actual_dirs}. "
            f"GLB names must match training-time submesh names (which become filesystem dir names)."
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


def detect_material_textures_flip_y(glb_path: str) -> bool:
    """Detect whether material textures should be flipped on Y during GPU upload.

    Strategy: inspect TEXCOORD_0 V range across all primitives.
    - If V is mostly in [0, 1]: GLB was authored with V=0 at top (image-data
      convention, common for Sketchfab exports like the piano). Web Viewer
      needs flipY=False to preserve nvdiffrast's raw texture layout.
    - If V is mostly in [1, 2] (or otherwise offset by ~1): GLB was authored
      with V=0 at bottom (OpenGL convention, e.g., KHR DAMAGED_HELMET). Web
      Viewer needs flipY=True (Three.js default).

    Args:
        glb_path: Path to .glb file.

    Returns:
        True if flipY should be enabled, False otherwise.
    """
    import json as json_mod
    import numpy as np

    with open(glb_path, "rb") as f:
        f.read(12)  # magic/version/length
        chunk_length = int.from_bytes(f.read(4), "little")
        f.read(4)  # chunk type
        json_data = json_mod.loads(f.read(chunk_length).decode("utf-8"))
        bin_len = int.from_bytes(f.read(4), "little")
        f.read(4)
        bin_data = f.read(bin_len)

    accessors = json_data.get("accessors", [])
    buffer_views = json_data.get("bufferViews", [])

    # Collect V values from TEXCOORD_0 of every primitive
    all_v_values: list[float] = []
    for mesh in json_data.get("meshes", []):
        for prim in mesh.get("primitives", []):
            texcoord_0 = prim.get("attributes", {}).get("TEXCOORD_0")
            if texcoord_0 is None:
                continue
            acc = accessors[texcoord_0]
            bv = buffer_views[acc["bufferView"]]
            offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
            count = acc["count"]
            arr = np.frombuffer(bin_data, dtype=np.float32, count=count * 2, offset=offset).reshape(count, 2)
            all_v_values.extend(arr[:, 1].tolist())

    if not all_v_values:
        # No UVs — default to OpenGL convention
        return True

    v_arr = np.array(all_v_values)
    v_mean = float(v_arr.mean())
    # Heuristic: if mean V is in [0.5, 1.5], it's [0,1]-style → flipY=False.
    # If mean V is in [1.5, 2.5], it's [1,2]-style (OpenGL offset) → flipY=True.
    flip_y = v_mean >= 1.5
    print(f"  [INFO] UV V mean={v_mean:.3f} min={v_arr.min():.3f} max={v_arr.max():.3f} → flipY={flip_y}")
    return flip_y


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


def _export_brdf_lut_data_png(brdf_lut_pt_path: Path) -> bytes:
    """Convert brdf_lut.pt [256, 256, 2] float → 256×256 PNG bytes.

    Channel mapping: R=scale, G=bias, B=0 (unused). 8-bit quantization
    of [0,1] floats (sufficient precision for BRDF integration LUT).

    Args:
        brdf_lut_pt_path: Path to brdf_lut.pt file.

    Returns:
        PNG file bytes (256×256 RGB).
    """
    import io
    import torch
    from PIL import Image
    import numpy as np

    lut = torch.load(brdf_lut_pt_path, map_location='cpu')  # [256, 256, 2]
    if lut.dim() == 3 and lut.shape[-1] == 2:
        # Expected HWC layout
        scale = lut[:, :, 0].numpy()
        bias = lut[:, :, 1].numpy()
    elif lut.dim() == 3 and lut.shape[0] == 2:
        # Alternative CHW layout
        scale = lut[0].numpy()
        bias = lut[1].numpy()
    else:
        raise ValueError(f"Unexpected BRDF LUT shape: {lut.shape}")

    size = scale.shape[0]  # 256
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[:, :, 0] = (np.clip(scale, 0, 1) * 255).astype(np.uint8)
    rgb[:, :, 1] = (np.clip(bias, 0, 1) * 255).astype(np.uint8)
    # B channel = 0 (unused by shader)

    buf = io.BytesIO()
    Image.fromarray(rgb, 'RGB').save(buf, format='PNG')
    return buf.getvalue()


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

    # Validate env_map: prefer .hdr (preserves training-time HDR values),
    # fall back to .png (lossy clamp to [0,1]).
    env_map_hdr_file = epoch_dir / "env_map.hdr"
    env_map_png_file = epoch_dir / "env_map.png"

    env_map_zip_path: str
    env_map_is_hdr: bool
    env_map_bytes: bytes | None = None  # only set when we generate on-the-fly

    if env_map_hdr_file.exists():
        env_map_zip_path = "textures/env_map.hdr"
        env_map_is_hdr = True
        print(f"  [INFO] Using HDR env_map: {env_map_hdr_file}")
    else:
        # Try to generate .hdr from checkpoint on-the-fly
        ckpt_file = epoch_dir / "pbr_checkpoint.pt"
        if ckpt_file.exists():
            try:
                import torch
                import torch.nn.functional as F
                from src.shading.pbr.hdr_writer import write_hdr_from_tensor

                state = torch.load(ckpt_file, map_location="cpu", weights_only=False)
                if "env_map" in state:
                    raw = state["env_map"]
                    decoded = F.softplus(raw)
                    buf = io.BytesIO()
                    # write_hdr_from_tensor writes to file path; use temp file
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tf:
                        temp_path = tf.name
                    write_hdr_from_tensor(temp_path, decoded)
                    with open(temp_path, "rb") as f:
                        env_map_bytes = f.read()
                    os.unlink(temp_path)
                    env_map_zip_path = "textures/env_map.hdr"
                    env_map_is_hdr = True
                    # Also persist alongside epoch_dir for reuse
                    try:
                        write_hdr_from_tensor(str(env_map_hdr_file), decoded)
                        print(f"  [INFO] Generated HDR env_map from checkpoint: {env_map_hdr_file}")
                    except Exception:
                        pass
                else:
                    raise KeyError("checkpoint has no 'env_map' key")
            except Exception as e:
                print(f"  [WARN] Could not generate HDR env_map from checkpoint: {e}")
                env_map_bytes = None

        if env_map_bytes is None:
            # Fall back to PNG
            if not env_map_png_file.exists():
                raise FileNotFoundError(
                    f"Missing env_map.png and env_map.hdr (and no usable checkpoint): {epoch_dir}"
                )
            env_map_zip_path = "textures/env_map.png"
            env_map_is_hdr = False
            print(f"  [WARN] Using LDR env_map.png (lossy clamp — colors may shift)")

    # BRDF LUT: prefer brdf_lut.png (already data format from new logger);
    # fall back to .pt regeneration for old epoch dirs.
    brdf_lut_pt_file = epoch_dir / "brdf_lut.pt"
    brdf_lut_png_file = epoch_dir / "brdf_lut.png"
    brdf_lut_data_png: bytes | None = None
    if brdf_lut_png_file.exists():
        # Check if PNG is already the engine-friendly format (256x256).
        # Old debug visualization is 512x256; trigger .pt fallback in that case.
        from PIL import Image
        try:
            with Image.open(brdf_lut_png_file) as _img:
                _w, _h = _img.size
            if _w == 256 and _h == 256:
                # New data format — use as-is
                pass
            elif brdf_lut_pt_file.exists():
                brdf_lut_data_png = _export_brdf_lut_data_png(brdf_lut_pt_file)
                print(f"  [INFO] brdf_lut.png is old debug format ({_w}x{_h}); regenerated from .pt")
            else:
                print(f"  [WARN] brdf_lut.png is debug format ({_w}x{_h}) and no .pt to regenerate from")
        except Exception as e:
            print(f"  [WARN] Failed to inspect brdf_lut.png: {e}; using as-is")
    elif brdf_lut_pt_file.exists():
        brdf_lut_data_png = _export_brdf_lut_data_png(brdf_lut_pt_file)
        print(f"  [INFO] brdf_lut.png missing; generated from {brdf_lut_pt_file.name}")
    else:
        raise FileNotFoundError(f"Missing both brdf_lut.pt and brdf_lut.png in {epoch_dir}")

    # Detect material texture flipY from GLB UV layout
    material_textures_flip_y = detect_material_textures_flip_y(glb_path)

    # Build manifest
    manifest = build_manifest(
        scene_name=scene_name,
        glb_path="geometry/scene.glb",
        submeshes=submeshes,
        env_map_path=env_map_zip_path,
        brdf_lut_path="textures/brdf_lut.png",
        epoch=epoch,
        psnr_db=psnr_db,
        is_hdr=env_map_is_hdr,
        material_textures_flip_y=material_textures_flip_y,
    )

    # Build the zip
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Manifest
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        # Geometry
        zf.write(glb_path, "geometry/scene.glb")

        # Env map: HDR bytes (generated) or source file (.hdr or .png)
        if env_map_bytes is not None:
            zf.writestr(env_map_zip_path, env_map_bytes)
        elif env_map_is_hdr:
            zf.write(env_map_hdr_file, env_map_zip_path)
        else:
            zf.write(env_map_png_file, env_map_zip_path)

        # BRDF LUT: use regenerated data PNG if available, else fallback
        if brdf_lut_data_png is not None:
            zf.writestr("textures/brdf_lut.png", brdf_lut_data_png)
        else:
            zf.write(brdf_lut_png_file, "textures/brdf_lut.png")

        # Per-submesh textures
        # Detect layout: any subdirectory matching a submesh name → multi-mesh
        sub_dirs_exist = any((epoch_dir / sub["name"]).is_dir() for sub in submeshes)

        for sub in submeshes:
            sub_name = sub["name"]
            if sub_dirs_exist:
                src_dir = epoch_dir / sub_name  # name-based lookup
            else:
                src_dir = epoch_dir  # single-mesh fallback (flat layout)

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
    parser.add_argument("--output", default=None, help="Output .zip path (default: export/scenes/{scene}_pbr.zip)")
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

    # Default output path — subdir scenes/ matches Vite publicDir URL convention
    output_path = args.output or f"export/scenes/{args.scene_name}_pbr.zip"

    # Ensure the scenes/ subdirectory exists (package_asset creates parent, but index_path
    # lives at output root; we create scenes/ here for clarity)
    Path("export/scenes").mkdir(parents=True, exist_ok=True)

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

    # scenes_index.json lives at output root (not in scenes/ subdir)
    # so Vite serves it at /scenes_index.json (root URL, matching App.ts fetch)
    index_path = Path("export") / "scenes_index.json"
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
