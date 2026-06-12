"""glTF loader — pygltflib-based, extracts per-mesh geometry + material refs."""
from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import numpy as np
from pygltflib import GLTF2


def _extract_image(gltf, image_idx) -> np.ndarray | None:
    """Extract an image from glTF as RGB numpy array [H, W, 3], float32 [0,1].

    Returns None if extraction fails.
    """
    from PIL import Image as PILImage

    img_info = gltf.images[image_idx]

    # Try embedded bufferView
    if img_info.bufferView is not None:
        bv = gltf.bufferViews[img_info.bufferView]
        bin_data = gltf.binary_blob() or b""
        start = bv.byteOffset or 0
        raw = bin_data[start:start + bv.byteLength]
        pil = PILImage.open(io.BytesIO(raw))
    elif img_info.uri is not None:
        # External file reference
        pil = PILImage.open(img_info.uri)
    else:
        return None

    pil = pil.convert("RGB")
    arr = np.array(pil, dtype=np.float32) / 255.0
    return arr


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
    gltf = GLTF2.load(str(path))

    # Load binary data (GLB binary blob)
    bin_data = gltf.binary_blob() or b""

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
                    faces = accessor_cache[prim.indices].astype(np.int64).reshape(-1, 3)
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

                # Material name + normal map
                mat_name = None
                normal_map_image = None
                if prim.material is not None:
                    mat = gltf.materials[prim.material]
                    mat_name = mat.name or f"material_{prim.material}"
                    if mat.normalTexture is not None:
                        nt_idx = mat.normalTexture.index
                        tex_info = gltf.textures[nt_idx]
                        try:
                            normal_map_image = _extract_image(gltf, tex_info.source)
                        except Exception:
                            normal_map_image = None

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
                    "normal_map_image": normal_map_image,
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
                    faces = accessor_cache[prim.indices].astype(np.int64).reshape(-1, 3)
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
                normal_map_image = None
                if prim.material is not None:
                    mat = gltf.materials[prim.material]
                    mat_name = mat.name or f"material_{prim.material}"
                    if mat.normalTexture is not None:
                        nt_idx = mat.normalTexture.index
                        tex_info = gltf.textures[nt_idx]
                        try:
                            normal_map_image = _extract_image(gltf, tex_info.source)
                        except Exception:
                            normal_map_image = None

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
                    "normal_map_image": normal_map_image,
                })

    # UV V-axis fix: glTF V=0 at bottom → V=1 at top (nvdiffrast convention)
    for sub in submeshes:
        if sub["uvs"].shape[0] > 0:
            sub["uvs"][:, 1] = sub["uvs"][:, 1] % 1.0

    return submeshes
