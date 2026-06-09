# Multi-Submesh PBR Baking — Design Spec

## Problem

Current pipeline loads the entire scene as a single merged mesh with one shared texture. For complex models like the piano (6 submeshes, 62–47831 faces), small submeshes get insufficient texture resolution on the shared atlas, losing detail.

## Goal

Support per-submesh independent textures while keeping the single-mesh pipeline fully backward compatible.

**Key change**: Multi-mesh mode uses the **original high-poly model** (same geometry as GT rendering), not the decimated low-poly. This ensures geometry is perfectly aligned with GT, eliminating mesh approximation as a source of error.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Texture resolution | Uniform for all submeshes | Simple, sufficient for ≤10 submeshes |
| UV handling | Preserve original glTF UVs | All submeshes already have UVs; no remapping needed |
| Rendering | Per-submesh serial rasterize + composite | N=6 submeshes, negligible overhead |
| GLTF loading | `pygltflib` | Precise glTF 2.0 spec, direct access to mesh/material/texture references |
| Geometry source | Original high-poly (same as GT) | Multi-mesh mode uses `original_with_mats.glb`, geometry fully aligned with GT rendering |
| Backward compat | Single-mesh detected → single-texture path unchanged | Zero regression risk |

## Architecture

### 1. Data Layer — `MultiMeshData`

```
src/mesh.py (extended)
├── MeshData          # existing, unchanged
├── SubMeshData       # new: one submesh
│   ├── name: str
│   ├── vertices, faces, uvs, uv_idx, normals, normal_idx, tangents, bitangents
│   └── material_name: str | None
├── MultiMeshData     # new: collection of SubMeshData
│   ├── submeshes: list[SubMeshData]
│   └── num_submeshes: int
└── load_mesh(path) → MeshData | MultiMeshData
    ├── .obj / single-mesh .glb → MeshData (existing path)
    └── multi-mesh .glb → MultiMeshData (new path)
```

**pygltflib loading** (`src/gltf_loader.py`, new file):

- Parse glTF binary data directly via `pygltflib`
- Extract per-mesh: positions, normals, texcoords, indices, material reference
- Apply scene graph transforms (node hierarchy)
- Compute tangents/bitangents per submesh (reuse existing `MeshData.compute_vertex_tangents`)
- UV V-axis fix (glTF V=0 at bottom → V=1 at top) handled per submesh

### 2. Renderer Layer

No new renderer class. The existing `DifferentiableRenderer` is instantiated once per submesh. The trainer orchestrates the loop:

```
for submesh in multi_mesh.submeshes:
    rast, texc, ... = renderers[submesh.name].rasterize_and_interpolate(camera)
    rgb_sub, mask_sub = model.shade_submesh(submesh.name, rast, texc, ...)
    composite rgb_sub into full frame (z-buffer or additive)
```

**Composite strategy**: Each submesh renders independently. Results are composited using the rasterizer's own z-buffer — later submeshes overwrite earlier ones where they occlude. Since all submeshes share the same camera, this is equivalent to rendering them as a single scene. In practice, simply summing `rgb * mask` and tracking the occupied mask works for non-overlapping meshes. For overlapping meshes, use per-pixel argmax of rast z-channel.

### 3. Shading Layer — `PBRShadingModel` extension

```
PBRShadingModel
├── mat_texture: nn.Parameter              # single mesh (existing)
├── mat_textures: dict[str, nn.Parameter]  # multi mesh (new)
├── env_map: EnvironmentMap                # shared across all submeshes
├── is_multi: bool
│
├── init_textures(resolution, submesh_names=None)
│   ├── submesh_names is None → single texture (existing behavior)
│   └── submesh_names given → one texture per name
│
├── shade_submesh(name, rast, texc, ...) → rgb, mask
│   └── Same as shade() but uses mat_textures[name]
│
├── shade(rast, texc, ...) → rgb, mask
│   └── Existing single-texture path (unchanged)
│
├── parameters() → list[nn.Parameter]
│   └── Returns all mat_textures values + env_map.raw
│
├── state_dict() / load_state_dict()
│   └── Serialized as {mat_textures: {name: tensor}, env_map: tensor}
│
└── export(output_dir)
    └── One subfolder per submesh with material maps
```

**Backward compat**: When `is_multi == False`, all methods delegate to the existing single-texture fields. No behavioral change.

### 4. Trainer Layer — `Trainer` adaptation

```
Trainer.__init__(config)
├── mesh = load_mesh(...)
├── if isinstance(mesh, MultiMeshData):
│   ├── Create one DifferentiableRenderer per submesh
│   ├── model.init_textures(res, submesh_names=[s.name for s in mesh.submeshes])
│   └── self.is_multi = True
├── else:
│   ├── Existing single-mesh path (unchanged)
│   └── self.is_multi = False
```

**Training loop** (PBR multi-mesh path):

```python
for idx in batch:
    img_np, camera = dataset[idx]
    
    # Render all submeshes
    rendered_total = zeros(1, H, W, 3)
    mask_total = zeros(1, H, W)
    
    for submesh in multi_mesh.submeshes:
        rast, texc, wpos, inorm, vdir, tang, btang = \
            renderers[submesh.name].rasterize_and_interpolate(camera)
        rgb_sub, mask_sub = model.shade_submesh(
            submesh.name, rast, texc, wpos, inorm, vdir, camera, res, tang, btang)
        rendered_total += rgb_sub
        mask_total = max(mask_total, mask_sub)
    
    # Loss / backward (same as single mesh)
    loss = criterion(rendered_total, gt_linear, mask_total, ...)
    loss.backward()
```

**TV loss / seam padding**: Applied per-submesh texture independently. Env map regularization unchanged.

**PSNR**: Computed on the composite frame, same as single mesh.

### 5. Config

New optional field in `DataConfig`:

```yaml
data:
  mesh_path: data/piano_260604/scene/original_with_mats.glb  # high-poly, same as GT
  # For single-mesh backward compat: data/helmet_260604/scene/lowpoly.glb still works
```

Auto-detection: `load_mesh()` inspects the glTF. If it contains multiple mesh primitives, return `MultiMeshData`. Single mesh → `MeshData`. Users point `mesh_path` to the original high-poly model for multi-mesh mode.

### 6. Export

- Single mesh: existing behavior (one set of material maps)
- Multi mesh: one subfolder per submesh

```
output/
└── {dataset}/
    └── epoch2000/
        ├── submesh_0/
        │   ├── base_color.png
        │   ├── roughness.png
        │   ├── metallic.png
        │   └── normal_map.png
        ├── submesh_1/
        │   └── ...
        ├── env_map.png
        └── curves.png
```

## Backward Compatibility

| Scenario | Behavior |
|----------|----------|
| Single mesh GLB (helmet, lowpoly) | `load_mesh()` returns `MeshData`, all downstream unchanged |
| Multi mesh GLB (piano, original high-poly) | `load_mesh()` returns `MultiMeshData`, trainer uses multi path, geometry = GT geometry |
| OBJ files | Always single mesh, returns `MeshData` |
| Existing configs | No changes required, multi-mesh detected automatically |
| Existing tests | All pass unchanged, new tests for multi-mesh |

## Dependencies

- `pygltflib` — new dependency for glTF loading
- `trimesh` — retained for OBJ loading and fallback
- No changes to nvdiffrast, PyTorch, or other existing deps

## Files Changed

| File | Change |
|------|--------|
| `src/gltf_loader.py` | **New** — pygltflib-based glTF loader |
| `src/mesh.py` | **Modified** — add `SubMeshData`, `MultiMeshData`, extend `load_mesh()` |
| `src/shading/pbr_model.py` | **Modified** — add multi-texture support |
| `src/trainer.py` | **Modified** — add multi-mesh training loop |
| `src/shading/pbr_logger.py` | **Modified** — per-submesh debug export |
| `src/video.py` | **Modified** — multi-mesh video rendering |
| `src/config.py` | **Minimal** — no new fields needed |
| `tests/` | **New** — tests for multi-mesh loading and rendering |

## Out of Scope

- Per-submesh resolution tuning (future work)
- Texture atlas packing (future work)
- UV re-optimization for submeshes
- SH multi-mesh support (PBR only for now)
