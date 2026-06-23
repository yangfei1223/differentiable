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

### AB Pixel Comparison

For end-to-end validation against the training pipeline's reference render:

1. Pack an asset from a training output:
   ```bash
   python -m scripts.package_runtime_asset \
     --glb data/helmet_260604/scene/lowpoly.glb \
     --epoch-dir output/helmet_no_normal/epoch2000 \
     --scene-name helmet
   ```
2. Run the dev server, load with a camera hash from `data/helmet_260604/cameras.json`:
   ```
   http://localhost:5173/#cam=px,py,pz,tx,ty,tz,ux,uy,uz,fov
   ```
3. Capture a 1024×1024 render via the `?render=1024` query (auto-downloads PNG), or use `window.__pipeline` in the console to drive `setSize` + `render` + `readPixels` directly.
4. Compare against the top-right ("Rendered") panel of `output/{scene}_no_normal/epoch2000/compare_NNNN.png`:
   ```bash
   python scripts/ab_compare.py helmet
   ```

GT panel layout in `compare_*.png` (from `src/shading/pbr_logger.py`):

```
┌───────────┬───────────┐
│   GT      │ Rendered  │   ← top row
├───────────┼───────────┤
│ Diffuse   │ Specular  │   ← bottom row
└───────────┴───────────┘
```

Each panel is gamma-2.2 encoded (`pbr_logger.py:162`).

## Helmet Validation Results (2026-06-23)

**Data source**: `output/helmet_no_normal/epoch2000` (single-mesh, `disable_normal_map=True`).

**Shader setting**: `uNormalMapEnabled = false` in `PBRMesh.ts`. The Python training pipeline skips normal mapping in its compare/video export paths (`pbr_logger._export_compare` calls `model.shade()` without tangents, and `pbr_model.py:78` requires both `not disable_normal_map` AND non-null tangents), so the Web viewer must match that behavior regardless of what the `normal_map.png` file contains.

**AB stats at camera[50]** (compare_0001, viewport ~56% foreground):

| Metric | Value |
|---|---|
| PSNR (overlap foreground) | **17.47 dB** |
| RMSE | 34.10 |
| Web fg-mean RGB | [146, 143, 129] |
| GT fg-mean RGB | [148, 146, 134] |
| Mean diff (Web − GT) | [−1, −2, −5] |
| Edge density (Sobel >50, fg) | Web 19.58% vs GT 19.61% |

**Interpretation**: per-pixel difference is large (low PSNR), but aggregate statistics are nearly identical — overall color, brightness, foreground coverage, and edge density all match within 1–3%. Visual inspection of side-by-side `helmet_compare_cam{0,50,100,150}.png` confirms framing, silhouette, diffuse highlight position, and edge fragmentation are all aligned. The remaining per-pixel error is concentrated in sub-pixel silhouette misalignment and specular highlight shape, not in the shading model itself.

## Known Issues / TODO

- **Specular LOD semantics mismatch (likely root cause of remaining per-pixel error)**.
  `pbr.frag:76` uses `textureLod(uEnvMap, uv, specLod)` (absolute LOD, skips screen-space derivatives), but Python's `env_map.sample_specular` passes `mip_level_bias` to nvdiffrast (relative bias added to auto-LOD from UV derivatives). On curved surfaces the auto-LOD contribution is non-trivial. Switching to `texture(uEnvMap, uv, specLod)` (bias mode, matching `pbr.frag:60` for diffuse) should close most of the gap. Symptom: Web specular highlights are slightly more blurred and brighter than GT.

- **BRDF LUT UV mapping may not match Python's `grid_sample(align_corners=True)`**.
  `pbr.frag:78` uses `(val*(size−1)+0.5)/size` (pixel-center/`align_corners=False` semantics), Python `brdf_lut.py:120` uses `grid_sample(..., align_corners=True)` with `grid ∈ [0,1]`. These are not equivalent at the boundaries. Needs verification with a unit test before changing either side.

- **baseColor gamma encoding**: Python export uses `pow(1/2.2)` (`pbr_model.py:259`); Web relies on Three.js `SRGBColorSpace` auto-decode, which uses the true sRGB EOTF. The two curves differ by ~1–5% in the midtones. Acceptable for now.

- **Diffuse channel skews slightly cooler/darker than GT** in side-by-side (`helmet_compare_diffuse_cam50.png`). Cause not yet isolated — possibly related to the env map mip bias or to the baseColor sRGB curve above.

- **Piano scene not yet re-validated** under the no_normal data source. The `piano_pbr.zip` in `export/scenes/` is still the `_pbr` (normal-mapped) version; needs repacking from `output/piano_no_normal/epoch2000/` and an AB run before this tag can claim dual-scene support.

- **Debug scaffolding in `pbr.frag`** (28 debug channels, `uDebug` uniform, HUD overlay in `App.ts`) is still in the shader. Should be cleaned up once the specular LOD fix lands.
