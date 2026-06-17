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
