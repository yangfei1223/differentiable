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

## 验证报告

Web 渲染与训练管线 GT 的 AB 像素对比。报告在 `app/reports/`，图片资源在 `app/resource/`（与主仓 `docs/reports/` + `resource/` 约定一致）。

| 场景 | 数据源 | PSNR | 状态 | 报告 |
|------|--------|------|------|------|
| 头盔 | `helmet_no_normal/epoch2000` | 17.47 dB | 视觉对齐通过（前景均值偏差 ≤3 RGB，边缘密度一致）；PSNR 未达 30 dB 目标，主因是 specular LOD 语义不一致 | [Helmet AB no_normal](reports/01_Helmet_AB_no_normal.md) |

## 遗留问题

- **Specular LOD 语义不一致**（helmet 像素误差主因）。`pbr.frag:76` 用 `textureLod`（绝对 LOD），Python `env_map.sample_specular` 用 nvdiffrast `mip_level_bias`（相对偏置）。改为 `texture(uEnvMap, uv, specLod)`（与 `pbr.frag:60` diffuse 一致）。
- **BRDF LUT UV 映射**可能与 Python `grid_sample(align_corners=True)` 不等价，需单测验证。
- **baseColor gamma 曲线**：Python 导出用 `pow(1/2.2)`，Web 用 Three.js `SRGBColorSpace`（真正 sRGB EOTF），mid-tone 差 1–5%。暂可接受。
- **钢琴场景**未在 `no_normal` 数据源下验证（`piano_pbr.zip` 还是 `_pbr` 版本）。
- **调试代码**（`pbr.frag` 28 个 debug 通道、HUD）待 specular LOD 修好后清理。
