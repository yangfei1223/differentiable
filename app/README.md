# PBR Web Viewer

WebGL2 viewer，用于验证可微烘焙管线的 PBR 输出。加载 `scripts/package_runtime_asset.py` 打包的 `.zip` 资产，用 GLSL 着色器渲染，着色数学严格对齐训练管线（`src/shading/pbr_model.py`）。

## 快速开始

```bash
# 从项目根目录打包一个训练输出
python -m scripts.package_runtime_asset \
  --glb data/helmet_260604/scene/lowpoly.glb \
  --epoch-dir output/helmet_no_normal/epoch2000 \
  --scene-name helmet

# 启动 dev server
cd app
npm install
npm run dev
```

浏览器打开 `http://localhost:5173`，自动加载 helmet 场景。

## 资产包格式

完整规范见 `docs/superpowers/specs/2026-06-17-pbr-web-viewer-design.md` §3。最小目录结构：

```
scene.zip
├── manifest.json
├── geometry/scene.glb
├── textures/
│   ├── env_map.png       # 或 env_map.hdr（RGBE，保留训练时 HDR 值）
│   ├── brdf_lut.png
│   └── {submesh_name}/
│       ├── base_color.png
│       ├── roughness.png
│       ├── metallic.png
│       └── normal_map.png
```

也支持把 `.zip` 文件直接拖到浏览器窗口加载。

## 操作

| 操作 | 桌面 | 移动端 |
|------|------|--------|
| 旋转 | 左键拖动 | 单指拖动 |
| 平移 | 右键拖动 | 双指拖动 |
| 缩放 | 滚轮 | 双指捏合 |
| 复位 | `R` 键 | — |

## GLSL 文件（移植参考）

PBR 数学全部在 `src/shaders/`：

- `common.glsl` — `PI` 常量 + `direction_to_uv()` 辅助函数
- `pbr.vert` — 顶点变换，输出世界空间 normal/tangent/view
- `pbr.frag` — 片段着色器，包含完整 split-sum PBR 数学

这些文件**不依赖 Three.js**，可直接复制到原生引擎（Vulkan/Metal/GLES），改动很小。

### Uniforms 契约

片段着色器期望的 uniforms：

| Uniform | 类型 | 说明 |
|---------|------|------|
| `uBaseColor` | sampler2D | sRGB base color 贴图 |
| `uRoughness` | sampler2D | Linear roughness（R 通道） |
| `uMetallic` | sampler2D | Linear metallic（R 通道） |
| `uNormalMap` | sampler2D | Tangent-space normal map |
| `uEnvMap` | sampler2D | Equirect 环境贴图（含 mipmap） |
| `uBRDFLut` | sampler2D | 双通道 GGX BRDF 积分 LUT |
| `uMaxEnvMip` | float | `floor(log2(max(envH, envW)))` |
| `uDiffuseMipBias` | float | 等于 `uMaxEnvMip` |
| `uNormalMapEnabled` | bool | 是否启用法线贴图 |
| `uBRDFLutSize` | vec2 | LUT 分辨率（如 256×256） |

## 开发

```bash
npm run test      # vitest 单元测试（数学等价性）
npm run build     # 生产构建到 dist/
npm run preview   # 预览生产构建
```

## 等价性测试

`tests/equivalence.test.ts` 在 TypeScript 里镜像 GLSL 数学，验证与 Python `src/shading/pbr_model.py` 的数值等价性。改完 shader 后跑一遍。

### AB 像素对比

与训练管线 GT 端到端对比：

1. 从训练输出打包资产：
   ```bash
   python -m scripts.package_runtime_asset \
     --glb data/helmet_260604/scene/lowpoly.glb \
     --epoch-dir output/helmet_no_normal/epoch2000 \
     --scene-name helmet
   ```
2. 启动 dev server，加载相机 hash（参数从 `data/helmet_260604/cameras.json` 取）：
   ```
   http://localhost:5173/#cam=px,py,pz,tx,ty,tz,ux,uy,uz,fov
   ```
3. 离屏渲染 1024×1024：URL 加 `?render=1024` 自动下载 PNG；或在 console 手动驱动 `window.__pipeline.setSize(1024,1024); .render(); readPixels; toDataURL`。
4. 与 `output/{scene}_no_normal/epoch2000/compare_NNNN.png` 的右上"Rendered"面板对比：
   ```bash
   python scripts/ab_compare.py helmet
   ```

GT 面板布局（来自 `src/shading/pbr_logger.py`）：

```
┌───────────┬───────────┐
│   GT      │ Rendered  │   ← 上行
├───────────┼───────────┤
│ Diffuse   │ Specular  │   ← 下行
└───────────┴───────────┘
```

每个面板经过 `pow(1/2.2)` gamma 编码（`pbr_logger.py:162`）。

## 验证报告

Web 渲染与训练管线 GT 的 AB 像素对比。报告在 `app/reports/`，图片资源在 `app/resource/`（与主仓 `docs/reports/` + `resource/` 约定一致）。

| 场景 | 数据源 | PSNR | 状态 | 报告 |
|------|--------|------|------|------|
| 头盔 | `helmet_no_normal/epoch2000` | 17.47 dB | 视觉对齐通过（前景均值偏差 ≤3 RGB，边缘密度一致）；PSNR 未达 30 dB 目标，主因是 specular LOD 语义不一致 | [Helmet AB no_normal](reports/01_Helmet_AB_no_normal.md) |

### 下一步

按预期收益排序：

1. **Specular LOD 改用 `texture(bias)` 模式**（helmet PSNR 主因）。`pbr.frag:76` 把 `textureLod(uEnvMap, uv, specLod)` 改成 `texture(uEnvMap, uv, specLod)`，与 diffuse 路径（`pbr.frag:60`）一致，恢复 nvdiffrast 的 auto-LOD 贡献。
2. **BRDF LUT UV 映射单测**。`pbr.frag:78` 的 `(val*(size−1)+0.5)/size`（pixel-center）vs Python `grid_sample(align_corners=True)`，写一个单测确认边界是否等价，不等价再决定改哪边。
3. **Piano 场景同源验证**。当前 `piano_pbr.zip` 还是 `_pbr` 版本，需要重新从 `output/piano_no_normal/epoch2000/` 打包，跑同流程 AB 对比。
4. **调试代码清理**。`pbr.frag` 的 28 个 debug 通道、HUD overlay 等 speculative 调试代码，等 specular LOD 修复落地后清理。

## 遗留问题

- **Specular LOD 语义不一致**（helmet 像素误差主因）。`pbr.frag:76` 用 `textureLod`（绝对 LOD），Python `env_map.sample_specular` 用 nvdiffrast `mip_level_bias`（相对偏置）。改为 `texture(uEnvMap, uv, specLod)`（与 `pbr.frag:60` diffuse 一致）。
- **BRDF LUT UV 映射**可能与 Python `grid_sample(align_corners=True)` 不等价，需单测验证。
- **baseColor gamma 曲线**：Python 导出用 `pow(1/2.2)`，Web 用 Three.js `SRGBColorSpace`（真正 sRGB EOTF），mid-tone 差 1–5%。暂可接受。
- **钢琴场景**未在 `no_normal` 数据源下验证（`piano_pbr.zip` 还是 `_pbr` 版本）。
- **调试代码**（`pbr.frag` 28 个 debug 通道、HUD）待 specular LOD 修好后清理。
