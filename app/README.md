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

## 项目结构

```
app/
├── index.html                  # 入口 HTML
├── package.json                # 依赖与 npm scripts
├── vite.config.ts              # Vite 配置
├── tsconfig.json               # TypeScript 配置
├── src/
│   ├── main.ts                 # 入口，实例化 App
│   ├── app/
│   │   ├── App.ts              # 应用编排：场景加载、相机覆盖、离屏渲染、动画循环
│   │   └── SceneLoader.ts      # .zip 解包、manifest 解析、blob URL 管理
│   ├── render/
│   │   ├── PBRPipeline.ts      # 渲染器、glTF 加载、node→mesh 映射、tangent 计算
│   │   ├── PBRMesh.ts          # 单 primitive 的 PBR ShaderMaterial 封装
│   │   ├── Environment.ts      # env map（HDR/LDR）+ BRDF LUT 加载、mipmap 链
│   │   └── computeTangents.ts  # JS Mikktspace 切线计算（镜像 mesh.py:78-141）
│   ├── shaders/                # GLSL（移植参考，无 Three.js 依赖）
│   │   ├── common.glsl         # PI 常量 + direction_to_uv()
│   │   ├── pbr.vert            # 顶点着色器
│   │   └── pbr.frag            # 片段着色器（split-sum PBR + debug 通道）
│   ├── ui/
│   │   ├── CameraControls.ts   # OrbitControls + Blender Z-up→Y-up 转换
│   │   ├── LoadingOverlay.ts   # 加载状态 UI
│   │   ├── PerfStats.ts        # FPS / draw call 统计
│   │   └── ScenePicker.ts      # 场景选择 / 拖拽
│   ├── math/
│   │   └── pbr_math.ts         # PBR 数学的 TS 镜像（供单测用）
│   ├── types/
│   │   └── manifest.ts         # manifest.json 的 TS 类型
│   └── vite/
│       ├── glsl-plugin.ts      # Vite 插件：#include 解析 + ?raw 加载
│       └── raw.d.ts            # ?raw 导入的 TS 声明
├── tests/
│   └── equivalence.test.ts     # GLSL 数学 vs Python pbr_model.py 等价性
├── reports/                    # AB 验证报告（markdown）
└── resource/                   # 报告配套图片资源
```

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
| `uDiffuseMipBias` | float | 等于 `uMaxEnvMip`（保留供调试用；当前 diffuse irradiance 实际用 `textureLod(uEnvMap, uv, uMaxEnvMip)` 直接采最模糊 mip，不再使用此 uniform） |
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
   > ⚠ **注意**：离屏渲染路径与浏览器 live viewport 渲染结果不一致（详见遗留问题第 4 条），目前**不可用于严格 AB 对比**。PSNR 报告改用 viewport 中心方图裁剪 + resize 替代。
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

| 场景 | 数据源 | PSNR（4 相机） | 状态 | 报告 |
|------|--------|----------------|------|------|
| 头盔 | `helmet_no_normal/epoch2000` | 16.82 / 17.57 / 18.09 / 19.65 dB | 视觉对齐通过（修复 irradiance mip 后无染色，前景重叠率 24–56%）；PSNR 未达 30 dB，主因是 specular LOD 语义不一致 | [Helmet AB no_normal](reports/01_Helmet_AB_no_normal.md) |
| 钢琴 | `piano_no_normal/epoch2000` | 18.48 / 25.99 / 24.31 / 19.58 dB | 视觉对齐通过（琴键白色、共鸣板暖木色、金属架金色，修复 irradiance mip + DoubleSide + 深嵌套 node 解析后）；PSNR 未达 30 dB | [Piano AB no_normal](reports/02_Piano_AB_no_normal.md) |

> **PSNR 解读**：所有 PSNR 用浏览器 viewport 中心方图裁剪 + LANCZOS resize 到 1024×1024 后跟 GT 比对（离屏 1024×1024 渲染与 live viewport 不一致，见遗留问题第 4 条）。**视觉对齐是主要判定**，PSNR 是参考。

## 遗留问题

按预期收益排序：

1. **Specular LOD 语义不一致**（helmet/piano PSNR 主因）。`pbr.frag:76` 用 `textureLod`（绝对 LOD），Python `env_map.sample_specular` 用 nvdiffrast `mip_level_bias`（相对偏置）。改为 `texture(uEnvMap, uv, specLod)`（与 diffuse 路径 `pbr.frag:60` 一致），恢复 auto-LOD 贡献。

2. **共鸣板色温轻微偏差**（piano 特有）。Web 比 GT 稍偏黄/欠饱和，可能与 specular LOD（第 1 条）或 baseColor gamma（第 3 条）相关。视觉不显著，PSNR 影响较大。

3. **BRDF LUT UV 映射**可能与 Python `grid_sample(align_corners=True)` 不等价。`pbr.frag:78` 用 `(val*(size−1)+0.5)/size`（pixel-center 语义），需单测验证边界行为。

4. **离屏渲染路径不可靠**：`setSize(1024,1024)` + `readPixels` 得到的图与浏览器 live screenshot 差异大（mean diff ~18 RGB）。当前 AB PSNR 用 viewport 中心方图裁剪 + resize 替代，存在亚像素错位和插值损失。根因未深究。

5. **baseColor gamma 曲线**：Python 导出用 `pow(1/2.2)`（`pbr_model.py:259`），Web 用 Three.js `SRGBColorSpace`（真正 sRGB EOTF），mid-tone 差 1–5%。暂可接受。

6. **纹理数据导出方式**：当前打包脚本用训练 logger 写的 PNG 作为纹理源（已验证与 checkpoint tensor 一致）。用户期望未来直接从 `pbr_checkpoint.pt` 的原始 tensor 导出，跳过 PNG 中间件。

7. **调试代码**（`pbr.frag` 28 个 debug 通道、HUD overlay）待 specular LOD 修好后清理。
