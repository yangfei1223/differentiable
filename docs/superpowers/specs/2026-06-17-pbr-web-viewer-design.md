# PBR Web Viewer 设计规范

**日期**：2026-06-17
**状态**：已批准（待 spec 复审）
**关联**：v0.3 split-sum PBR 管线输出验证

## 1. 目标与非目标

### 1.1 目标

1. **验证烘焙管线输出**：消费 PBR 训练结果（GLB + 材质贴图 + env map + BRDF LUT），在浏览器中完成实时渲染，提供与训练等价的视觉输出
2. **提供移植样例**：核心 PBR 数学用 GLSL ES 3.00 实现，按移动端 native 移植友好度组织（vertex + fragment pair）
3. **支持任意资产加载**：WebUI 接受任意 glTF + 烘焙输出组合（不限于 helmet/piano），通过预打包 .zip 输入

### 1.2 非目标（YAGNI）

- ❌ GT 参考对比、轨道相机重放、调试通道切换（首期不做）
- ❌ Tone mapping、bloom、SSAO、shadow
- ❌ 透明物体、排序
- ❌ 灯光对象（env map 已包含所有光照）
- ❌ SH / NLM 模型支持（首期只 PBR）
- ❌ 多 viewport、画中画
- ❌ 持久化场景管理（每次加载即用即弃）

### 1.3 设计原则

1. **运行态逻辑严格对齐训练实现**：Web 端的着色数学必须与 Python `src/shading/pbr_model.py` 逐行等价，不引入近似、不引入简化替代（如 fresnel_schlick 替代）、不省略任何步骤
2. **GLSL 是移植目标，不是 Three.js 配方**：着色器代码独立于 Three.js，可整体复制到任何 native 引擎
3. **契约优先**：Python 端与 Web 端通过 manifest.json 显式约定，不依赖目录约定

## 2. 项目结构

```
app/
├── package.json              # Vite + TypeScript + Three.js
├── vite.config.ts
├── tsconfig.json
├── public/
│   └── favicon.svg
├── src/
│   ├── main.ts               # 入口：初始化 Renderer、UI 事件绑定
│   ├── app/
│   │   ├── App.ts            # 顶层协调：loadScene → buildSceneGraph → animate
│   │   └── SceneLoader.ts    # .zip 解包 → manifest 验证 → 资产 URL 化 → 回传 AssetBundle
│   ├── render/
│   │   ├── PBRPipeline.ts    # 构造 Three.WebGLRenderer、场景、相机
│   │   ├── PBRMesh.ts        # 单个 submesh 的 Mesh + ShaderMaterial 包装
│   │   └── Environment.ts    # env map 纹理 + mip 链构造
│   ├── ui/
│   │   ├── ScenePicker.ts    # 下拉选择预置场景 + .zip 拖入
│   │   ├── CameraControls.ts # 包装 OrbitControls（鼠标 + 触摸）
│   │   └── PerfStats.ts      # FPS / draw calls / tri count / texture mem
│   ├── shaders/
│   │   ├── common.glsl       # 常量 + 辅助函数（移植复用）
│   │   ├── pbr.vert          # 顶点着色器
│   │   └── pbr.frag          # 片段着色器（含全部 PBR 数学）
│   └── types/
│       └── manifest.ts       # TS 接口定义（与 Python 端共享 schema）
├── tests/                    # Vitest 单元测试
└── README.md
```

### 2.1 技术栈

- **TypeScript**：用接口锁定 manifest 契约（避免 Python 端与 JS 端字段漂移）
- **Vite**：HMR 快、原生支持 `?raw` 导入 .glsl 字符串
- **Three.js r160+**：稳定、glTFLoader 成熟、OrbitControls 内置触摸支持
- **JSZip**：浏览器端解压 .zip
- **GLSL ES 3.00**（WebGL2）：与移动端 GLSL ES 3.1（Vulkan）/Metal SL 几乎逐行可移植

## 3. 资产包格式（.zip）

### 3.1 目录结构

```
helmet_pbr.zip              ← 一个完整可加载的资产包
├── manifest.json           ← 唯一可信源
├── geometry/
│   └── scene.glb           ← 原始 glTF（含 UV、法线、可选 TANGENT）
├── textures/
│   ├── env_map.png         ← equirect LDR sRGB（运行时反 gamma 到线性）
│   ├── brdf_lut.png        ← 256×256 RG8（B 通道忽略）
│   └── helmet/             ← 每个 submesh 一个子目录
│       ├── base_color.png  ← sRGB
│       ├── roughness.png   ← R 通道线性
│       ├── metallic.png    ← R 通道线性
│       └── normal_map.png  ← tangent-space [-1,1]→[0,255]
```

多 submesh 场景：`textures/{submesh_name}/{...}`，每个 submesh 一个子目录。

### 3.2 manifest.json schema

```jsonc
{
  "schema_version": 1,
  "scene_name": "helmet",
  "generator": {
    "tool": "differentiable-baker",
    "version": "v0.4",
    "render_mode": "pbr",
    "epoch": 2000,
    "psnr_db": 20.81
  },
  "geometry": {
    "glb_path": "geometry/scene.glb",
    "up_axis": "Y",              // glTF 标准 Y-up
    "scale": 1.0
  },
  "environment": {
    "env_map_path": "textures/env_map.png",
    "is_hdr": false,             // 当前导出 clamp 到 [0,1]，按 LDR 处理
    "diffuse_mip_bias": -1,      // -1 表示用最模糊 mip（运行时计算 = floor(log2(max(H,W)))）
    "specular_mip_levels": -1    // -1 表示自动（= floor(log2(max(H,W)))）
  },
  "brdf_lut_path": "textures/brdf_lut.png",
  "submeshes": [
    {
      "name": "helmet",          // 与 glTF primitive/mesh name 一致
      "match_by": "primitive_name",  // primitive_name | material_name | mesh_index
      "textures": {
        "base_color": "textures/helmet/base_color.png",
        "roughness": "textures/helmet/roughness.png",
        "metallic": "textures/helmet/metallic.png",
        "normal_map": "textures/helmet/normal_map.png"
      }
    }
  ]
}
```

### 3.3 Submesh 匹配机制

Python 端用 `mesh.name or f"mesh_{node.mesh}"` + `_prim{pi}` 后缀（多 primitive 时）作为 submesh name。glTF 端可能出现同名 mesh 或空 name，因此提供三种匹配策略：

- `primitive_name`（默认）：匹配 glTF 中 primitive 所属 mesh 的 name
- `material_name`：匹配 primitive 引用的 material name
- `mesh_index`：按 mesh 在 glTF 中的整数索引匹配（最稳定兜底）

打包脚本默认生成 `primitive_name`，必要时可手改 manifest 切换策略。

### 3.4 法线与切线约定

- **法线**：tangent-space，[-1,1] → [0,255] 编码（与 Python `export` 一致）
- **Tangent 来源**：优先用 glTF `TANGENT` attribute；缺失时由 Three.js glTFLoader 自动生成（Mikktspace 风格，与 Python `compute_vertex_tangents` 等价）
- **Bitangent 计算**：`B = cross(N, T)`，右手系（与 Python 一致）

### 3.5 Env Map 运行时处理

- Python 训练用 `softplus(raw)` 解码 HDR，但导出 PNG 时 `clamp(0,1)` → 实际为 LDR sRGB
- 运行时方案：加载 PNG → `pow(2.2)` 反 sRGB 到线性 → 用于 diffuse/specular 计算
- **未来 HDR 支持**：manifest 中 `is_hdr` 字段；后续 Python 端可加 EXR→RGBE 转换

## 4. 渲染管线

### 4.1 数据流

```
SceneLoader
  ↓ AssetBundle (manifest + blob URLs)
PBRPipeline.loadScene(bundle)
  ├── Environment.fromBundle()
  │     ├── env_map.png → THREE.Texture (RGBA, sRGB→linear, generateMipmaps=true)
  │     └── brdf_lut.png → THREE.DataTexture (RG8)
  ├── glTFLoader.parse(bundle.glbBlobUrl)
  └── 遍历 glTF primitives：
        PBRMesh.build(primitive, submeshManifest)
          ├── 匹配 submesh by match_by
          ├── 构造 ShaderMaterial（注入 uniforms + pbr.vert/pbr.frag 字符串）
          ├── 上传 4 张纹理（color space 标注）
          └── 包装为 THREE.Mesh 加入场景
  ↓
CameraControls.fitToBoundingSphere(scene)
animate loop 启动
```

### 4.2 GLSL 文件布局（3 个）

**`shaders/common.glsl`** — 常量与辅助函数

```glsl
const float PI = 3.14159265359;

// 与 Python EnvironmentMap.direction_to_uv 逐行对齐
vec2 direction_to_uv(vec3 dir) {
  float u = atan(dir.z, dir.x) / (2.0 * PI) + 0.5;
  float v = asin(clamp(dir.y, -0.999, 0.999)) / PI + 0.5;
  return vec2(u, v);
}
```

**`shaders/pbr.vert`** — 顶点着色器

- 输入 attribute：`position, uv, normal, tangent`
- 输出 varying：`vUV, vNormalW, vTangentW, vBitangentW, vViewDirW`
- `vNormalW = normalize(mat3(normalMatrix) * normal)`
- `vTangentW = normalize(mat3(modelMatrix) * tangent.xyz)`（glTF 中 tangent.w=1 表示 bitangent 不需翻转）
- `vBitangentW = cross(vNormalW, vTangentW) * tangent.w`（与 Python `compute_vertex_tangents` 的 `B = cross(N, T)` 一致）
- `vViewDirW = normalize(cameraPosition - worldPos)` — Python `view_dirs` 也是从相机指向顶点的归一化向量

**`shaders/pbr.frag`** — 片段着色器（严格 mirror `shade_submesh`）

```glsl
#version 300 es
precision highp float;

#include "common.glsl"

in vec2 vUV;
in vec3 vNormalW;
in vec3 vTangentW;
in vec3 vBitangentW;
in vec3 vViewDirW;
out vec4 fragColor;

uniform sampler2D uBaseColor;
uniform sampler2D uRoughness;
uniform sampler2D uMetallic;
uniform sampler2D uNormalMap;
uniform sampler2D uEnvMap;
uniform sampler2D uBRDFLut;
uniform float uMaxEnvMip;         // = floor(log2(max(envH, envW)))
uniform float uDiffuseMipBias;    // = uMaxEnvMip（取最模糊 mip 近似 diffuse irradiance）
uniform bool  uNormalMapEnabled;  // manifest 或运行时开关
uniform vec2  uBRDFLutSize;       // 用于 align_corners 修正

void main() {
  vec3 N = normalize(vNormalW);
  vec3 T = normalize(vTangentW);
  vec3 B = normalize(vBitangentW);
  vec3 V = normalize(vViewDirW);

  // ===== 1. Material decode（mirror decode_material）=====
  vec3 baseColor = pow(texture(uBaseColor, vUV).rgb, vec3(2.2));
  float roughness = texture(uRoughness, vUV).r;
  float metallic = texture(uMetallic, vUV).r;
  vec3 normalTS = texture(uNormalMap, vUV).rgb * 2.0 - 1.0;
  normalTS = normalize(normalTS);  // mirror F.normalize

  // ===== 2. Normal mapping（直接 TBN，无 Gram-Schmidt）=====
  if (uNormalMapEnabled) {
    N = normalize(T * normalTS.x + B * normalTS.y + N * normalTS.z);
  }

  // ===== 3. Reflect（mirror shade_submesh 第 3 步）=====
  float NdotV = clamp(dot(N, V), 0.0, 1.0);
  vec3 R = 2.0 * NdotV * N - V;
  R = normalize(R);

  // ===== 4. F0 + Diffuse（mirror compute_F0 + shade_submesh 第 4 步）=====
  vec3 F0 = mix(vec3(0.04), baseColor, metallic);  // dielectric_F0 = 0.04
  vec3 kd = (1.0 - metallic) * (1.0 - F0);
  vec3 irradiance = textureLod(uEnvMap, direction_to_uv(N), uDiffuseMipBias).rgb;
  vec3 diffuse = kd * baseColor * irradiance;

  // ===== 5. Specular（mirror shade_submesh 第 5 步）=====
  float specLod = roughness * uMaxEnvMip;  // 线性映射，与 Python 一致
  vec3 prefiltered = textureLod(uEnvMap, direction_to_uv(R), specLod).rgb;
  // BRDF LUT 采样，align_corners 修正
  vec2 brdfUv = (vec2(NdotV, roughness) * (uBRDFLutSize - 1.0) + 0.5) / uBRDFLutSize;
  vec2 brdf = texture(uBRDFLut, brdfUv).rg;
  vec3 specular = (F0 * brdf.x + brdf.y) * prefiltered;

  // ===== 6. Combine（mirror shade_submesh 第 6 步）=====
  vec3 rgb = diffuse + specular;
  rgb = clamp(rgb, 0.0, 1.0);

  fragColor = vec4(rgb, 1.0);
}
```

### 4.3 等价性保证清单

| Python 实现 | GLSL 实现 | 等价点 |
|---|---|---|
| `dr.texture(filter="linear", boundary="clamp")` | `texture()` 默认 `LINEAR` + `CLAMP_TO_EDGE` | ✓ |
| `dr.texture(filter="linear-mipmap-linear")` | `textureLod()` + `LINEAR_MIPMAP_LINEAR` | ✓ |
| `softplus(raw)` decode env | 预解码为 LDR PNG（Python export 已做） | ✓ |
| `brdf_lut` grid_sample `align_corners=True` | `texture()` align_corners=False + 运行时坐标修正 | ✓ |
| `F.normalize(normal_raw)` | `normalize(normalTS)` | ✓ |
| `2*NdotV*N - V` 然后 normalize | 同 | ✓ |
| `kd = (1-metallic)*(1-F0)` | 同 | ✓ |
| `specular = (F0*scale + bias) * prefiltered` | 同 | ✓ |
| `rgb.clamp(0,1)` | `clamp(rgb, 0, 1)` | ✓ |
| diffuse mip = `floor(log2(max(H,W)))` | `uDiffuseMipBias = uMaxEnvMip` | ✓ |
| specular mip = `roughness * max_mip`（线性） | 同（不引入 roughness² ） | ✓ |

### 4.4 关键决策

1. **无运行时 Fresnel Schlick**：Fresnel 已烤进 BRDF LUT（Python 训练时即如此），运行时只做 `F0*scale + bias`
2. **BRDF LUT align_corners 修正**：Python 用 `grid_sample(align_corners=True)`，WebGL `texture()` 默认 `False`。运行时坐标变换 `uv = (uv * (size-1) + 0.5) / size` 一次性修正
3. **sRGB 处理**：baseColor/normalMap 上传时设 `SRGBColorSpace`；roughness/metallic 设 `LinearSRGBColorSpace`；输出 framebuffer 由 Three.js 自动 sRGB encode（`renderer.outputColorSpace = SRGB`）
4. **Mipmap 生成**：env_map 上传时 `texture.generateMipmaps = true; texture.minFilter = LinearMipmapLinear` → WebGL 自动生成 mip 链
5. **Tangent 缺失回退**：glTFLoader 自动注入 TANGENT；若仍未有，fragment shader 用 derivative-based TBN（dFdx/dFdy）兜底
6. **YAGNI**：暂不实现 SSAO、shadow、tone mapping、bloom；与"验证烘焙结果"无关

## 5. 应用层与 UI

### 5.1 启动流程

```
main.ts
  ↓ 初始化
PBRPipeline 构造（WebGL2 context, Three.js renderer, sRGB output）
  ↓
ScenePicker 渲染（顶部 toolbar：[场景下拉] [或拖入 .zip] [?]）
  ↓ 用户选择（或启动时自动加载第一个预置场景）
SceneLoader.load(source)
  ↓ JSZip 解包
验证 manifest.schema_version == 1
  ↓
构造 blob URLs 指向 GLB / 各 PNG
  ↓
返回 AssetBundle
  ↓
PBRPipeline.loadScene(bundle)
  ↓
CameraControls.fitToBoundingSphere(scene)
animate loop 启动
```

### 5.2 UI 布局

```
┌─────────────────────────────────────────────────────────┐
│ Differentiable Baker — PBR Viewer     [helmet ▼] [?]    │ ← 顶部 toolbar
├─────────────────────────────────────────────────────────┤
│                                                         │
│              [3D 渲染区域 - 全屏 canvas]                  │
│                                                         │
│                                       ┌────────────┐    │
│                                       │ FPS: 60    │    │
│                                       │ Draw: 1    │    │
│                                       │ Tris: 12K  │    │
│                                       │ Tex: 48MB  │    │
│                                       └────────────┘    │
└─────────────────────────────────────────────────────────┘
```

**布局策略**：纯 canvas 全屏 + 浮层 UI（CSS `position: absolute`），避免布局重排影响渲染性能。性能仪表放右上角，半透明背景。

### 5.3 交互

| 操作 | 桌面 | 移动 |
|---|---|---|
| 旋转 | 鼠标左键拖动 | 单指拖动 |
| 平移 | 鼠标右键拖动 | 双指拖动 |
| 缩放 | 滚轮 | 双指捏合 |
| 重置视角 | `R` 键 / 按钮 | 重置按钮 |

OrbitControls 原生支持以上所有手势。

### 5.4 场景加载 UX

**预置场景下拉**：
- 启动时 fetch `/scenes/index.json`（由 Python 打包脚本生成）
- 内容：`[{name: "helmet", file: "/scenes/helmet_pbr.zip", psnr_db: 20.81}, ...]`
- 默认加载第一个

**拖入 .zip**：
- 全窗口 drag-over 高亮
- drop 后调用 `SceneLoader.load(file)`
- 失败时显示错误 toast（manifest 缺失、schema 不匹配、GLB 解析失败）

**加载状态**：
- Loading spinner 覆盖层
- 显示当前阶段：`Unzipping → Parsing manifest → Loading GLB → Uploading textures → Ready`

### 5.5 性能仪表（PerfStats）

每 500ms 采样：
- **FPS**：`requestAnimationFrame` 间隔平均
- **Draw calls**：`renderer.info.render.calls`
- **Triangles**：`renderer.info.render.triangles`
- **Texture memory**：遍历所有纹理 `width * height * channels * 4`（粗略估算）

## 6. 测试与构建

### 6.1 测试策略

| 层次 | 工具 | 范围 |
|---|---|---|
| 单元 | Vitest（纯 JS，无 WebGL） | manifest schema 校验、direction_to_uv 数学等价性、submesh 匹配逻辑、material decode 数值、brdf_lut align_corners 修正、split_sum 组合 |
| 集成 | Vitest + headless WebGL（node-canvas 或 playwright） | 加载预置 .zip → 构造场景 → 检查 mesh/uniform 数量；可选，首期可跳过 |
| 端到端（人工） | 浏览器 | 视觉验证 + 性能仪表读数 |

**核心单测（保移植等价）：**

```ts
// tests/equivalence.test.ts
- direction_to_uv(0,1,0) ≈ (0.5, 1.0)        // 与 Python 一致
- direction_to_uv(1,0,0) ≈ (0.5, 0.5)
- decode_material: raw [0,0,0,0,-5,0,0,1] → baseColor≈(0.5,0.5,0.5), roughness=0.5, metallic≈0.007, normal=(0,0,1)
- brdf lut align_corners 修正公式正确性
- perturb_normal 当 normalTS=(0,0,1) 时 N 不变
- split_sum 当 roughness=1, metallic=0 → 主要 diffuse，specular≈0
```

测试无需启动 Three.js / WebGL，**纯 JS/TS 函数验证数学**（把 GLSL 逻辑用 TS 重写一份做 mirror 测试）。

### 6.2 构建命令

```bash
# 开发（HMR）
cd app
npm install
npm run dev          # → http://localhost:5173

# 生产构建
npm run build        # → app/dist/
npm run preview      # 预览生产包

# 测试
npm run test         # Vitest 单测
```

### 6.3 Vite 配置要点

- `publicDir`：dev 模式下指向 `../output`，让 `/scenes/xxx.zip` 可访问
- `?raw` 导入 .glsl 字符串：`import fragSrc from '../shaders/pbr.frag?raw'`
- GLSL `#include` 处理：自定义 Vite 插件（轻量字符串替换，不引入 glslify 依赖）

## 7. Python 打包脚本

`scripts/package_runtime_asset.py`：

```python
"""将烘焙输出打包为 Web Viewer 可加载的 .zip 资产包。

用法：
  python -m scripts.package_runtime_asset \
    --glb data/helmet_260604/scene/lowpoly.glb \
    --epoch-dir output/helmet_260604_pbr/epoch2000 \
    --scene-name helmet \
    --output output/helmet_pbr.zip
"""
```

**输入参数**：
- `--glb`：原始 glTF 模型路径（必需）
- `--epoch-dir`：烘焙输出目录（含 base_color.png 等）（必需）
- `--scene-name`：场景名（必需，用于 manifest.scene_name）
- `--psnr`：可选，训练 PSNR；不提供则从 epoch-dir/curves.png 同目录的 checkpoint 中读取
- `--epoch`：可选，默认从 --epoch-dir 目录名提取
- `--output`：可选，默认 `output/{scene_name}_pbr.zip`

**逻辑**：
1. 用 `gltf_loader.py` 现有逻辑解析 GLB，提取所有 primitive/mesh name
2. 检查 `--epoch-dir` 下的纹理文件完整性
3. 单 mesh：直接复制顶层纹理；多 mesh：遍历 `Object_*/` 子目录
4. 生成 manifest.json（按 §3.2 schema）
5. 打包为 .zip（按 §3.1 结构）
6. 更新 `output/scenes_index.json`（追加或更新条目）

**不修改核心代码**，纯独立脚本。

### 7.1 scenes_index.json 格式

由打包脚本维护，追加/更新条目：

```json
[
  {"name": "helmet", "file": "/scenes/helmet_pbr.zip", "psnr_db": 20.81, "epoch": 2000},
  {"name": "piano",  "file": "/scenes/piano_pbr.zip",  "psnr_db": 28.80, "epoch": 2000}
]
```

## 8. 风险与开放问题

| 风险 | 影响 | 缓解 |
|---|---|---|
| Three.js glTFLoader 生成的 tangent 与 Python `compute_vertex_tangents` 数值不完全一致 | normal mapping 方向可能微差 | spec 已要求使用 Mikktspace 风格；如出现明显差异，可改用打包脚本预生成 TANGENT 注入 GLB |
| Env map 是 LDR 而非真 HDR | 高光强度与训练有差异 | 已记录在 manifest.is_hdr；后续 Python 加 EXR→RGBE 转换 |
| BRDF LUT 是 RGB PNG 而非 RG8 | 内存浪费 ~33% | 首期接受；后续可让 Python 端加选项导出 RG8 |
| OrbitControls 在低端移动设备性能 | 移植参考价值降低 | 性能仪表监控；首期不优化 |
| WebGL2 在 iOS Safari 兼容性 | 部分老设备不支持 | 首期目标桌面浏览器 + 现代移动端；不做 WebGL1 回退 |

## 9. 后续工作（不在本 spec 范围）

- v0.5+：GT 参考对比、轨道相机重放、调试通道切换（diffuse/specular/normal/roughness/metallic）
- HDR env map 支持（EXR → RGBE 打包）
- NLM 模型运行态（需运行 TinyMLP 推理，移植到 GLSL 复杂度高，需单独 spec）
- 移植样例文档：如何把 GLSL + JS 资产加载逻辑迁移到 native（Vulkan/Metal/GLES）

## 10. 验收标准

实现完成后必须满足：

1. ✅ 能加载 `helmet_pbr.zip` 和 `piano_pbr.zip`，无控制台错误
2. ✅ PBR 渲染结果在视觉上与训练 orbit.mp4 一致（无明显色偏、法线方向、高光位置）
3. ✅ 移动端手势（单指旋转、双指缩放）正常工作
4. ✅ 性能仪表显示合理的 FPS / draw call / triangle 数
5. ✅ 单测全部通过，覆盖数学等价性（direction_to_uv、decode_material、brdf_lut 修正、split_sum 组合）
6. ✅ GLSL 文件可独立抽取（不依赖 Three.js 类型或 import）
