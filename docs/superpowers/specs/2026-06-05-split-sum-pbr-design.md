# Split-Sum PBR 可微烘焙管线设计

> 版本: v0.3 | 日期: 2026-06-05 | 分支: feature/split-sum-pbr

## 背景

v0.2 的 SH 参数化对漫反射材质效果良好（钢琴 20.37 dB），但对金属/玻璃等镜面材质严重不足（头盔 13.19 dB）。根本原因：SH order 2 只有 9 个系数，无法表达高频镜面反射。

Split-sum（Epic Games 2013）将渲染方程近似为：
```
∫ f(l,v) L(l) dl ≈ (∫ f_diff L(l) dl) + (∫ f_spec L(l) dl)
                  ≈ base_color * irradiance + (F0 * scale + bias) * prefiltered_color
```

这是移动端 PBR 的事实标准，2-texture lookup 即可完成着色。

## 目标

- 替代 SH 参数化，支持金属/镜面材质的高质量烘焙
- 全程 2-texture lookup（训练 = 部署），移动端直接可用
- 与现有 SH 路径共存，config 开关切换
- 联合优化材质贴图 + 环境贴图

## 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 部署平台 | 移动端 GPU | 项目核心需求 |
| 材质模型 | Standard PBR (base_color + roughness + metallic) | 5 通道，移动端广泛支持 |
| 环境光表示 | Equirectangular 经纬图 | 2D 友好，nn.Parameter 直接优化 |
| 近似策略 | 全程 2-texture lookup | 训练部署一致性 |
| 优化策略 | 联合优化 | 物理正确，实现简洁 |
| 架构风格 | 子目录 + ShadingModel 协议 | 与 SH 解耦，可插拔 |
| 代码兼容 | render_mode 开关，分支开发 | 保留现有 SH 代码不动 |

## 1. 材质参数化

### 1.1 材质贴图

单张 `nn.Parameter [1, H, W, 5]`，5 通道：

| 通道 | 含义 | 值域 | 渲染时约束 |
|------|------|------|------------|
| 0-2 | base_color (RGB) | [0, 1] | sigmoid |
| 3 | roughness | [0, 1] | sigmoid |
| 4 | metallic | [0, 1] | sigmoid |

选择单张而非分开存储：seam padding 一次处理、TV loss 一次计算、checkpoint 一次保存。渲染时 slicing 提取。

sigmoid 而非 clamp：sigmoid 处处可导，梯度不在边界处消失。

### 1.2 环境贴图

`nn.Parameter [1, Eh, Ew, 3]`，equirectangular 格式：
- 默认分辨率 `64×128`（中低频足够，移动端友好）
- 值域 [0, +∞)，softplus 约束保证非负
- 支持用户提供的 HDR/PNG 作为初始化

## 2. Split-Sum 渲染管线

### 2.1 PBR 着色公式

```
Lo = kd * diffuse + specular

kd = (1 - metallic) * (1 - F0)           # 漫反射权重
F0 = lerp(0.04, base_color, metallic)     # 菲涅尔 F0

diffuse = base_color * irradiance         # 漫反射项
  irradiance ≈ sample_prefiltered(env, level=0, direction=N)

specular = (F0 * scale + bias) * prefiltered_color
  prefiltered_color = trilinear_sample(prefiltered_env, R, roughness_level)
  (scale, bias) = sample_lut(brdf_lut, NdotV, roughness)
  R = reflect(-V, N)                      # 反射方向
```

### 2.2 Prefiltered Env Map（可导 mipmap）

对 `env_map` 做可导的 2D 高斯卷积，生成 M 个 mipmap 级别：

- M = 5 级（roughness 0.0 → 1.0 均匀分布）
- 每级 σ = roughness * max_sigma
- 用 `F.conv2d` 实现，权重固定不参与梯度，输入 env_map 参与梯度
- 输出 `prefiltered [1, M+1, Eh, Ew, 3]`

MVP 阶段用标准 2D 高斯，不处理 equirect 极点失真（后续优化）。

### 2.3 BRDF LUT

固定 `[256, 256, 2]` GGX BRDF 积分查找表：
- 横轴 = NdotV (0→1)
- 纵轴 = roughness (0→1)
- 通道 0 = scale, 通道 1 = bias
- 不参与优化，预计算
- 用 `F.grid_sample` 双线性插值采样

### 2.4 Equirect 坐标变换

方向向量 `[x, y, z]` → equirect UV：
```
u = atan2(z, x) / (2π) + 0.5    # [-π,π] → [0,1]
v = asin(y) / π + 0.5           # [-1,1] → [0,1]
```
全程可导，用 `F.grid_sample` 采样。

### 2.5 渲染流程

```
输入: mat_texture [1,H,W,5], env_map [1,Eh,Ew,3], camera, mesh
  ↓
1. Rasterize → UV, WorldPos, Normals, FaceMask
  ↓
2. Sample mat_texture → base_color, roughness, metallic (sigmoid)
  ↓
3. Compute: view_dir, reflect_dir, NdotV, F0
  ↓
4. Prefilter env_map → mipmap chain
  ↓
5. Diffuse: (1-metallic)*(1-F0) * base_color * sample_prefiltered(level=0, N)
  ↓
6. Specular: (F0*scale+bias) * trilinear_sample(prefiltered, R, roughness)
  ↓
7. Lo = diffuse + specular
```

## 3. 优化与训练

### 3.1 优化参数

| 参数 | 形状 | 学习率 | 初始化 |
|------|------|--------|--------|
| mat_texture | `[1, H, W, 5]` | base_lr | base_color=0.5, roughness=0.5, metallic=0.0 |
| env_map | `[1, 64, 128, 3]` | base_lr * env_lr_ratio | 均匀灰(0.5) 或用户 HDR |

env_lr_ratio 默认 1.0。

### 3.2 Loss

- L1 loss: weight=1.0
- SSIM loss: weight=0.2
- TV loss (mat_texture 全 5 通道): weight=0.005
- 环境贴图 TV 正则化 (可选): env_tv_weight=0.001

### 3.3 分辨率调度

材质贴图按现有 coarse-to-fine schedule 缩放。环境贴图固定分辨率。

### 3.4 Seam Padding

复用现有 `seam_padding.py`，对 5 通道材质贴图一次处理。

### 3.5 Checkpoint

```python
{
    'render_mode': 'pbr',
    'mat_texture': Tensor,     # [1, H, W, 5]
    'env_map': Tensor,         # [1, Eh, Ew, 3]
    'epoch': int,
    'config': dict,
}
```

## 4. 模块架构

### 4.1 目录结构

```
src/
├── renderer.py          # 共享渲染基础 (不变)
│   ├── rasterize()
│   ├── interpolate_uvs()
│   ├── interpolate_positions()
│   ├── interpolate_normals()    # [新增]
│   └── compute_view_dirs()
├── sh.py                # SH 数学库 (不变)
├── shading/             # [新增] 着色模型可插拔层
│   ├── __init__.py      # create_shading_model(render_mode, cfg)
│   ├── base.py          # ShadingModel 协议
│   ├── sh_model.py      # SH 着色模型包装
│   ├── pbr_model.py     # PBR 着色模型
│   └── pbr/
│       ├── __init__.py
│       ├── material.py  # 材质参数化
│       ├── env_map.py   # Equirect + 可导 prefilter
│       └── brdf_lut.py  # 固定 BRDF LUT
├── trainer.py           # 通用训练器 (修改: 接受 ShadingModel)
├── exporter.py          # 通用导出器 (修改: 按 model 分发)
├── mesh.py              # (修改: 增加法线加载)
├── config.py            # (修改: render_mode + PBRConfig)
├── video.py             # (修改: 支持 ShadingModel)
└── [其他不变]: dataset, camera, losses, seam_padding, utils
```

### 4.2 ShadingModel 协议

```python
class ShadingModel(Protocol):
    def parameters(self) -> list[nn.Parameter]: ...
    def init_textures(self, resolution: int) -> None: ...
    def shade(self, rast_out, uv, world_pos, normals, view_dirs,
              camera, resolution) -> tuple[Tensor, Tensor]: ...
    def get_material_texture(self) -> Tensor: ...
    def set_material_texture(self, texture: Tensor) -> None: ...
    def get_debug_info(self) -> dict: ...
    def export(self, output_dir: str) -> list[str]: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
```

### 4.3 关键模块职责

**shading/base.py** — ShadingModel 协议定义

**shading/sh_model.py** — SH 着色模型
- 包装现有 `sh.py` 的 init/decode 逻辑
- `parameters()` 返回 `[features_dc, features_rest]`
- `shade()` 内部调用 `decode_sh()`

**shading/pbr_model.py** — PBR 着色模型
- `parameters()` 返回 `[mat_texture, env_map]`
- `shade()` 调用 pbr/ 子模块完成 split-sum 着色
- `get_debug_info()` 返回 diffuse/specular 分量用于可视化

**shading/pbr/material.py**
- `init_material_texture(res) → nn.Parameter [1,H,W,5]`
- `decode_material(raw) → (base_color, roughness, metallic)`
- `compute_F0(base_color, metallic) → F0`

**shading/pbr/env_map.py**
- `init_env_map(h, w, init=None) → nn.Parameter`
- `direction_to_equirect(dirs) → (u, v)`
- `prefilter_env_map(env_map, n_levels) → Tensor`
- `sample_prefiltered(prefiltered, dirs, roughness) → color`

**shading/pbr/brdf_lut.py**
- `generate_brdf_lut(size=256) → Tensor [size,size,2]`
- `sample_brdf(lut, NdotV, roughness) → (scale, bias)`

### 4.4 Config 扩展

```python
@dataclass
class PBRConfig:
    env_map_res: tuple = (64, 128)
    n_mip_levels: int = 5
    brdf_lut_size: int = 256
    env_lr_ratio: float = 1.0
    env_tv_weight: float = 0.001
    init_env_map: Optional[str] = None

@dataclass
class Config:
    render_mode: str = "sh"  # "sh" | "pbr"
    pbr: PBRConfig = field(default_factory=PBRConfig)
    # ... 现有字段不变
```

### 4.5 main.py 分发

```python
model = create_shading_model(cfg.render_mode, cfg)
trainer = Trainer(cfg, model)  # 通用训练器
```

## 5. Mesh 法线支持

### mesh.py 修改
- `MeshData` 新增 `normals` 和 `normal_idx` 字段
- OBJ: 解析 `vn` 和法线索引
- GLB: 提取 `mesh.primitives[0].normal`
- 无顶点法线时: 自动计算面法线 → 平均转顶点法线

### renderer.py 修改
- 新增 `interpolate_normals()` 方法
- 光栅化同时插值 UV、world_pos、normals
- 输出归一化 `normals [1, H, W, 3]`

## 6. 调试输出

### 训练过程
- Compare 图 (每 N epoch): 2×2 atlas — GT | Rendered / Diffuse | Specular
- 材质贴图导出: base_color.png, roughness.png, metallic.png, env_map.png

### 训练结束
- 视频: orbit.mp4 (Full), orbit_diffuse.mp4, orbit_specular.mp4

## 7. 导出产物

- `base_color.png` — [H,W,3] sRGB
- `roughness.png` — [H,W,1] 灰度
- `metallic.png` — [H,W,1] 灰度
- `env_map.hdr` — equirect 环境贴图
- `model.glb` — 带 PBR 材质的 glTF

## 8. 测试策略

### 新增测试
- `tests/test_pbr_material.py`: sigmoid decode roundtrip, F0 计算, 初始化
- `tests/test_env_map.py`: equirect 坐标变换 roundtrip, prefilter 输出形状, 梯度流
- `tests/test_brdf_lut.py`: LUT 生成, 采样边界条件
- `tests/test_pbr_model.py`: ShadingModel 协议合规, 前向形状, 梯度流
- `tests/test_sh_model.py`: SH 模型包装协议合规

### 修改测试
- `tests/test_mesh.py`: 法线加载
- `tests/test_renderer.py`: 法线插值
- `tests/test_trainer.py`: 接受 ShadingModel 的通用训练器

## 9. 实施计划概要

1. **阶段 1: 基础设施** — mesh 法线、ShadingModel 协议、config 扩展
2. **阶段 2: PBR 组件** — material.py, env_map.py, brdf_lut.py
3. **阶段 3: PBR 模型** — pbr_model.py 集成
4. **阶段 4: SH 包装** — sh_model.py 包装现有代码
5. **阶段 5: 训练器泛化** — trainer.py 重构
6. **阶段 6: 导出与可视化** — exporter, video, debug 输出
7. **阶段 7: 测试与验证** — 单元测试 + 头盔数据集对比实验
