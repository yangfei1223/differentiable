# Neural Lightmap (NLM) 着色模型设计文档

**日期**：2026-06-15
**状态**：已确认，待编写实现计划
**关联文档**：
- [神经光照贴图 POC 技术方案](../../神经光照贴图%20(Neural%20Lightmap)%20POC%20技术方案文档.md)
- [神经光照贴图烘焙方案图](../../神经光照贴图烘焙方案.png)

## 1. 目标与范围

### 1.1 目标

在现有可微烘焙管线上新增第三个着色模型 `NeuralLightmapShadingModel`（render_mode = `"nlm"`），与 SH/PBR 平级共存。通过神经特征图 + TinyMLP 的组合，学习视角相关的光照积分（包含高光、GI、阴影的隐式表达），作为 PBR split-sum 近似的替代方案。

### 1.2 范围

- **数据**：复用现有 Blender PNG sRGB 数据（头盔 + 钢琴），不新建 EXR 管线
- **场景**：头盔（单 submesh）+ 钢琴（6 submesh），与现有 PBR 对齐
- **前向兼容**：SH/PBR 现有训练流程、配置、输出目录、checkpoint 格式全部不受影响
- **POC 阶段**：验证「光栅化 → 特征采样 → MLP 解码 → 反向传播」闭环连通性 + 高光拟合能力

### 1.3 非目标（POC 阶段）

- 不做 EXR / Linear HDR 数据管线（特征图输出虽用 Softplus 允许 >1.0，但 GT 被 clamp 在 [0,1]）
- 不做运行时引擎集成（Fragment Shader / UBO / ONNX 导出）
- 不做特征纹理量化压缩（fp16/int8）
- 不做多 submesh 之外的 multi-mesh 新场景支持

## 2. 整体架构

### 2.1 三着色模型平级共存

```
render_mode: "sh" | "pbr" | "nlm"
create_shading_model("nlm", config) → NeuralLightmapShadingModel
```

### 2.2 基础设施复用清单

| 模块 | 复用程度 | 说明 |
|------|---------|------|
| `dataset.py` / `camera.py` | 100% | GT PNG sRGB 加载、相机参数 |
| `mesh.py` / `gltf_loader.py` | 100% | 单 mesh / multi-mesh 几何加载 |
| `renderer.py` | 100% | nvdiffrast 光栅化 + 属性插值 |
| `losses.py` | 100% | L1 + SSIM + TV |
| `seam_padding.py` | 100% | UV 边界膨胀 |
| `config.py` | 改 | 新增 `NeuralLightmapConfig` |
| `trainer.py` | 改 | `_train_step_multi_pbr` → `_train_step_multi` 通用化 |
| `shading/base.py` | 改 | 新增 3 个可选钩子方法 |
| `shading/__init__.py` / `shading/logger.py` | 改 | 工厂加 `"nlm"` 分支 |
| `video.py` | 改 | dispatch 加 `"nlm"` |

### 2.3 数据流（单帧，multi-mesh 时外层 for submesh）

```
GT image (PNG sRGB)
    ↓ pow(2.2) → linear, clamp [0,1]
nvdiffrast rast → (rast_out, texc, world_pos, normals, view_dirs)
    ↓
shade_submesh(name, rast_out, texc, ..., view_dirs, ...):
    1. feature = dr.texture(feature_tex[name], texc)       # [1,H,W,12]
    2. mask = (rast_out[..., 3] > 0)                        # [1,H,W] bool
    3. [mask 索引] feat_valid = feature[mask]               # [N,12]
                    view_valid = view_dirs[mask]            # [N,3]
    4. view_pe = γ(view_valid, L=2)                         # [N,15]
    5. x = cat(feat_valid, view_pe, dim=-1)                 # [N,27]
    6. rgb_valid = Softplus(TinyMLP(x))                     # [N,3]  允许 >1.0
    7. rgb = zeros[1,H,W,3]; rgb[mask] = rgb_valid          # scatter 回
    ↓
L1 + SSIM loss vs GT_linear + TV on feature_tex[name]
    ↓ backward (梯度累积 per submesh)
```

### 2.4 渲染方程对应关系

文档定义的渲染方程：

$$L_o(p, \omega_o) = \mathcal{M}_\theta\Big( \mathcal{T}\big(\Phi(p)\big), \gamma(\omega_o) \Big)$$

| 数学符号 | 代码实现 |
|---------|---------|
| $\Phi(p) = (u,v)$ | `texc` (nvdiffrast 插值的 UV) |
| $\mathcal{T}$ | `self.feature_maps[name]` (`nn.Parameter [1,res,res,12]`) |
| $\gamma(\omega_o)$ | `positional_encode(view_dirs, L=2)` → 15D |
| $\mathcal{M}_\theta$ | `self.mlp` (`TinyMLP`: 27→32→32→3→Softplus) |

## 3. NeuralLightmapShadingModel 详细设计

### 3.1 模块文件结构

```
src/shading/
├── nlm_model.py              # NeuralLightmapShadingModel（主类）
├── nlm_logger.py             # NLM 调试输出
└── nlm/
    ├── __init__.py
    ├── feature_map.py         # FeatureMap 初始化 + 采样工具
    ├── positional_encode.py   # γ(d) 视角位置编码
    └── tiny_mlp.py            # TinyMLP nn.Module
```

### 3.2 FeatureMap（per-submesh 可学习特征纹理）

- **存储**：`dict[str, nn.Parameter]`，每个 submesh 一张 `[1, res, res, C=12]`
- **初始化**：`randn × feature_init_std`（默认 std=0.1）
- **采样**：`dr.texture(feature, texc, filter_mode="linear", boundary_mode="clamp")`
- **结构**：与 PBR `mat_textures` 字典完全一致，便于复用 trainer 通用路径

### 3.3 Positional Encoding（无参数）

$$\gamma(\mathbf{d}) = \Big[\ \mathbf{d},\ \sin(2^0 \pi \mathbf{d}),\ \cos(2^0 \pi \mathbf{d}),\ \sin(2^1 \pi \mathbf{d}),\ \cos(2^1 \pi \mathbf{d})\ \Big]$$

- 输入：归一化视角方向 `[N, 3]`
- 输出：`[N, 15]`（L=2 时）
- 纯函数，无学习参数

### 3.4 TinyMLP（全局共享解码器）

```python
nn.Sequential(
    nn.Linear(27, 32), nn.ReLU(inplace=True),
    nn.Linear(32, 32),  nn.ReLU(inplace=True),
    nn.Linear(32, 3),   nn.Softplus()       # 输出 ≥0，允许 HDR
)
```

- **参数量**：~3K
- **共享性**：所有 submesh 共用一个 MLP（学"如何解码"，不学"具体是什么"）
- **Softplus 输出**：保证物理辐射度非负，允许 >1.0 的 HDR 高光

### 3.5 shade_submesh() 实现（掩码索引）

```python
def shade_submesh(self, name, rast_out, texc, world_pos,
                  normals, view_dirs, camera, resolution, ...):
    import nvdiffrast.torch as dr

    # 1. UV 采样特征
    feature = dr.texture(self.feature_maps[name], texc,
                         filter_mode="linear", boundary_mode="clamp")  # [1,H,W,12]

    # 2. 掩码
    mask = (rast_out[..., 3] > 0)                                        # [1,H,W] bool
    if mask.sum() == 0:
        rgb = torch.zeros(1, resolution, resolution, 3, device=self.device)
        return rgb, mask.float()

    # 3. 仅有效像素前向（节省 ~80% FLOPs）
    feat_valid = feature[mask]                                           # [N,12]
    view_valid = view_dirs[mask]                                         # [N,3]

    # 4. 视角 PE
    view_pe = positional_encode(view_valid, self.pe_level)               # [N,15]

    # 5. 拼接 + MLP 前向
    x = torch.cat([feat_valid, view_pe], dim=-1)                         # [N,27]
    rgb_valid = self.mlp(x)                                              # [N,3] ≥0

    # 6. scatter 回完整图像
    rgb = torch.zeros(1, resolution, resolution, 3, device=self.device)
    rgb[mask] = rgb_valid

    self._last_debug = {"feature": feature.detach(), "view_pe": view_pe.detach()}
    return rgb, mask.float()
```

**设计要点**：返回的 `rgb` 仍是 `[1,H,W,3]`，背景为 0，trainer 的 L1/SSIM 计算通过 `pixel_mask` 屏蔽背景——与 PBR 接口完全一致，trainer 无需感知"内部用了掩码索引"。

### 3.6 shade()（单 mesh 路径）

调用 `shade_submesh("__default__", ...)`。NLM 在单 mesh 时也走字典路径（只有一个 `"__default__"` 键），保持代码统一。

### 3.7 TTUR 双学习率

`parameters()` 返回顺序：

```python
def parameters(self) -> list[nn.Parameter]:
    feat_params = list(self.feature_maps.values())   # 学习率 1e-1
    mlp_params  = list(self.mlp.parameters())         # 学习率 1e-3
    return feat_params + mlp_params
```

**为什么需要 TTUR**：特征图 texel 只被部分视角像素 hit（局部梯度，更新慢），MLP 每个有效像素都贡献梯度（全局梯度，更新快）。相同学习率会导致 MLP 过拟合 + 特征图欠拟合。

### 3.8 state_dict / load_state_dict

```python
def state_dict(self) -> dict:
    return {
        "render_mode": "nlm",
        "is_multi": True,
        "feature_maps": {k: v.data.detach().cpu() for k, v in self.feature_maps.items()},
        "mlp_state": self.mlp.state_dict(),
        "resolution": self.resolution,  # 当前特征图分辨率，便于 resume
    }

def load_state_dict(self, state: dict) -> None:
    self.is_multi = state.get("is_multi", True)
    self.resolution = state.get("resolution", self.config.texture.base_resolution)
    self.feature_maps = {
        k: nn.Parameter(v.to(self.device)) for k, v in state["feature_maps"].items()
    }
    self.mlp.load_state_dict(state["mlp_state"])
```

### 3.9 export()

```python
def export(self, output_dir: str) -> list[str]:
    """导出特征图（PNG 可视化 + PT 张量）和 MLP 权重。"""
    # 每个 submesh：
    #   feature_map_{name}.png  — 前 3 通道 RGB 可视化（debug 用）
    #   feature_map_{name}.pt   — 完整 12 通道 float32 张量
    # mlp_weights.pt            — TinyMLP state_dict
```

### 3.10 钩子方法实现

```python
def regularization_loss(self) -> torch.Tensor:
    """NLM 没有全局正则（特征图 TV 在 submesh 循环里）。"""
    return torch.tensor(0.0, device=self.device)

def get_submesh_texture(self, name: str) -> torch.Tensor:
    """返回指定 submesh 的特征图（用于 TV loss）。"""
    return self.feature_maps[name]

def post_backward_hook(self) -> None:
    """NLM 无需后向清理（无法线冻结、无 env_map）。"""
    pass
```

## 4. 配置设计

### 4.1 NeuralLightmapConfig

```python
@dataclass
class NeuralLightmapConfig:
    feature_dim: int = 12              # 特征维度 C
    pe_level: int = 2                  # 视角 PE 阶数 L（→ 15D）
    mlp_hidden_dim: int = 32           # MLP 隐藏层宽度
    feature_lr: float = 0.1            # 特征纹理学习率（TTUR 大值）
    mlp_lr: float = 0.001              # MLP 学习率（TTUR 小值）
    feature_tv_weight: float = 0.00001 # 特征图 TV 正则（文档建议 1e-5）
    feature_init_std: float = 0.1      # 初始化标准差
```

### 4.2 Config 集成

```python
@dataclass
class Config:
    render_mode: str = "sh"  # "sh" | "pbr" | "nlm"
    pbr: PBRConfig = field(default_factory=PBRConfig)
    nlm: NeuralLightmapConfig = field(default_factory=NeuralLightmapConfig)  # 新增
    ...

def load_config(path):
    ...
    if "nlm" in raw:
        cfg.nlm = NeuralLightmapConfig(**raw["nlm"])
```

### 4.3 配置文件示例

```yaml
# configs/train_nlm_helmet.yaml
render_mode: nlm
data:
  mesh_path: data/helmet_260604/scene/original_with_mats.glb
  ...
nlm:
  feature_dim: 12
  pe_level: 2
  feature_lr: 0.1
  mlp_lr: 0.001
  feature_tv_weight: 0.00001
training:
  num_epochs: 2000
  resolution_schedule:
    - {epoch: 0, resolution: 512}
    - {epoch: 300, resolution: 1024}
    - {epoch: 700, resolution: 2048}
```

## 5. Trainer 通用化改造

### 5.1 目标

把 `_train_step_multi_pbr()` 重构为 `_train_step_multi()`，PBR 和 NLM 共用同一条梯度累积路径。SH 走原 single-mesh 路径，完全不受影响。

### 5.2 ShadingModel 基类钩子

在 `base.py` 新增 3 个可选方法：

```python
class ShadingModel:
    ...
    def regularization_loss(self) -> torch.Tensor:
        """全局正则损失（如 PBR 的 env_map TV/L2）。默认 0。"""
        return torch.tensor(0.0)

    def get_submesh_texture(self, name: str) -> torch.Tensor:
        """返回指定 submesh 的可优化纹理（用于 TV loss）。"""
        raise NotImplementedError

    def post_backward_hook(self) -> None:
        """backward 后的清理钩子（如 PBR 冻结法线梯度）。默认无操作。"""
        pass
```

PBR 实现这些钩子（env 正则 + mat_textures + 法线冻结），NLM 实现特征图版本，SH 不需要（走原路径）。

### 5.3 _train_step_multi 通用化

```python
def _train_step_multi(self, camera, gt: torch.Tensor) -> float:
    # Phase 1: GT prep（不变）
    gt_linear = ...

    # Phase 2: 全局正则（通用钩子，PBR 返回 env TV/L2，NLM 返回 0）
    reg_loss = self.model.regularization_loss()
    if reg_loss.requires_grad:
        reg_loss.backward()
    total_loss = reg_loss.item()

    # Phase 3: 逐 submesh 梯度累积
    for k, sub_name in enumerate(self.submesh_names):
        sub_mask = (ownership == k).float()
        if sub_mask.sum() < 1: continue

        rast, texc, wpos, inorm, vdir, tang, btang = (
            self.renderers[sub_name].rasterize_and_interpolate(camera)
        )
        rgb_sub, _ = self.model.shade_submesh(
            sub_name, rast, texc, wpos, inorm, vdir, camera, res, tang, btang
        )
        rgb_sub = rgb_sub.flip(1)

        pixel_mask = (sub_mask * mask).unsqueeze(-1)
        l1 = ((rgb_sub - gt_linear).abs() * pixel_mask).sum() / n_valid
        sub_rendered_full = rgb_sub * pixel_mask + gt_linear * (1 - pixel_mask)
        ssim = ssim_loss(...)
        tv = tv_loss(self.model.get_submesh_texture(sub_name))  # 通用

        loss = lambda_l1 * l1 + lambda_ssim * ssim + lambda_tv * tv
        loss.backward()
        total_loss += loss.item()

    # Phase 4: NaN 清理（通用，遍历所有 params）
    for p in self.model.parameters():
        if p.grad is not None:
            p.grad = torch.nan_to_num(p.grad, nan=0.0)

    # Phase 5: 后向钩子（PBR 冻结法线，NLM 无操作）
    self.model.post_backward_hook()

    return total_loss
```

### 5.4 optimizer 构建

```python
# trainer.py __init__
if config.render_mode == "nlm":
    feat_params = list(model.feature_maps.values())
    mlp_params = list(model.mlp.parameters())
    self.optimizer = Adam([
        {"params": feat_params, "lr": config.nlm.feature_lr},
        {"params": mlp_params, "lr": config.nlm.mlp_lr},
    ])
else:
    self.optimizer = Adam(model.parameters(), lr=config.training.lr)
```

lr_scheduler 适配参数组（NLM 两个 lr 独立按 decay 比例衰减）。

`_rebuild_optimizer()`（分辨率切换后重建）同样需要按上述逻辑构建，保持参数组 lr 不变。

### 5.5 dispatch 改造

```python
# 训练步 dispatch
if self.is_multi and self.config.render_mode in ("pbr", "nlm"):
    step_loss = self._train_step_multi(camera, gt)
else:
    step_loss = self._train_step_single(camera, gt)
```

### 5.6 分辨率切换适配

NLM 没有法线烘焙，但需要把特征图 resize 到新分辨率：

```python
# trainer.py 分辨率切换
if self.config.render_mode == "nlm":
    for name in self.submesh_names:
        old_feat = self.model.feature_maps[name]
        new_feat = F.interpolate(
            old_feat.permute(0, 3, 1, 2),
            size=(new_res, new_res), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        self.model.feature_maps[name] = nn.Parameter(new_feat.contiguous())
    # MLP 权重无需 resize
```

### 5.7 SH 兼容性保证

SH 走 `_train_step_single()` 原路径，所有 SH 专属代码（`features_dc`/`features_rest` 拆分、SH logger）完全不动。dispatch 通过 `is_multi + render_mode in ("pbr","nlm")` 路由，SH 永远不会进入 `_train_step_multi`。

## 6. Logger / Video 适配

### 6.1 nlm_logger.py

复用 `pbr_logger.py` 的 `_export_compare_multi()` 模板：
- 输入 `model.shade_submesh(name, ...)` → 和 PBR 一样
- 输出 compare atlas（GT / NLM rendered / 特征图前3通道 / Residual）
- NLM 没有 diffuse/specular 分离，第三象限显示特征图可视化

### 6.2 video.py

`render_mode` dispatch 加 `"nlm"` 分支，走 multi-mesh 渲染路径（和 PBR multi 一样）。

### 6.3 logger 工厂

```python
# shading/logger.py
def create_logger(render_mode, config):
    if render_mode == "sh": ...
    elif render_mode == "pbr": ...
    elif render_mode == "nlm":
        from src.shading.nlm_logger import NLMLogger
        return NLMLogger(config)
```

## 7. 文件清单总览

### 7.1 新增文件

```
src/shading/nlm_model.py              # NeuralLightmapShadingModel 主类
src/shading/nlm_logger.py             # NLM 调试输出
src/shading/nlm/__init__.py
src/shading/nlm/feature_map.py        # FeatureMap 初始化工具
src/shading/nlm/positional_encode.py  # γ(d) PE 函数
src/shading/nlm/tiny_mlp.py           # TinyMLP nn.Module
configs/train_nlm_helmet.yaml         # 头盔 NLM 配置
configs/train_nlm_piano_multi.yaml    # 钢琴 multi NLM 配置
tests/test_nlm.py                     # NLM 单元测试
```

### 7.2 改动文件

| 文件 | 改动内容 | 估计行数 |
|------|---------|---------|
| `src/shading/__init__.py` | 加 `"nlm"` 分支 | +3 |
| `src/shading/logger.py` | 加 `"nlm"` 分支 | +3 |
| `src/shading/base.py` | 加 3 个钩子方法 | +15 |
| `src/config.py` | 加 `NeuralLightmapConfig` + 解析 | +20 |
| `src/trainer.py` | `_train_step_multi` 通用化 + NLM optimizer + dispatch | ~50 重构 |
| `src/video.py` | dispatch 加 `"nlm"` | +5 |

## 8. 测试计划

### 8.1 单元测试 (`tests/test_nlm.py`)

1. `TinyMLP` 前向 shape：输入 `[10, 27]` → 输出 `[10, 3]` 且 ≥0
2. `positional_encode(view_dirs, L=2)` 输出 shape `[N, 15]`
3. `init_feature_map(512, ["Object_0"])` 创建正确 shape `[1, 512, 512, 12]`
4. `NeuralLightmapShadingModel.shade_submesh()` 输出 `[1, H, W, 3]` + mask
5. `state_dict` / `load_state_dict` 往返一致性（save → load → 参数相等）
6. **梯度连通性**：随机输入 → shade → L1 loss → backward → feature_map 和 mlp 都有非零梯度

### 8.2 集成测试（手动）

1. **头盔**（单 submesh）跑 200 epochs，PSNR > 15 dB（架构连通性）
2. **钢琴**（6 submesh）跑 200 epochs，6 个 submesh 都收敛，无 NaN
3. **对比**：头盔 NLM vs PBR vs SH 的金属面罩高光质量（主观 + PSNR）
4. **TTUR 验证**：关闭 TTUR（统一 lr=0.01）跑同样轮数，对比收敛速度

## 9. 后续 ablation 路径（POC 验证不通过时）

如果头盔面罩高光不够锐利，按以下顺序递进强化编码：

| 级别 | 编码方案 | MLP 输入维度 | 物理动机 |
|------|---------|------------|---------|
| L0（默认） | `feature(12) ⊕ PE(V)(15)` | 27D | 文档原始方案 |
| L1 | `+ 原始法线 N(3)` | 30D | 让 MLP 感知几何 |
| L2 | `PE(R)(15) + NdotV(1)` 替换 `PE(V)` | 28D | 直接编码反射方向 + Fresnel |
| L3 | `feature_dim` 12 → 16/24 | 32~40D | 增加特征容量 |

每级 ablation 通过 `NeuralLightmapConfig` 字段切换，不改架构。

## 10. 导出资产格式

```
output/{scene}_nlm/epoch{N}/
├── feature_map_Object_0.png      # 前3通道可视化（debug 用）
├── feature_map_Object_0.pt       # 完整 12 通道 float32 张量
├── feature_map_Object_1.png
├── feature_map_Object_1.pt
├── ...
├── mlp_weights.pt                # TinyMLP state_dict
└── compare_XXXX.png              # GT vs NLM 对比图
```

运行时引擎加载 `.pt` 文件即可推理。特征纹理量化到 fp16/int8 留到后续优化阶段。

## 11. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 高光不够锐利（PE(V) 不足以建模镜面） | Section 9 的 ablation 路径（L1/L2/L3） |
| 特征图 TV 过强抹杀细节 | 文档建议 1e-5 起步，必要时降到 1e-6 或关闭 |
| MLP 过拟合（3K 参数对 N 像素） | TTUR 让特征图追赶 + TV 正则 |
| 坐标系不一致（Blender vs nvdiffrast） | 复用现有 renderer.py 已验证的坐标转换 |
| 分辨率切换特征图失真 | 双线性插值（与 PBR 材质纹理一致） |
