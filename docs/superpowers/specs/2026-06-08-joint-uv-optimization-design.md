# Joint UV + Texture Optimization Design

**Date**: 2026-06-08
**Branch**: `feature/uv-optimization`
**Reference**: Knodt et al., "Joint UV Optimization and Texture Baking", SIGGRAPH Asia 2023

## Problem

钢琴 mesh 有 177K 顶点 / 99K 面，UV 利用率仅 39.7%，且 UV 岛极度碎片化。纹理优化受限于 UV 布局质量，PBR 仅 21.41 dB（vs 头盔 21.97 dB）。

## Goal

在现有 PBR split-sum 管线基础上，联合优化 UV 坐标 + 纹理贴图，通过 content-aware 重分配纹理面积提升渲染质量。

## Scope

- **仅 PBR 着色模型**（SH 管线不动）
- **Per-vertex UV 坐标优化**（不做 seam 重切，保持 chart 拓扑）
- **交替优化**：L-BFGS 更新 UV，Adam 更新纹理
- **Content-Aware Symmetric Dirichlet** 正则化

## Architecture

### 可学习参数

| 参数 | 形状 | 优化器 | 说明 |
|------|------|--------|------|
| `mat_texture` | `[1, H, W, 8]` | Adam | PBR 材质贴图（已有） |
| `env_map.raw` | `[1, H_e, W_e, 3]` | Adam | 环境贴图（已有） |
| `uv_params` | `[V, 2]` | L-BFGS | UV 坐标（新增） |

`uv_params` 通过 sigmoid 解码到 [0,1]：`uv = sigmoid(uv_params)`

### 数据流

```
uv_params ──sigmoid──► uv_coords [V, 2]
                              │
Mesh vertices ──rasterize──► rast [1, H, W, 4]
                              │
uv_coords + rast + uv_idx ──dr.interpolate──► texc [1, H, W, 2]
                                                   │
mat_texture + texc ──dr.texture──► material params
                                        │
material + env_map + lighting ──PBR shade──► rgb [1, H, W, 3]
                                                 │
                                            rgb vs GT ──► loss
```

梯度回传路径：
```
loss → ∂loss/∂rgb → ∂loss/∂material → ∂loss/∂texc
                                           │
                              ┌────────────┴────────────┐
                              │                         │
                     ∂texc/∂mat_texture          ∂texc/∂uv_coords
                     (Adam 更新纹理)              (L-BFGS 更新 UV)
```

### 交替优化

```
for epoch in epochs:
    for step in range(tex_steps_per_uv):       # 默认 5
        # Phase 1: 纹理 + 环境贴图 (Adam)
        render → L_render + L_env_reg
        update mat_texture, env_map (Adam)
    
    # Phase 2: UV 坐标 (L-BFGS)
    render → L_render
    compute L_uv = L_render + λ * L_sym_dirichlet(L_render_per_tri)
    update uv_params (L-BFGS, max_iter=20)
```

## UV Regularization: Content-Aware Symmetric Dirichlet

### 核心思想

对每个三角形计算 UV→3D Jacobian 的 Symmetric Dirichlet energy，用该三角形的渲染误差加权。渲染差的区域获得更强的正则化梯度→分配更多 UV 面积。

### Per-Triangle Jacobian

对于三角形 (v0, v1, v2)，UV 坐标 (uv0, uv1, uv2)，3D 坐标 (p0, p1, p2)：

```
du1 = uv1[0] - uv0[0]    dv1 = uv1[1] - uv0[1]
du2 = uv2[0] - uv0[0]    dv2 = uv2[1] - uv0[1]

e1 = p1 - p0              e2 = p2 - p0

J = [e1, e2] @ inv([du1, dv1; du2, dv2])    # [3, 2] Jacobian
```

### Symmetric Dirichlet Energy

```
# SVD of J: J = U @ diag(σ1, σ2) @ V^T
σ1, σ2 = singular values of J

E_sym = σ1² + σ2² + 1/σ1² + 1/σ2²

# 翻转检测：det(J) < 0 时 E_sym → ∞
det_J = σ1 * σ2
if det_J < 0: E_sym 加惩罚
```

### Content-Aware 加权

```
# 每三角形的渲染误差（从 rasterizer 的 per-pixel loss 聚合到 per-triangle）
L_render_tri = mean(pixel_loss for pixels covered by triangle tri)

# Content-Aware Symmetric Dirichlet
L_uv = Σ_tri  E_sym(tri) × L_render_tri(tri)
```

### 面积保持

```
# 防止 UV 均匀缩到一点
target_area_tri = 3d_area_tri / total_3d_area  # 按比例
current_area_tri = uv_area_tri
L_area = Σ |current_area_tri - target_area_tri|
```

## 新增文件

```
src/uv/
├── __init__.py          # 公共接口
├── param.py             # UVParameterizer — sigmoid 解码, clamp
├── losses.py            # SymDirichletLoss, AreaPreserveLoss
└── optimizer.py         # UVOptimizer — L-BFGS 包装, 交替调度
```

### `src/uv/param.py` — UVParameterizer

```python
class UVParameterizer(nn.Module):
    """管理 UV 坐标的可优化参数。"""
    
    def __init__(self, initial_uvs: np.ndarray, uv_idx: np.ndarray):
        # initial_uvs: [V, 2] 原始 UV 坐标 [0,1]
        # 存储 inverse_sigmoid 作为参数
        self.raw = nn.Parameter(inverse_sigmoid(initial_uvs))
        self.uv_idx = uv_idx  # [F, 3]
    
    def get_uvs(self) -> torch.Tensor:
        """返回 sigmoid 解码后的 UV [V, 2] ∈ [0, 1]。"""
        return torch.sigmoid(self.raw)
```

### `src/uv/losses.py` — UV 正则化损失

```python
class SymDirichletLoss(nn.Module):
    """Content-Aware Symmetric Dirichlet Energy。"""
    
    def forward(self, uv_coords, vertices, faces, per_tri_render_loss):
        # 1. 计算 per-triangle Jacobian
        # 2. 计算 Symmetric Dirichlet energy
        # 3. 用 per_tri_render_loss 加权
        # 返回 scalar loss

class AreaPreserveLoss(nn.Module):
    """面积保持正则化。"""
    
    def forward(self, uv_coords, vertices, faces, initial_uv_areas):
        # 每三角形 UV 面积 vs 目标面积
        # 返回 scalar loss
```

### `src/uv/optimizer.py` — UV 优化器

```python
class UVOptimizer:
    """管理 UV 坐标的 L-BFGS 优化。"""
    
    def __init__(self, uv_param, lr, max_iter, sym_dirichlet_weight, area_weight):
        self.lbfgs = torch.optim.LBFGS(uv_param.parameters(), lr=lr, max_iter=max_iter)
    
    def step(self, closure):
        """执行一步 L-BFGS 优化。"""
        self.lbfgs.step(closure)
```

## 修改文件

### `src/renderer.py`

- `__init__`：`uvs` 参数改为可选，运行时可更新
- 新增 `set_uvs(uv_coords, uv_idx)` 方法
- `rasterize_and_interpolate`：使用当前 UV 坐标

### `src/trainer.py`

- 新增 `uv_param: UVParameterizer` 初始化
- 训练循环改为交替优化：
  - `tex_steps_per_uv` 步 Adam（纹理 + 环境贴图）
  - 1 步 L-BFGS（UV 坐标）
- UV 正则化 loss 集成到 L-BFGS closure 中

### `src/config.py`

- 新增 `UVOptConfig` dataclass：
  - `enabled: bool = False`
  - `lr: float = 0.001`
  - `tex_steps_per_uv: int = 5`
  - `sym_dirichlet_weight: float = 0.01`
  - `area_preserve_weight: float = 0.1`
  - `lbfgs_max_iter: int = 20`
  - `start_epoch: int = 100`（先跑纹理再解锁 UV）

### `configs/train_pbr.yaml`

```yaml
uv_optimization:
  enabled: true
  lr: 0.001
  tex_steps_per_uv: 5
  sym_dirichlet_weight: 0.01
  area_preserve_weight: 0.1
  lbfgs_max_iter: 20
  start_epoch: 100
```

### `src/mesh.py`

- 无修改（`MeshData.uvs` 仍为初始值，优化由 `UVParameterizer` 管理）

## 验证指标

| 指标 | 当前 PBR | 目标 |
|------|---------|------|
| 钢琴 PSNR | 21.41 dB | ≥ 22.0 dB |
| 头盔 PSNR | 21.97 dB | 不退化 |
| UV 利用率 | 39.7% | ≥ 60% |
| 训练时间 | ~10 min | ≤ 20 min |

## 测试计划

1. `tests/test_uv_param.py` — UVParameterizer sigmoid 编解码
2. `tests/test_uv_losses.py` — SymDirichletLoss 正确性（已知 Jacobian 的三角形）
3. `tests/test_uv_optimizer.py` — L-BFGS 步骤执行
4. 集成测试：`quick_test.yaml` + UV 优化开关

## 风险

1. **Per-triangle render loss 聚合**：从 per-pixel loss 映射到 per-triangle 需要额外计算。方案：用 rast 的 triangle ID 做 scatter mean。
2. **177K 顶点的 L-BFGS**：L-BFGS 对 355K 维参数（177K × 2）内存占用可控（history buffer 有限），但 closure 需要多次 forward。设 `max_iter=20` 控制开销。
3. **UV 翻转检测**：SymDirichlet 在 det(J)=0 处奇点。用 `max(det, ε)` 做数值保护。
