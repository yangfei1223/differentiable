# Joint UV + Texture Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add joint UV + texture optimization to the PBR pipeline, using content-aware Symmetric Dirichlet energy to improve UV layout quality and rendering PSNR.

**Architecture:** UV coordinates become learnable parameters (sigmoid-constrained). Training alternates between Adam (texture + env map) and L-BFGS (UV coords). Symmetric Dirichlet energy weighted by per-triangle render loss regularizes UV distortion.

**Tech Stack:** PyTorch, nvdiffrast (dr.interpolate, dr.texture, dr.rasterize)

---

### Task 1: UVOptConfig + YAML config parsing

**Files:**
- Modify: `src/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing test for UVOptConfig parsing**

Add to `tests/test_config.py`:

```python
def test_uv_opt_config_defaults():
    from src.config import UVOptConfig
    cfg = UVOptConfig()
    assert cfg.enabled is False
    assert cfg.lr == 0.001
    assert cfg.tex_steps_per_uv == 5
    assert cfg.sym_dirichlet_weight == 0.01
    assert cfg.area_preserve_weight == 0.1
    assert cfg.lbfgs_max_iter == 20
    assert cfg.start_epoch == 100


def test_uv_opt_config_from_yaml(tmp_path):
    import yaml
    from src.config import load_config
    data = {
        "render_mode": "pbr",
        "uv_optimization": {
            "enabled": True,
            "lr": 0.002,
            "sym_dirichlet_weight": 0.05,
        },
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(data))
    cfg = load_config(str(p))
    assert cfg.uv_opt.enabled is True
    assert cfg.uv_opt.lr == 0.002
    assert cfg.uv_opt.sym_dirichlet_weight == 0.05
    assert cfg.uv_opt.area_preserve_weight == 0.1  # default
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py::test_uv_opt_config_defaults tests/test_config.py::test_uv_opt_config_from_yaml -v`
Expected: FAIL — `ImportError: cannot import name 'UVOptConfig'`

- [ ] **Step 3: Add UVOptConfig dataclass and parsing**

In `src/config.py`, add after `PBRConfig`:

```python
@dataclass
class UVOptConfig:
    enabled: bool = False
    lr: float = 0.001
    tex_steps_per_uv: int = 5
    sym_dirichlet_weight: float = 0.01
    area_preserve_weight: float = 0.1
    lbfgs_max_iter: int = 20
    start_epoch: int = 100
```

Add `uv_opt: UVOptConfig = field(default_factory=UVOptConfig)` to the `Config` dataclass.

In `load_config()`, add parsing:

```python
if "uv_optimization" in raw:
    cfg.uv_opt = UVOptConfig(**raw["uv_optimization"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py::test_uv_opt_config_defaults tests/test_config.py::test_uv_opt_config_from_yaml -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: add UVOptConfig dataclass and YAML parsing"
```

---

### Task 2: UVParameterizer (sigmoid encoding/decoding)

**Files:**
- Create: `src/uv/__init__.py`
- Create: `src/uv/param.py`
- Test: `tests/test_uv_param.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_uv_param.py`:

```python
import torch
import numpy as np
import pytest


def test_uv_param_sigmoid_roundtrip():
    """Sigmoid encode → decode should recover original UVs."""
    from src.uv.param import UVParameterizer
    uvs = np.random.rand(100, 2).astype(np.float32) * 0.8 + 0.1  # [0.1, 0.9]
    uv_idx = np.zeros((50, 3), dtype=np.int64)
    param = UVParameterizer(uvs, uv_idx)
    decoded = param.get_uvs()
    assert decoded.shape == (100, 2)
    # sigmoid range is (0, 1), initial decode should be close to original
    diff = (decoded - torch.from_numpy(uvs)).abs().max().item()
    assert diff < 0.02, f"Roundtrip error too large: {diff}"


def test_uv_param_requires_grad():
    """raw parameter should have requires_grad=True."""
    from src.uv.param import UVParameterizer
    uvs = np.ones((10, 2), dtype=np.float32) * 0.5
    uv_idx = np.zeros((5, 3), dtype=np.int64)
    param = UVParameterizer(uvs, uv_idx)
    assert param.raw.requires_grad is True


def test_uv_param_get_uv_idx():
    """uv_idx should be stored as int tensor."""
    from src.uv.param import UVParameterizer
    uvs = np.ones((10, 2), dtype=np.float32) * 0.5
    uv_idx = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    param = UVParameterizer(uvs, uv_idx)
    idx = param.get_uv_idx()
    assert idx.dtype == torch.int32
    assert (idx == torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)).all()


def test_uv_param_gradient_flows():
    """Gradient should flow from decoded UVs back to raw params."""
    from src.uv.param import UVParameterizer
    uvs = np.ones((10, 2), dtype=np.float32) * 0.5
    uv_idx = np.zeros((5, 3), dtype=np.int64)
    param = UVParameterizer(uvs, uv_idx)
    decoded = param.get_uvs()
    loss = decoded.sum()
    loss.backward()
    assert param.raw.grad is not None
    assert param.raw.grad.norm().item() > 0


def test_uv_param_output_range():
    """Decoded UVs should always be in (0, 1)."""
    from src.uv.param import UVParameterizer
    uvs = np.ones((10, 2), dtype=np.float32) * 0.5
    uv_idx = np.zeros((5, 3), dtype=np.int64)
    param = UVParameterizer(uvs, uv_idx)
    # Push raw to extreme values
    with torch.no_grad():
        param.raw.fill_(10.0)
    decoded = param.get_uvs()
    assert decoded.min().item() > 0.0
    assert decoded.max().item() < 1.0
    with torch.no_grad():
        param.raw.fill_(-10.0)
    decoded = param.get_uvs()
    assert decoded.min().item() > 0.0
    assert decoded.max().item() < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_uv_param.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.uv'`

- [ ] **Step 3: Create `src/uv/__init__.py`**

```python
"""UV 优化模块 — 可微 UV 坐标参数化与正则化。"""
```

- [ ] **Step 4: Create `src/uv/param.py`**

```python
"""UV 坐标参数化 — sigmoid 编码，保证输出在 (0, 1)。"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


def _inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """sigmoid 的逆函数: logit(x) = log(x / (1-x))。"""
    x = x.clamp(1e-6, 1 - 1e-6)
    return torch.log(x / (1 - x))


class UVParameterizer(nn.Module):
    """可微 UV 坐标参数化。

    用 sigmoid 解码保证 UV 坐标严格在 (0, 1) 内。
    内部存储 sigmoid 的逆（logit）作为优化参数。

    Args:
        initial_uvs: 初始 UV 坐标 [V, 2]，值在 [0, 1]。
        uv_idx: 面-UV 索引 [F, 3]，int64。
    """

    def __init__(self, initial_uvs: np.ndarray, uv_idx: np.ndarray):
        super().__init__()
        uvs_tensor = torch.from_numpy(initial_uvs.astype(np.float32))
        self.raw = nn.Parameter(_inverse_sigmoid(uvs_tensor))
        self.register_buffer(
            "_uv_idx",
            torch.from_numpy(uv_idx.astype(np.int32)),
        )

    def get_uvs(self) -> torch.Tensor:
        """返回解码后的 UV 坐标 [V, 2]，值在 (0, 1)。"""
        return torch.sigmoid(self.raw)

    def get_uv_idx(self) -> torch.Tensor:
        """返回 UV 索引 [F, 3]，int32。"""
        return self._uv_idx
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_uv_param.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/uv/__init__.py src/uv/param.py tests/test_uv_param.py
git commit -m "feat: UVParameterizer with sigmoid encoding"
```

---

### Task 3: Symmetric Dirichlet Loss + Area Preserve Loss

**Files:**
- Create: `src/uv/losses.py`
- Test: `tests/test_uv_losses.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_uv_losses.py`:

```python
import torch
import pytest


def test_sym_dirichlet_identity():
    """Identity UV→3D mapping (square triangle) should have minimal energy."""
    from src.uv.losses import SymDirichletLoss
    # Equilateral-ish triangle in UV and same in 3D → low distortion
    uv = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]], device="cuda")  # [3, 2]
    verts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0]], device="cuda")  # [3, 3]
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64, device="cuda")  # [1, 3]
    render_loss = torch.tensor([0.5], device="cuda")  # [1] per-triangle render loss

    loss_fn = SymDirichletLoss()
    loss = loss_fn(uv, verts, faces, render_loss)
    # Identity-ish mapping → energy close to minimum (2 + 2 = 4 for equilateral)
    assert loss.item() > 0
    assert loss.item() < 20.0, f"Identity mapping energy too high: {loss.item()}"


def test_sym_dirichlet_flip_penalty():
    """Flipped triangle should have high energy."""
    from src.uv.losses import SymDirichletLoss
    # Normal triangle in 3D
    verts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0]], device="cuda")
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64, device="cuda")
    render_loss = torch.tensor([1.0], device="cuda")

    # Flipped UV (vertex 1 and 2 swapped in UV space)
    uv_flipped = torch.tensor([[0.0, 0.0], [0.5, 0.866], [1.0, 0.0]], device="cuda")
    # Normal UV
    uv_normal = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]], device="cuda")

    loss_fn = SymDirichletLoss()
    loss_flipped = loss_fn(uv_flipped, verts, faces, render_loss)
    loss_normal = loss_fn(uv_normal, verts, faces, render_loss)
    assert loss_flipped.item() > loss_normal.item(), "Flipped UV should have higher energy"


def test_area_preserve_same_area():
    """Same area → zero loss."""
    from src.uv.losses import AreaPreserveLoss
    # UV triangle with area = 0.433 (equilateral with side 1)
    uv = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]], device="cuda")
    verts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0]], device="cuda")
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64, device="cuda")
    target_areas = torch.tensor([0.433], device="cuda")

    loss_fn = AreaPreserveLoss()
    loss = loss_fn(uv, verts, faces, target_areas)
    assert loss.item() < 0.01, f"Same area should give ~0 loss: {loss.item()}"


def test_area_preserve_gradient():
    """Loss should produce gradient w.r.t. UV coords."""
    from src.uv.losses import AreaPreserveLoss
    uv = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]], device="cuda", requires_grad=True)
    verts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0]], device="cuda")
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64, device="cuda")
    target_areas = torch.tensor([0.1], device="cuda")  # different from actual

    loss_fn = AreaPreserveLoss()
    loss = loss_fn(uv, verts, faces, target_areas)
    loss.backward()
    assert uv.grad is not None
    assert uv.grad.norm().item() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_uv_losses.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.uv.losses'`

- [ ] **Step 3: Create `src/uv/losses.py`**

```python
"""UV 正则化损失 — Symmetric Dirichlet + 面积保持。"""
from __future__ import annotations

import torch
import torch.nn as nn


def _triangle_uv_areas(uv: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """计算每个三角形的 UV 面积。

    Args:
        uv: UV 坐标 [V, 2]。
        faces: 三角面索引 [F, 3]。
    Returns:
        面积 [F]。
    """
    v0 = uv[faces[:, 0]]
    v1 = uv[faces[:, 1]]
    v2 = uv[faces[:, 2]]
    # 2D cross product
    cross = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - \
            (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
    return cross.abs() * 0.5


class SymDirichletLoss(nn.Module):
    """Content-Aware Symmetric Dirichlet Energy。

    对每个三角形计算 UV→3D Jacobian 的 Symmetric Dirichlet energy，
    用该三角形的渲染误差加权。渲染差的区域获得更强的正则化。

    E_sym = σ1² + σ2² + 1/σ1² + 1/σ2²
    L = Σ_tri  E_sym(tri) × render_loss(tri)
    """

    def forward(
        self,
        uv: torch.Tensor,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        per_tri_render_loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            uv: UV 坐标 [V, 2]。
            vertices: 3D 顶点 [V, 3]。
            faces: 三角面索引 [F, 3]。
            per_tri_render_loss: 每三角形渲染误差 [F]。
        Returns:
            标量 loss。
        """
        # UV 边
        uv0 = uv[faces[:, 0]]
        uv1 = uv[faces[:, 1]]
        uv2 = uv[faces[:, 2]]

        du1 = uv1[:, 0] - uv0[:, 0]
        dv1 = uv1[:, 1] - uv0[:, 1]
        du2 = uv2[:, 0] - uv0[:, 0]
        dv2 = uv2[:, 1] - uv0[:, 1]

        # 3D 边
        p0 = vertices[faces[:, 0]]
        p1 = vertices[faces[:, 1]]
        p2 = vertices[faces[:, 2]]
        e1 = p1 - p0  # [F, 3]
        e2 = p2 - p0  # [F, 3]

        # Jacobian: J = [e1, e2] @ inv([du1, dv1; du2, dv2])
        det = du1 * dv2 - du2 * dv1  # [F]
        # 数值保护：避免退化三角形
        det_safe = torch.where(det.abs() < 1e-8, torch.ones_like(det) * 1e-8, det)

        # inv = (1/det) * [dv2, -du2; -dv1, du1]
        inv_det = 1.0 / det_safe

        # J 的两列 [F, 3] each
        j_col0 = e1 * (dv2 * inv_det).unsqueeze(-1) + e2 * (-dv1 * inv_det).unsqueeze(-1)
        j_col1 = e1 * (-du2 * inv_det).unsqueeze(-1) + e2 * (du1 * inv_det).unsqueeze(-1)

        # J = [j_col0, j_col1], shape [F, 3, 2]
        J = torch.stack([j_col0, j_col1], dim=-1)  # [F, 3, 2]

        # SVD to get singular values
        # For a 3x2 matrix, use economy SVD
        # J^T J is [2, 2], eigenvalues are σ1², σ2²
        JtJ = torch.bmm(J.transpose(1, 2), J)  # [F, 2, 2]
        # Eigenvalues of JtJ = σ²
        # For 2x2 matrix: trace = σ1² + σ2², det = σ1² * σ2²
        trace = JtJ[:, 0, 0] + JtJ[:, 1, 1]  # σ1² + σ2²
        det_JtJ = JtJ[:, 0, 0] * JtJ[:, 1, 1] - JtJ[:, 0, 1] * JtJ[:, 1, 0]  # σ1² * σ2²

        # Symmetric Dirichlet: σ1² + σ2² + 1/σ1² + 1/σ2²
        # = trace + (σ1² + σ2²) / (σ1² * σ2²)
        # = trace + trace / det_JtJ
        det_JtJ_safe = torch.where(det_JtJ.abs() < 1e-10,
                                    torch.sign(det_JtJ) * 1e-10 + (det_JtJ.abs() < 1e-10).float() * 1e-10,
                                    det_JtJ)
        e_sym = trace + trace / det_JtJ_safe.abs()

        # 翻转惩罚：det(J) < 0 时加大能量
        det_J = det  # det of UV Jacobian inverse matrix
        flip_penalty = torch.where(det_J < 0, (det_J.abs() + 1.0) * 10.0, torch.zeros_like(det_J))
        e_sym = e_sym + flip_penalty

        # Content-aware 加权
        loss = (e_sym * per_tri_render_loss).mean()
        return loss


class AreaPreserveLoss(nn.Module):
    """面积保持正则化。

    约束每个三角形的 UV 面积接近目标面积（初始 UV 面积或按 3D 面积比例分配）。

    L = mean(|uv_area - target_area|)
    """

    def forward(
        self,
        uv: torch.Tensor,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        target_areas: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            uv: UV 坐标 [V, 2]。
            vertices: 3D 顶点 [V, 3]（未使用，保留接口一致性）。
            faces: 三角面索引 [F, 3]。
            target_areas: 目标 UV 面积 [F]。
        Returns:
            标量 loss。
        """
        current_areas = _triangle_uv_areas(uv, faces)
        return (current_areas - target_areas).abs().mean()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_uv_losses.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/uv/losses.py tests/test_uv_losses.py
git commit -m "feat: Symmetric Dirichlet loss and area preserve loss"
```

---

### Task 4: Renderer UV update method

**Files:**
- Modify: `src/renderer.py:57-61` (uvs storage)
- Modify: `src/renderer.py:105` (interpolate call)
- Test: `tests/test_renderer.py`

- [ ] **Step 1: Write failing test for set_uvs**

Add to `tests/test_renderer.py`:

```python
def test_renderer_set_uvs():
    """set_uvs should update UVs used for interpolation."""
    import torch
    from src.renderer import DifferentiableRenderer
    from src.camera import Camera
    import numpy as np

    # Simple triangle
    verts = torch.tensor([[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [0.5, 1.0, 0.5]], dtype=torch.float32)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    uvs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=torch.float32)
    uv_idx = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    normals = torch.tensor([[0, 0, 1.0], [0, 0, 1.0], [0, 0, 1.0]], dtype=torch.float32)
    normal_idx = faces.clone()

    renderer = DifferentiableRenderer(
        vertices=verts, faces=faces, uvs=uvs, uv_idx=uv_idx,
        normals=normals, normal_idx=normal_idx,
        resolution=64, device="cuda",
    )

    cam = Camera(
        position=np.array([0.5, 0.5, 2.0]),
        look_at=np.array([0.5, 0.5, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
        fov_deg=45.0, image_width=64, image_height=64,
    )

    # Get texc with original UVs
    _, texc1, *_ = renderer.rasterize_and_interpolate(cam)

    # Update UVs to shifted values
    new_uvs = torch.tensor([[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]], dtype=torch.float32)
    renderer.set_uvs(new_uvs.unsqueeze(0))

    _, texc2, *_ = renderer.rasterize_and_interpolate(cam)

    # texc should differ
    visible = texc1[..., 0] > 0
    if visible.any():
        diff = (texc1[visible] - texc2[visible]).abs().max().item()
        assert diff > 0.01, "UVs should change after set_uvs"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_renderer.py::test_renderer_set_uvs -v`
Expected: FAIL — `AttributeError: 'DifferentiableRenderer' object has no attribute 'set_uvs'`

- [ ] **Step 3: Add `set_uvs` method to renderer**

In `src/renderer.py`, add after `__init__` (after line 82):

```python
    def set_uvs(self, uvs: torch.Tensor) -> None:
        """更新 UV 坐标（用于 UV 优化）。

        Args:
            uvs: UV 坐标 [1, V, 2] 或 [V, 2]，值在 (0, 1)。
        """
        if uvs.dim() == 2:
            uvs = uvs.unsqueeze(0)
        self.uvs = uvs.to(self.device).float()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_renderer.py::test_renderer_set_uvs -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/renderer.py tests/test_renderer.py
git commit -m "feat: add set_uvs method to DifferentiableRenderer"
```

---

### Task 5: UVOptimizer (L-BFGS wrapper + alternating schedule)

**Files:**
- Create: `src/uv/optimizer.py`
- Test: `tests/test_uv_optimizer.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_uv_optimizer.py`:

```python
import torch
import numpy as np


def test_uv_optimizer_step():
    """UVOptimizer.step should reduce loss."""
    from src.uv.param import UVParameterizer
    from src.uv.optimizer import UVOptimizer

    uvs = np.ones((10, 2), dtype=np.float32) * 0.5
    uv_idx = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    param = UVParameterizer(uvs, uv_idx)
    optimizer = UVOptimizer(param, lr=0.1, max_iter=5)

    # Dummy closure that computes a loss
    def closure():
        optimizer.zero_grad()
        decoded = param.get_uvs()
        loss = ((decoded - 0.7) ** 2).sum()
        loss.backward()
        return loss

    loss_before = closure().item()
    optimizer.step(closure)
    loss_after = closure().item()
    assert loss_after < loss_before, f"Loss should decrease: {loss_before} -> {loss_after}"


def test_uv_optimizer_zero_grad():
    """zero_grad should clear gradients."""
    from src.uv.param import UVParameterizer
    from src.uv.optimizer import UVOptimizer

    uvs = np.ones((10, 2), dtype=np.float32) * 0.5
    uv_idx = np.zeros((5, 3), dtype=np.int64)
    param = UVParameterizer(uvs, uv_idx)
    optimizer = UVOptimizer(param, lr=0.1, max_iter=5)

    # Create some gradients
    decoded = param.get_uvs()
    decoded.sum().backward()
    assert param.raw.grad is not None

    optimizer.zero_grad()
    assert param.raw.grad is None or param.raw.grad.norm().item() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_uv_optimizer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.uv.optimizer'`

- [ ] **Step 3: Create `src/uv/optimizer.py`**

```python
"""UV 优化器 — L-BFGS 包装。"""
from __future__ import annotations

from typing import Callable

import torch

from src.uv.param import UVParameterizer


class UVOptimizer:
    """UV 坐标的 L-BFGS 优化器。

    包装 torch.optim.LBFGS，提供 zero_grad / step 接口。

    Args:
        uv_param: UVParameterizer 实例。
        lr: 学习率。
        max_iter: L-BFGS 每步最大迭代次数。
    """

    def __init__(self, uv_param: UVParameterizer, lr: float = 0.001, max_iter: int = 20):
        self.uv_param = uv_param
        self._optimizer = torch.optim.LBFGS(
            uv_param.parameters(),
            lr=lr,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

    def zero_grad(self) -> None:
        """清零梯度。"""
        self._optimizer.zero_grad()
        if self.uv_param.raw.grad is not None:
            self.uv_param.raw.grad.zero_()

    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """执行一步 L-BFGS 优化。

        Args:
            closure: 返回 loss 的闭包函数（需内部计算梯度）。

        Returns:
            最终 loss 值。
        """
        return self._optimizer.step(closure)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_uv_optimizer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/uv/optimizer.py tests/test_uv_optimizer.py
git commit -m "feat: UVOptimizer with L-BFGS wrapper"
```

---

### Task 6: Per-triangle render loss aggregation utility

**Files:**
- Create: `src/uv/aggregate.py`
- Test: `tests/test_uv_aggregate.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_uv_aggregate.py`:

```python
import torch


def test_per_triangle_render_loss_shape():
    """Output should be [F] tensor."""
    from src.uv.aggregate import per_triangle_render_loss
    # 2 faces, 4x4 image
    pixel_loss = torch.ones(1, 4, 4, 3, device="cuda")
    tri_ids = torch.zeros(1, 4, 4, dtype=torch.int64, device="cuda")
    tri_ids[0, 0:2, :] = 0
    tri_ids[0, 2:4, :] = 1
    mask = torch.ones(1, 4, 4, dtype=torch.bool, device="cuda")
    num_faces = 2

    result = per_triangle_render_loss(pixel_loss, tri_ids, mask, num_faces)
    assert result.shape == (2,)


def test_per_triangle_render_loss_values():
    """Each triangle should get mean of its pixels' loss."""
    from src.uv.aggregate import per_triangle_render_loss
    pixel_loss = torch.ones(1, 4, 4, 3, device="cuda")
    # Triangle 0 gets loss 0.5, triangle 1 gets loss 1.0
    pixel_loss[0, :2, :] = 0.5
    tri_ids = torch.zeros(1, 4, 4, dtype=torch.int64, device="cuda")
    tri_ids[0, 2:, :] = 1
    mask = torch.ones(1, 4, 4, dtype=torch.bool, device="cuda")
    num_faces = 2

    result = per_triangle_render_loss(pixel_loss, tri_ids, mask, num_faces)
    assert abs(result[0].item() - 0.5) < 0.01
    assert abs(result[1].item() - 1.0) < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_uv_aggregate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.uv.aggregate'`

- [ ] **Step 3: Create `src/uv/aggregate.py`**

```python
"""Per-triangle 渲染误差聚合。"""
from __future__ import annotations

import torch


def per_triangle_render_loss(
    pixel_loss: torch.Tensor,
    tri_ids: torch.Tensor,
    mask: torch.Tensor,
    num_faces: int,
) -> torch.Tensor:
    """将 per-pixel 渲染误差聚合为 per-triangle 平均误差。

    Args:
        pixel_loss: 每像素误差 [B, H, W, C]。
        tri_ids: 每像素所属三角形 ID [B, H, W]，int64。
            从 rast[..., 0] 提取并转换为整数。
        mask: 有效像素掩码 [B, H, W]，bool。
        num_faces: 总面数 F。

    Returns:
        每三角形平均误差 [F]。
    """
    # Per-pixel scalar loss (mean over C)
    scalar_loss = pixel_loss.mean(dim=-1)  # [B, H, W]

    # Flatten
    flat_loss = scalar_loss[mask]  # [N]
    flat_tri = tri_ids[mask]  # [N]

    # nvdiffrast rast triangle ID is normalized [0, 1]: tri_id = raw_id / num_faces
    # Convert back to integer: raw_id = round(tri_id * num_faces)
    flat_tri_idx = (flat_tri.float() * num_faces).long().clamp(0, num_faces - 1)

    # Scatter sum + count → mean
    tri_loss_sum = torch.zeros(num_faces, device=pixel_loss.device, dtype=pixel_loss.dtype)
    tri_count = torch.zeros(num_faces, device=pixel_loss.device, dtype=pixel_loss.dtype)

    tri_loss_sum.scatter_add_(0, flat_tri_idx, flat_loss)
    tri_count.scatter_add_(0, flat_tri_idx, torch.ones_like(flat_loss))

    # Avoid division by zero; unseen triangles get 0 loss
    tri_mean = torch.where(tri_count > 0, tri_loss_sum / tri_count, torch.zeros_like(tri_loss_sum))

    return tri_mean
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_uv_aggregate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/uv/aggregate.py tests/test_uv_aggregate.py
git commit -m "feat: per-triangle render loss aggregation"
```

---

### Task 7: Integrate UV optimization into Trainer

**Files:**
- Modify: `src/trainer.py`
- Modify: `configs/train_pbr.yaml`

This is the core integration task. The trainer needs to:
1. Initialize `UVParameterizer` when `uv_opt.enabled`
2. Alternate between Adam (texture) and L-BFGS (UV)
3. Compute UV regularization loss
4. Update renderer UVs after each L-BFGS step

- [ ] **Step 1: Add UV initialization to `Trainer.__init__`**

In `src/trainer.py`, after line 81 (`self.renderer = self._create_renderer(self.current_resolution)`), add:

```python
        # ---- 9. UV 优化 ----
        self.uv_param = None
        self.uv_optimizer = None
        if config.render_mode == "pbr" and config.uv_opt.enabled:
            from src.uv.param import UVParameterizer
            from src.uv.optimizer import UVOptimizer
            self.uv_param = UVParameterizer(mesh.uvs, mesh.uv_idx).to(self.device)
            self.uv_optimizer = UVOptimizer(
                self.uv_param,
                lr=config.uv_opt.lr,
                max_iter=config.uv_opt.lbfgs_max_iter,
            )
            # 计算初始目标 UV 面积（用于面积保持）
            from src.uv.losses import _triangle_uv_areas
            with torch.no_grad():
                init_uvs = self.uv_param.get_uvs()
                self._uv_target_areas = _triangle_uv_areas(init_uvs, self.uv_param.get_uv_idx())
```

- [ ] **Step 2: Add helper to sync UVs to renderer**

Add method to `Trainer`:

```python
    def _sync_uvs_to_renderer(self) -> None:
        """将优化后的 UV 坐标同步到渲染器。"""
        if self.uv_param is not None:
            new_uvs = self.uv_param.get_uvs().unsqueeze(0)  # [1, V, 2]
            self.renderer.set_uvs(new_uvs)
```

- [ ] **Step 3: Add UV optimization step method**

Add method to `Trainer`:

```python
    def _uv_optimization_step(self, indices: list[int]) -> float:
        """执行一步 UV L-BFGS 优化。

        Args:
            indices: 当前 batch 的视角索引。

        Returns:
            UV 正则化 loss 值。
        """
        from src.uv.losses import SymDirichletLoss, AreaPreserveLoss
        from src.uv.aggregate import per_triangle_render_loss

        sym_dirichlet = SymDirichletLoss()
        area_preserve = AreaPreserveLoss()
        cfg = self.config.uv_opt

        def closure():
            self.uv_optimizer.zero_grad()
            self._sync_uvs_to_renderer()

            total_loss = torch.tensor(0.0, device=self.device)
            all_pixel_loss = []
            all_tri_ids = []
            all_masks = []
            num_faces = self.faces.shape[0]

            for idx in indices:
                img_np, camera = self.dataset[idx]
                gt = torch.from_numpy(img_np).unsqueeze(0).to(self.device)
                gt_hw = gt.permute(0, 1, 2, 3)
                H, W = self.current_resolution, self.current_resolution
                gt_resized = F.interpolate(gt_hw, size=(H, W), mode="bilinear", align_corners=False)
                gt_resized = gt_resized.squeeze(0).permute(1, 2, 0).unsqueeze(0)
                gt_linear = gt_resized.clamp(0, 1).pow(2.2)

                rast, texc, wpos, inorm, vdir, tang, btang = self.renderer.rasterize_and_interpolate(camera)
                rendered, mask = self.model.shade(rast, texc, wpos, inorm, vdir, camera, self.current_resolution, tang, btang)
                rendered = rendered.flip(1)
                mask = mask.flip(1)

                # Per-pixel loss
                mask_f = mask.unsqueeze(-1).float()
                pixel_loss = (rendered - gt_linear).abs() * mask_f  # [1, H, W, 3]
                total_loss = total_loss + pixel_loss.sum() / (mask.sum() * 3 + 1e-8)

                all_pixel_loss.append(pixel_loss.detach())
                all_tri_ids.append((rast[0, :, :, 0] * num_faces).long().unsqueeze(0))
                all_masks.append(mask.bool())

            # Mean render loss
            total_loss = total_loss / len(indices)

            # UV regularization
            uv_coords = self.uv_param.get_uvs()
            verts_3d = self.vertices.squeeze(0)  # [V, 3]
            faces_64 = self.faces.long()  # [F, 3]

            # Aggregate per-triangle render loss from all views
            combined_pixel_loss = torch.cat(all_pixel_loss, dim=0)
            combined_tri_ids = torch.cat(all_tri_ids, dim=0)
            combined_masks = torch.cat(all_masks, dim=0)
            tri_render_loss = per_triangle_render_loss(combined_pixel_loss, combined_tri_ids, combined_masks, num_faces)

            # Symmetric Dirichlet
            sd_loss = sym_dirichlet(uv_coords, verts_3d, faces_64, tri_render_loss)
            total_loss = total_loss + cfg.sym_dirichlet_weight * sd_loss

            # Area preserve
            ap_loss = area_preserve(uv_coords, verts_3d, faces_64, self._uv_target_areas)
            total_loss = total_loss + cfg.area_preserve_weight * ap_loss

            total_loss.backward()
            return total_loss

        loss_val = self.uv_optimizer.step(closure)
        self._sync_uvs_to_renderer()
        return loss_val.item()
```

- [ ] **Step 4: Modify training loop for alternating optimization**

In the `train()` method, inside the `for epoch` loop, replace the single-batch training with alternating logic. Find the block starting at `for idx in indices:` (around line 246) and wrap it:

After the line `indices = random.sample(range(num_views), min(batch_size, num_views))` (line 243), replace the training block with:

```python
            epoch_loss = 0.0

            # 判断是否启用 UV 交替优化
            uv_active = (
                self.uv_param is not None
                and epoch >= self.config.uv_opt.start_epoch
            )

            if uv_active:
                # 交替优化：tex_steps_per_uv 步 Adam (纹理)，然后 1 步 L-BFGS (UV)
                for _ in range(self.config.uv_opt.tex_steps_per_uv):
                    for idx in indices:
                        self.optimizer.zero_grad()

                        img_np, camera = self.dataset[idx]
                        gt = torch.from_numpy(img_np).unsqueeze(0).to(self.device)

                        self._sync_uvs_to_renderer()
                        rast, texc, wpos, interp_normals, vdirs, tangents, bitangents = self.renderer.rasterize_and_interpolate(camera)
                        rendered, mask = self.model.shade(rast, texc, wpos, interp_normals, vdirs, camera, self.current_resolution, tangents, bitangents)

                        rendered = rendered.flip(1)
                        mask = mask.flip(1)

                        gt_hw = gt.permute(0, 1, 2, 3)
                        H, W = rendered.shape[1], rendered.shape[2]
                        gt_resized = F.interpolate(gt_hw, size=(H, W), mode="bilinear", align_corners=False)
                        gt_resized = gt_resized.squeeze(0).permute(1, 2, 0).unsqueeze(0)
                        gt_linear = gt_resized.clamp(0, 1).pow(2.2)

                        tex_for_loss = self.model.get_material_texture().to(self.device)
                        loss = self.criterion(rendered, gt_linear, mask, tex_for_loss)

                        if self.config.render_mode == "pbr":
                            from src.losses import tv_loss
                            env_tv = tv_loss(self.model.env_map.raw) * self.config.pbr.env_tv_weight
                            env_decoded = self.model.env_map.decode()
                            env_l2 = (env_decoded ** 2).mean() * self.config.pbr.env_l2_weight
                            loss = loss + env_tv + env_l2

                        loss.backward()
                        self.optimizer.step()
                        epoch_loss += loss.item()

                # UV optimization step (L-BFGS)
                uv_loss = self._uv_optimization_step(indices)
                epoch_loss += uv_loss
            else:
                # 原始训练逻辑（无 UV 优化）
                for idx in indices:
                    self.optimizer.zero_grad()

                    img_np, camera = self.dataset[idx]
                    gt = torch.from_numpy(img_np).unsqueeze(0).to(self.device)

                    if self.config.render_mode == "sh":
                        rendered, mask, _ = self.renderer.render(
                            self.model.features_dc, self.model.features_rest, camera,
                        )
                    else:
                        rast, texc, wpos, interp_normals, vdirs, tangents, bitangents = self.renderer.rasterize_and_interpolate(camera)
                        rendered, mask = self.model.shade(rast, texc, wpos, interp_normals, vdirs, camera, self.current_resolution, tangents, bitangents)

                    rendered = rendered.flip(1)
                    mask = mask.flip(1)

                    gt_hw = gt.permute(0, 1, 2, 3)
                    H, W = rendered.shape[1], rendered.shape[2]
                    gt_resized = F.interpolate(gt_hw, size=(H, W), mode="bilinear", align_corners=False)
                    gt_resized = gt_resized.squeeze(0).permute(1, 2, 0).unsqueeze(0)
                    gt_linear = gt_resized.clamp(0, 1).pow(2.2)

                    tex_for_loss = self.model.get_material_texture().to(self.device)
                    loss = self.criterion(rendered, gt_linear, mask, tex_for_loss)

                    if self.config.render_mode == "pbr":
                        from src.losses import tv_loss
                        env_tv = tv_loss(self.model.env_map.raw) * self.config.pbr.env_tv_weight
                        env_decoded = self.model.env_map.decode()
                        env_l2 = (env_decoded ** 2).mean() * self.config.pbr.env_l2_weight
                        loss = loss + env_tv + env_l2

                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
```

- [ ] **Step 5: Add checkpoint save/load for UV params**

In the checkpoint saving section (inside `_export_debug` or wherever checkpoints are saved), add UV param saving. In `trainer.py`, find the checkpoint save block and ensure UV state is included. Add after the existing `ckpt_path = self.logger.save_checkpoint(...)`:

```python
                # Save UV params if active
                if self.uv_param is not None:
                    uv_ckpt = {
                        "uv_raw": self.uv_param.raw.detach().cpu(),
                        "uv_target_areas": self._uv_target_areas.cpu(),
                        "epoch": epoch + 1,
                    }
                    torch.save(uv_ckpt, os.path.join(ep_dir, "uv_params.pt"))
```

In the resume section, add after `self._rebuild_optimizer()`:

```python
            # Resume UV params
            if self.uv_param is not None:
                uv_ckpt_path = os.path.join(os.path.dirname(resume_from), "uv_params.pt")
                if os.path.exists(uv_ckpt_path):
                    uv_ckpt = torch.load(uv_ckpt_path, map_location=self.device)
                    self.uv_param.raw.data.copy_(uv_ckpt["uv_raw"].to(self.device))
                    self._uv_target_areas = uv_ckpt["uv_target_areas"].to(self.device)
                    print(f"[Resume] Loaded UV params from epoch {uv_ckpt.get('epoch', '?')}")
```

- [ ] **Step 6: Add UV optimization config to `configs/train_pbr.yaml`**

Append to `configs/train_pbr.yaml`:

```yaml
uv_optimization:
  enabled: false  # Set to true to enable
  lr: 0.001
  tex_steps_per_uv: 5
  sym_dirichlet_weight: 0.01
  area_preserve_weight: 0.1
  lbfgs_max_iter: 20
  start_epoch: 100
```

- [ ] **Step 7: Run existing tests to verify no regressions**

Run: `python -m pytest tests/ -v --timeout=60`
Expected: All existing tests PASS (UV optimization is off by default, no behavior change)

- [ ] **Step 8: Commit**

```bash
git add src/trainer.py configs/train_pbr.yaml
git commit -m "feat: integrate UV optimization into Trainer with alternating schedule"
```

---

### Task 8: Add UV optimization training config for piano

**Files:**
- Create: `configs/train_pbr_piano_uvopt.yaml`

- [ ] **Step 1: Create config**

Create `configs/train_pbr_piano_uvopt.yaml`:

```yaml
render_mode: pbr

data:
  mesh_path: data/piano_260604/scene/lowpoly.glb
  gt_dir: data/piano_260604/gt
  camera_path: data/piano_260604/cameras.json

texture:
  base_resolution: 512
  target_resolution: 2048

training:
  num_epochs: 2000
  lr: 0.01
  lr_decay: 0.5
  lr_decay_epochs: [500, 1000, 1500]
  batch_size: 4
  resolution_schedule:
    - epoch: 0
      resolution: 512
    - epoch: 300
      resolution: 1024
    - epoch: 700
      resolution: 2048

loss:
  lambda_l1: 1.0
  lambda_ssim: 0.2
  lambda_tv: 0.005

seam_padding:
  dilation_radius: 3
  apply_every_n_epochs: 50

pbr:
  env_map_res: [256, 512]
  brdf_lut_size: 256
  env_lr_ratio: 1.0
  env_tv_weight: 0.0005
  env_l2_weight: 0.0001

uv_optimization:
  enabled: true
  lr: 0.001
  tex_steps_per_uv: 5
  sym_dirichlet_weight: 0.01
  area_preserve_weight: 0.1
  lbfgs_max_iter: 20
  start_epoch: 100
```

- [ ] **Step 2: Commit**

```bash
git add configs/train_pbr_piano_uvopt.yaml
git commit -m "feat: add piano PBR + UV optimization training config"
```

---

### Task 9: Validation smoke test

**Files:**
- Modify: `configs/quick_test.yaml`

- [ ] **Step 1: Add UV opt section to quick_test.yaml**

Add at end of `configs/quick_test.yaml`:

```yaml
uv_optimization:
  enabled: true
  lr: 0.001
  tex_steps_per_uv: 3
  sym_dirichlet_weight: 0.01
  area_preserve_weight: 0.1
  lbfgs_max_iter: 5
  start_epoch: 2
```

- [ ] **Step 2: Run quick test**

Run: `$env:PYTHONPATH="."; python main.py --config configs/quick_test.yaml --mode train`
Expected: Training completes without error. UV optimization activates at epoch 2. No CUDA errors.

- [ ] **Step 3: Verify UV params saved in checkpoint**

Check that `uv_params.pt` exists in the output checkpoint directory.

- [ ] **Step 4: Commit**

```bash
git add configs/quick_test.yaml
git commit -m "test: enable UV optimization in quick_test config"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** Each requirement in the design spec maps to a task:
  - UVOptConfig → Task 1
  - UVParameterizer → Task 2
  - SymDirichletLoss + AreaPreserveLoss → Task 3
  - Renderer set_uvs → Task 4
  - UVOptimizer → Task 5
  - Per-triangle aggregation → Task 6
  - Trainer integration → Task 7
  - Piano config → Task 8
  - Validation → Task 9
- [x] **Placeholder scan:** No TBD, TODO, or vague steps.
- [x] **Type consistency:** `get_uvs()` returns `[V, 2]`, `get_uv_idx()` returns `int32 [F, 3]`, `set_uvs()` accepts `[1, V, 2]` or `[V, 2]` — consistent across all tasks.
