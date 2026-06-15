# Neural Lightmap (NLM) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `NeuralLightmapShadingModel` (render_mode="nlm") as a third shading model alongside SH/PBR, using per-submesh learnable feature maps + a shared TinyMLP decoder.

**Architecture:** Per-submesh learnable feature texture (12D) is sampled via UV, concatenated with high-frequency positional encoding of the view direction (15D, L=2), and decoded by a shared TinyMLP (27→32→32→3→Softplus) into RGB radiance. The trainer's existing per-submesh gradient accumulation path is generalized from PBR-only to a shared `_train_step_multi()` consumed by both PBR and NLM. SH remains on its original single-mesh path, untouched.

**Tech Stack:** PyTorch, nvdiffrast, NumPy, OpenCV, Pillow.

**Spec:** `docs/superpowers/specs/2026-06-15-neural-lightmap-design.md`

**Branch:** `feature/neural-lightmap` (already created; spec already committed at `2988cc0`)

**Environment:**
- Python: `C:\Users\yangfei\miniconda3\envs\differentiable\python.exe`
- Set env before running: `$env:PYTHONPATH="C:\Users\yangfei\Code\differentiable"` (PowerShell)
- Test command: `C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -m pytest tests/test_nlm.py -v`
- GPU: RTX 3080 10GB VRAM

---

## File Structure

### New files

| Path | Responsibility |
|------|----------------|
| `src/shading/nlm/__init__.py` | Package marker |
| `src/shading/nlm/positional_encode.py` | `positional_encode(d, L)` pure function |
| `src/shading/nlm/tiny_mlp.py` | `TinyMLP` nn.Module |
| `src/shading/nlm/feature_map.py` | `init_feature_map()` factory |
| `src/shading/nlm_model.py` | `NeuralLightmapShadingModel` main class |
| `src/shading/nlm_logger.py` | `NLMLogger` for compare atlas + checkpoint |
| `configs/train_nlm_helmet.yaml` | Helmet NLM training config |
| `configs/train_nlm_piano_multi.yaml` | Piano multi-mesh NLM training config |
| `tests/test_nlm.py` | Unit tests |

### Modified files

| Path | Changes |
|------|---------|
| `src/shading/base.py` | Add 3 optional hook methods |
| `src/shading/__init__.py` | Add `"nlm"` branch |
| `src/shading/logger.py` | Add `"nlm"` branch |
| `src/config.py` | Add `NeuralLightmapConfig` + parse logic |
| `src/trainer.py` | Generalize `_train_step_multi_pbr` → `_train_step_multi`; NLM optimizer; dispatch; resize |
| `src/video.py` | Add `"nlm"` to multi-mesh dispatch |

---

## Task 1: Positional Encoding Module

**Files:**
- Create: `src/shading/nlm/__init__.py`
- Create: `src/shading/nlm/positional_encode.py`
- Create: `tests/test_nlm.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_nlm.py`:

```python
"""Neural Lightmap unit tests."""
import torch
import pytest


def test_positional_encode_shape_and_range():
    """L=2 produces 15D output (3 raw + 4 freq bands * 3 dims)."""
    from src.shading.nlm.positional_encode import positional_encode

    d = torch.tensor([[[[0.3, -0.5, 0.8]]]])  # [1,1,1,3]
    out = positional_encode(d, level=2)
    assert out.shape == (1, 1, 1, 15)


def test_positional_encode_batch():
    """Works on flat batched input [N,3]."""
    from src.shading.nlm.positional_encode import positional_encode

    d = torch.randn(100, 3)
    out = positional_encode(d, level=2)
    assert out.shape == (100, 15)


def test_positional_encode_zero_input():
    """Zero input produces [0,0,0,sin=0...,cos=1...] — sin(0)=0, cos(0)=1."""
    from src.shading.nlm.positional_encode import positional_encode

    d = torch.zeros(1, 3)
    out = positional_encode(d, level=2)
    # First 3 channels: raw d = 0
    assert torch.allclose(out[0, :3], torch.zeros(3))
    # Next 3 channels: sin(2^0 * pi * 0) = sin(0) = 0
    assert torch.allclose(out[0, 3:6], torch.zeros(3), atol=1e-6)
    # Next 3 channels: cos(2^0 * pi * 0) = cos(0) = 1
    assert torch.allclose(out[0, 6:9], torch.ones(3), atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
$env:PYTHONPATH="C:\Users\yangfei\Code\differentiable"
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -m pytest tests/test_nlm.py::test_positional_encode_shape_and_range -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'src.shading.nlm'`

- [ ] **Step 3: Create package init**

Create `src/shading/nlm/__init__.py`:

```python
"""Neural Lightmap (NLM) submodules."""
```

- [ ] **Step 4: Write the implementation**

Create `src/shading/nlm/positional_encode.py`:

```python
"""Positional encoding for view directions (NeRF-style).

γ(d) = [d, sin(2^0 π d), cos(2^0 π d), ..., sin(2^{L-1} π d), cos(2^{L-1} π d)]

Output dim = 3 + 2*L*3 = 3*(1 + 2*L).
L=2 → 15D, L=3 → 21D.
"""
from __future__ import annotations

import torch


def positional_encode(d: torch.Tensor, level: int = 2) -> torch.Tensor:
    """Apply NeRF-style positional encoding to a direction vector.

    Args:
        d: direction tensor, last dim must be 3. Shape [..., 3].
        level: PE frequency levels L.

    Returns:
        Encoded tensor of shape [..., 3*(1 + 2*level)].
    """
    freqs = 2.0 ** torch.arange(level, device=d.device, dtype=d.dtype)  # [L]
    # Outer product: freqs × d → [L, ..., 3] via broadcast
    # We want sin(2^k * pi * d) for each k
    scaled = d.unsqueeze(-2) * (freqs * torch.pi).view(*([1] * (d.dim() - 1)), -1, 1)
    # scaled: [..., L, 3]
    sin = torch.sin(scaled)
    cos = torch.cos(scaled)
    # Flatten last two dims: [..., L*3] each
    sin_flat = sin.flatten(start_dim=-2)
    cos_flat = cos.flatten(start_dim=-2)
    return torch.cat([d, sin_flat, cos_flat], dim=-1)
```

- [ ] **Step 5: Run tests to verify they pass**

```powershell
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -m pytest tests/test_nlm.py -v
```
Expected: 3 PASS

- [ ] **Step 6: Commit**

```powershell
git add src/shading/nlm/__init__.py src/shading/nlm/positional_encode.py tests/test_nlm.py
git commit -m "feat(nlm): add positional encoding for view directions"
```

---

## Task 2: TinyMLP Module

**Files:**
- Create: `src/shading/nlm/tiny_mlp.py`
- Modify: `tests/test_nlm.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nlm.py`:

```python
def test_tiny_mlp_shape():
    """TinyMLP maps 27D → 3D."""
    from src.shading.nlm.tiny_mlp import TinyMLP

    mlp = TinyMLP(in_dim=27, hidden_dim=32, out_dim=3)
    x = torch.randn(10, 27)
    out = mlp(x)
    assert out.shape == (10, 3)


def test_tiny_mlp_non_negative_output():
    """Softplus output is non-negative (HDR radiance ≥ 0)."""
    from src.shading.nlm.tiny_mlp import TinyMLP

    mlp = TinyMLP(in_dim=27, hidden_dim=32, out_dim=3)
    x = torch.randn(100, 27) * 10  # extreme inputs
    out = mlp(x)
    assert (out >= 0).all(), "Softplus output must be non-negative"


def test_tiny_mlp_param_count():
    """~3K params (27*32 + 32 + 32*32 + 32 + 32*3 + 3 = 2019)."""
    from src.shading.nlm.tiny_mlp import TinyMLP

    mlp = TinyMLP(in_dim=27, hidden_dim=32, out_dim=3)
    n = sum(p.numel() for p in mlp.parameters())
    assert 1500 < n < 3500, f"Expected ~2K params, got {n}"
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -m pytest tests/test_nlm.py::test_tiny_mlp_shape -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'src.shading.nlm.tiny_mlp'`

- [ ] **Step 3: Write the implementation**

Create `src/shading/nlm/tiny_mlp.py`:

```python
"""TinyMLP — lightweight decoder for Neural Lightmap.

Maps (feature ⊕ view_pe) → RGB radiance. Output uses Softplus to allow
HDR values > 1.0 while remaining non-negative.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    """3-layer MLP with Softplus output for HDR radiance."""

    def __init__(self, in_dim: int = 27, hidden_dim: int = 32, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -m pytest tests/test_nlm.py -v
```
Expected: 6 PASS

- [ ] **Step 5: Commit**

```powershell
git add src/shading/nlm/tiny_mlp.py tests/test_nlm.py
git commit -m "feat(nlm): add TinyMLP decoder with Softplus HDR output"
```

---

## Task 3: Feature Map Factory

**Files:**
- Create: `src/shading/nlm/feature_map.py`
- Modify: `tests/test_nlm.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nlm.py`:

```python
def test_init_feature_map_shape():
    """init_feature_map returns [1, res, res, C] tensor."""
    from src.shading.nlm.feature_map import init_feature_map

    fm = init_feature_map(resolution=64, feature_dim=12, init_std=0.1)
    assert fm.shape == (1, 64, 64, 12)
    assert fm.dtype == torch.float32


def test_init_feature_map_std():
    """Init std approximately matches configured value."""
    from src.shading.nlm.feature_map import init_feature_map

    fm = init_feature_map(resolution=512, feature_dim=12, init_std=0.1)
    # randn * 0.1 → std ≈ 0.1
    assert 0.08 < fm.std().item() < 0.12
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -m pytest tests/test_nlm.py::test_init_feature_map_shape -v
```
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/shading/nlm/feature_map.py`:

```python
"""Feature map initialization for Neural Lightmap.

Each submesh owns a learnable feature texture [1, H, W, C] that implicitly
encodes albedo, normals, AO, and incoming radiance.
"""
from __future__ import annotations

import torch


def init_feature_map(
    resolution: int,
    feature_dim: int = 12,
    init_std: float = 0.1,
) -> torch.Tensor:
    """Create a randomly initialized feature texture.

    Args:
        resolution: texture H/W.
        feature_dim: feature channels C.
        init_std: initialization standard deviation (small to start near flat).

    Returns:
        Tensor [1, resolution, resolution, feature_dim], float32.
    """
    return torch.randn(1, resolution, resolution, feature_dim) * init_std
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -m pytest tests/test_nlm.py -v
```
Expected: 8 PASS

- [ ] **Step 5: Commit**

```powershell
git add src/shading/nlm/feature_map.py tests/test_nlm.py
git commit -m "feat(nlm): add feature map initialization"
```

---

## Task 4: NeuralLightmapConfig

**Files:**
- Modify: `src/config.py`
- Modify: `tests/test_nlm.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nlm.py`:

```python
def test_nlm_config_defaults():
    """NeuralLightmapConfig has correct defaults."""
    from src.config import NeuralLightmapConfig

    cfg = NeuralLightmapConfig()
    assert cfg.feature_dim == 12
    assert cfg.pe_level == 2
    assert cfg.mlp_hidden_dim == 32
    assert cfg.feature_lr == 0.1
    assert cfg.mlp_lr == 0.001
    assert cfg.feature_tv_weight == 0.00001
    assert cfg.feature_init_std == 0.1


def test_config_load_nlm_yaml(tmp_path):
    """YAML with 'nlm' section parses correctly."""
    from src.config import load_config

    yaml_content = """
render_mode: nlm
nlm:
  feature_dim: 16
  pe_level: 3
  feature_lr: 0.05
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_content, encoding="utf-8")

    cfg = load_config(str(p))
    assert cfg.render_mode == "nlm"
    assert cfg.nlm.feature_dim == 16
    assert cfg.nlm.pe_level == 3
    assert cfg.nlm.feature_lr == 0.05
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -m pytest tests/test_nlm.py::test_nlm_config_defaults -v
```
Expected: FAIL with `ImportError: cannot import name 'NeuralLightmapConfig'`

- [ ] **Step 3: Add NeuralLightmapConfig to config.py**

In `src/config.py`, after `PBRConfig` (after line 20), insert:

```python
@dataclass
class NeuralLightmapConfig:
    feature_dim: int = 12              # 特征维度 C
    pe_level: int = 2                  # 视角 PE 阶数 L（→ 15D）
    mlp_hidden_dim: int = 32           # MLP 隐藏层宽度
    feature_lr: float = 0.1            # 特征纹理学习率（TTUR 大值）
    mlp_lr: float = 0.001              # MLP 学习率（TTUR 小值）
    feature_tv_weight: float = 0.00001 # 特征图 TV 正则
    feature_init_std: float = 0.1      # 初始化标准差
```

In `Config` dataclass (around line 100-109), add after `pbr` field:

```python
    nlm: NeuralLightmapConfig = field(default_factory=NeuralLightmapConfig)
```

In `load_config()` function (around line 122-143), add after the `pbr` block:

```python
    if "nlm" in raw:
        cfg.nlm = NeuralLightmapConfig(**raw["nlm"])
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -m pytest tests/test_nlm.py -v
```
Expected: 10 PASS

- [ ] **Step 5: Commit**

```powershell
git add src/config.py tests/test_nlm.py
git commit -m "feat(nlm): add NeuralLightmapConfig dataclass"
```

---

## Task 5: ShadingModel Base Hooks

**Files:**
- Modify: `src/shading/base.py`

This task adds 3 optional hook methods to the base class so `_train_step_multi` can be model-agnostic. These are non-breaking additions.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nlm.py`:

```python
def test_base_hooks_exist_with_defaults():
    """ShadingModel base provides default hook implementations."""
    from src.shading.base import ShadingModel

    m = ShadingModel()
    # regularization_loss returns 0 tensor
    reg = m.regularization_loss()
    assert reg.item() == 0.0
    # get_submesh_texture raises (subclass must override)
    with pytest.raises(NotImplementedError):
        m.get_submesh_texture("test")
    # post_backward_hook returns None (no-op)
    assert m.post_backward_hook() is None
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -m pytest tests/test_nlm.py::test_base_hooks_exist_with_defaults -v
```
Expected: FAIL with `AttributeError: 'ShadingModel' object has no attribute 'regularization_loss'`

- [ ] **Step 3: Add hooks to base.py**

In `src/shading/base.py`, after the `load_state_dict` method (before the end of class, line 49), add:

```python

    # ------------------------------------------------------------------
    # Optional hooks for multi-mesh training (used by PBR and NLM)
    # ------------------------------------------------------------------
    def regularization_loss(self) -> "torch.Tensor":
        """Global regularization loss (e.g., PBR env_map TV/L2).

        Returns:
            Scalar tensor. Default 0 (no global regularization).
        """
        import torch
        return torch.tensor(0.0)

    def get_submesh_texture(self, name: str) -> "torch.Tensor":
        """Return the optimizable texture for the named submesh (for TV loss).

        Subclasses participating in multi-mesh training must override this.
        """
        raise NotImplementedError

    def post_backward_hook(self) -> None:
        """Cleanup hook called after backward() (e.g., PBR freezes normal grads).

        Default: no-op.
        """
        pass
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -m pytest tests/test_nlm.py -v
```
Expected: 11 PASS

- [ ] **Step 5: Verify SH and PBR still pass existing tests (regression check)**

```powershell
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -m pytest tests/ -v
```
Expected: All existing tests PASS (the new hooks are optional and have defaults)

- [ ] **Step 6: Commit**

```powershell
git add src/shading/base.py tests/test_nlm.py
git commit -m "feat(shading): add optional regularization/texture/post-backward hooks to base"
```

---

## Task 6: Implement Hooks on PBRShadingModel

**Files:**
- Modify: `src/shading/pbr_model.py`

PBR must implement the 3 hooks so it can flow through the generalized `_train_step_multi`. The hooks replicate PBR-specific behavior that was previously hardcoded in `_train_step_multi_pbr`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nlm.py`:

```python
def test_pbr_regularization_loss_shape():
    """PBR regularization_loss returns scalar tensor."""
    import torch
    from src.config import Config, PBRConfig
    from src.shading.pbr_model import PBRShadingModel

    cfg = Config()
    cfg.render_mode = "pbr"
    model = PBRShadingModel(cfg)
    # Need to init env_map for regularization to be computable
    from src.shading.pbr.env_map import EnvironmentMap
    model.env_map = EnvironmentMap(256, 512)
    reg = model.regularization_loss()
    assert reg.dim() == 0 or reg.numel() == 1


def test_pbr_get_submesh_texture():
    """PBR.get_submesh_texture returns the named mat_texture."""
    import torch.nn as nn
    from src.config import Config
    from src.shading.pbr_model import PBRShadingModel

    cfg = Config()
    cfg.render_mode = "pbr"
    model = PBRShadingModel(cfg)
    model.is_multi = True
    fake = nn.Parameter(torch.zeros(1, 4, 4, 8))
    model.mat_textures = {"Object_0": fake}
    out = model.get_submesh_texture("Object_0")
    assert out is fake


def test_pbr_post_backward_hook_noop():
    """PBR post_backward_hook runs without error when no frozen normals."""
    from src.config import Config
    from src.shading.pbr_model import PBRShadingModel

    cfg = Config()
    cfg.render_mode = "pbr"
    model = PBRShadingModel(cfg)
    # Should not raise
    model.post_backward_hook()
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -m pytest tests/test_nlm.py::test_pbr_regularization_loss_shape -v
```
Expected: FAIL with `AttributeError: 'PBRShadingModel' object has no attribute 'regularization_loss'`

- [ ] **Step 3: Add hook implementations to pbr_model.py**

In `src/shading/pbr_model.py`, at the end of the `PBRShadingModel` class (after `_export_multi` method, line 324), add:

```python

    # ------------------------------------------------------------------
    # Multi-mesh training hooks
    # ------------------------------------------------------------------
    def regularization_loss(self) -> torch.Tensor:
        """Env map TV + L2 regularization."""
        if self.env_map is None:
            return torch.tensor(0.0, device=self.device)
        from src.losses import tv_loss
        env_tv = tv_loss(self.env_map.raw) * self.config.pbr.env_tv_weight
        env_decoded = self.env_map.decode()
        env_l2 = (env_decoded ** 2).mean() * self.config.pbr.env_l2_weight
        return env_tv + env_l2

    def get_submesh_texture(self, name: str) -> torch.Tensor:
        """Return the named submesh's material texture (for TV loss)."""
        return self.mat_textures[name]

    def post_backward_hook(self) -> None:
        """Freeze normal map channel gradients (channels 5:8)."""
        if not self.config.pbr.disable_normal_map:
            return
        if self.is_multi:
            for tex in self.mat_textures.values():
                if tex.grad is not None:
                    tex.grad[..., 5:8].zero_()
        else:
            if self.mat_texture is not None and self.mat_texture.grad is not None:
                self.mat_texture.grad[..., 5:8].zero_()
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -m pytest tests/test_nlm.py -v
```
Expected: 14 PASS

- [ ] **Step 5: Commit**

```powershell
git add src/shading/pbr_model.py tests/test_nlm.py
git commit -m "feat(pbr): implement regularization/texture/post-backward hooks"
```

---

## Task 7: NeuralLightmapShadingModel — Core

**Files:**
- Create: `src/shading/nlm_model.py`
- Modify: `tests/test_nlm.py`

This is the main class. Implements init, parameters, shade_submesh with mask indexing, shade, get/set material texture, state dict, export, and the 3 hooks.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nlm.py`:

```python
def _make_nlm_model(feature_dim=12, pe_level=2, resolution=32, submesh_names=None):
    """Helper: create an initialized NLM model on CPU."""
    from src.config import Config
    from src.shading.nlm_model import NeuralLightmapShadingModel

    cfg = Config()
    cfg.render_mode = "nlm"
    cfg.nlm.feature_dim = feature_dim
    cfg.nlm.pe_level = pe_level
    model = NeuralLightmapShadingModel(cfg)
    if submesh_names is None:
        submesh_names = ["__default__"]
    model.init_textures(resolution, submesh_names=submesh_names)
    return model


def test_nlm_parameters_has_feature_and_mlp():
    """parameters() returns feature params first, then MLP params."""
    model = _make_nlm_model(resolution=16, submesh_names=["A", "B"])
    params = model.parameters()
    # 2 feature maps + N MLP params (TinyMLP has 6 weight/bias tensors)
    assert len(params) == 2 + 6


def test_nlm_state_dict_roundtrip():
    """state_dict → load_state_dict preserves parameters."""
    model = _make_nlm_model(resolution=16)
    state = model.state_dict()
    assert state["render_mode"] == "nlm"
    assert "feature_maps" in state
    assert "mlp_state" in state

    # Create new model and load
    model2 = _make_nlm_model(resolution=16)
    model2.load_state_dict(state)
    # Compare a feature map
    fm1 = model.feature_maps["__default__"]
    fm2 = model2.feature_maps["__default__"]
    assert torch.allclose(fm1, fm2)


def test_nlm_regularization_returns_zero():
    """NLM has no global regularization."""
    model = _make_nlm_model(resolution=8)
    reg = model.regularization_loss()
    assert reg.item() == 0.0


def test_nlm_get_submesh_texture():
    """get_submesh_texture returns the feature map."""
    model = _make_nlm_model(resolution=8, submesh_names=["Obj0"])
    tex = model.get_submesh_texture("Obj0")
    assert tex is model.feature_maps["Obj0"]


def test_nlm_post_backward_hook_noop():
    """NLM post_backward_hook is a no-op."""
    model = _make_nlm_model(resolution=8)
    assert model.post_backward_hook() is None
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -m pytest tests/test_nlm.py::test_nlm_parameters_has_feature_and_mlp -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'src.shading.nlm_model'`

- [ ] **Step 3: Write the implementation**

Create `src/shading/nlm_model.py`:

```python
"""Neural Lightmap shading model.

Per-submesh learnable feature map + shared TinyMLP decoder.
L_o(p, ω_o) = MLP( T(u,v), γ(ω_o) )
"""
from __future__ import annotations

import os
from typing import List

import torch
import torch.nn as nn

from src.config import Config
from src.shading.base import ShadingModel
from src.shading.nlm.feature_map import init_feature_map
from src.shading.nlm.tiny_mlp import TinyMLP
from src.shading.nlm.positional_encode import positional_encode


class NeuralLightmapShadingModel(ShadingModel):
    """Neural Lightmap shading model."""

    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        nlm = config.nlm
        self.feature_dim = nlm.feature_dim
        self.pe_level = nlm.pe_level
        self.pe_dim = 3 * (1 + 2 * nlm.pe_level)  # L=2 → 15
        self.mlp_in_dim = self.feature_dim + self.pe_dim  # 12 + 15 = 27

        self.feature_maps: dict[str, nn.Parameter] = {}
        self.mlp: TinyMLP = TinyMLP(
            in_dim=self.mlp_in_dim,
            hidden_dim=nlm.mlp_hidden_dim,
            out_dim=3,
        ).to(self.device)
        self.is_multi: bool = False
        self.resolution: int = config.texture.base_resolution

    # ------------------------------------------------------------------
    # Parameters & initialization
    # ------------------------------------------------------------------
    def parameters(self) -> list[nn.Parameter]:
        # TTUR: feature maps first (lr=1e-1), MLP params second (lr=1e-3)
        return list(self.feature_maps.values()) + list(self.mlp.parameters())

    def init_textures(self, resolution: int, submesh_names: list[str] | None = None) -> None:
        self.resolution = resolution
        nlm = self.config.nlm
        if submesh_names is not None:
            self.is_multi = True
            self.feature_maps = {
                name: nn.Parameter(
                    init_feature_map(resolution, nlm.feature_dim, nlm.feature_init_std).to(self.device)
                )
                for name in submesh_names
            }
        else:
            self.is_multi = False
            self.feature_maps = {
                "__default__": nn.Parameter(
                    init_feature_map(resolution, nlm.feature_dim, nlm.feature_init_std).to(self.device)
                )
            }

    # ------------------------------------------------------------------
    # Shading
    # ------------------------------------------------------------------
    def shade(
        self,
        rast_out: torch.Tensor,
        texc: torch.Tensor,
        world_pos: torch.Tensor,
        normals: torch.Tensor,
        view_dirs: torch.Tensor,
        camera,
        resolution: int,
        tangents: torch.Tensor | None = None,
        bitangents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-mesh shade (delegates to shade_submesh with __default__)."""
        return self.shade_submesh(
            "__default__", rast_out, texc, world_pos, normals, view_dirs,
            camera, resolution, tangents, bitangents,
        )

    def shade_submesh(
        self,
        name: str,
        rast_out: torch.Tensor,
        texc: torch.Tensor,
        world_pos: torch.Tensor,
        normals: torch.Tensor,
        view_dirs: torch.Tensor,
        camera,
        resolution: int,
        tangents: torch.Tensor | None = None,
        bitangents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shade a submesh: sample feature → PE(view) → MLP → scatter back."""
        import nvdiffrast.torch as dr

        H, W = resolution, resolution
        # 1. Sample feature texture via UV
        feature = dr.texture(
            self.feature_maps[name], texc, filter_mode="linear", boundary_mode="clamp"
        )  # [1, H, W, C]

        # 2. Mask valid pixels
        mask = (rast_out[..., 3] > 0)  # [1, H, W] bool
        if mask.sum() == 0:
            rgb = torch.zeros(1, H, W, 3, device=self.device)
            return rgb, mask.float()

        # 3. Index valid pixels only (save ~80% FLOPs on background)
        feat_valid = feature[mask]                     # [N, C]
        view_valid = view_dirs[mask]                   # [N, 3]

        # 4. Positional encode view direction
        view_pe = positional_encode(view_valid, self.pe_level)  # [N, pe_dim]

        # 5. Concat & decode
        x = torch.cat([feat_valid, view_pe], dim=-1)   # [N, C+pe_dim]
        rgb_valid = self.mlp(x)                        # [N, 3], Softplus ≥ 0

        # 6. Scatter back to full image
        rgb = torch.zeros(1, H, W, 3, device=self.device)
        rgb[mask] = rgb_valid

        # Debug info (feature first 3 channels as visualization proxy)
        self._last_debug = {"feature": feature.detach()}

        return rgb, mask.float()

    # ------------------------------------------------------------------
    # Material texture accessors (for resize / seam padding)
    # ------------------------------------------------------------------
    def get_material_texture(self) -> dict[str, torch.Tensor]:
        return {k: v.data.detach().cpu() for k, v in self.feature_maps.items()}

    def set_material_texture(self, texture) -> None:
        if isinstance(texture, dict):
            self.is_multi = True
            self.feature_maps = {
                k: nn.Parameter(v.to(self.device).contiguous()) for k, v in texture.items()
            }
        else:
            # Single tensor — wrap in default key
            self.is_multi = False
            self.feature_maps = {
                "__default__": nn.Parameter(texture.to(self.device).contiguous())
            }

    def get_debug_info(self) -> dict:
        return getattr(self, "_last_debug", {})

    # ------------------------------------------------------------------
    # Multi-mesh training hooks
    # ------------------------------------------------------------------
    def regularization_loss(self) -> torch.Tensor:
        """NLM has no global regularization (feature TV is per-submesh)."""
        return torch.tensor(0.0, device=self.device)

    def get_submesh_texture(self, name: str) -> torch.Tensor:
        return self.feature_maps[name]

    def post_backward_hook(self) -> None:
        """NLM requires no post-backward cleanup."""
        pass

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "render_mode": "nlm",
            "is_multi": self.is_multi,
            "feature_maps": {k: v.data.detach().cpu() for k, v in self.feature_maps.items()},
            "mlp_state": self.mlp.state_dict(),
            "resolution": self.resolution,
        }

    def load_state_dict(self, state: dict) -> None:
        self.is_multi = state.get("is_multi", True)
        self.resolution = state.get("resolution", self.config.texture.base_resolution)
        self.feature_maps = {
            k: nn.Parameter(v.to(self.device)) for k, v in state["feature_maps"].items()
        }
        if "mlp_state" in state:
            self.mlp.load_state_dict(state["mlp_state"])

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export(self, output_dir: str) -> list[str]:
        import numpy as np
        from PIL import Image

        os.makedirs(output_dir, exist_ok=True)
        paths = []

        # Feature map PNG (first 3 channels) + PT (full)
        for name, fm in self.feature_maps.items():
            sub_dir = output_dir if not self.is_multi else os.path.join(output_dir, name)
            if self.is_multi:
                os.makedirs(sub_dir, exist_ok=True)

            # PNG: first 3 channels normalized to [0,255]
            vis = fm[0, ..., :3].clamp(-1, 1)  # feature may be negative
            vis = ((vis + 1) * 0.5 * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            png_path = os.path.join(sub_dir, f"feature_map_{name}.png")
            Image.fromarray(vis, "RGB").save(png_path)
            paths.append(png_path)

            # PT: full float32 tensor
            pt_path = os.path.join(sub_dir, f"feature_map_{name}.pt")
            torch.save(fm.data.detach().cpu(), pt_path)
            paths.append(pt_path)

        # MLP weights
        mlp_path = os.path.join(output_dir, "mlp_weights.pt")
        torch.save(self.mlp.state_dict(), mlp_path)
        paths.append(mlp_path)

        return paths
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -m pytest tests/test_nlm.py -v
```
Expected: 19 PASS

- [ ] **Step 5: Commit**

```powershell
git add src/shading/nlm_model.py tests/test_nlm.py
git commit -m "feat(nlm): add NeuralLightmapShadingModel core"
```

---

## Task 8: Gradient Connectivity Test

**Files:**
- Modify: `tests/test_nlm.py`

This is the most important test: verifies that gradients flow from loss back to both feature maps AND MLP weights (the "closed loop" verification from the spec).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nlm.py`:

```python
def test_nlm_gradient_connectivity():
    """Verify gradient flows to BOTH feature map and MLP after backward().

    This is the core 'closed loop' verification: rasterize → sample → MLP
    → loss → backward must update both feature map and MLP weights.
    """
    torch.manual_seed(42)
    model = _make_nlm_model(feature_dim=12, pe_level=2, resolution=16)

    # Fake rasterization outputs
    rast_out = torch.zeros(1, 16, 16, 4, device=model.device)
    rast_out[..., 3] = 1.0  # all pixels valid
    rast_out[..., 2] = 0.5  # depth

    # Fake UV coords in [0,1]
    texc = torch.rand(1, 16, 16, 2, device=model.device)

    # Fake view directions (normalized)
    view_dirs = torch.randn(1, 16, 16, 3, device=model.device)
    view_dirs = view_dirs / view_dirs.norm(dim=-1, keepdim=True)

    # Forward
    rgb, mask = model.shade(rast_out, texc, torch.zeros_like(texc),
                            torch.zeros_like(texc), view_dirs, None, 16)

    # Synthetic target
    target = torch.ones_like(rgb) * 0.5
    loss = (rgb - target).abs().mean()
    loss.backward()

    # Check feature map grad
    fm = model.feature_maps["__default__"]
    assert fm.grad is not None, "Feature map grad is None"
    assert not torch.allclose(fm.grad, torch.zeros_like(fm.grad)), \
        "Feature map grad is all zero — gradient did not flow"

    # Check at least one MLP param grad
    mlp_grad_any = False
    for p in model.mlp.parameters():
        if p.grad is not None and not torch.allclose(p.grad, torch.zeros_like(p.grad)):
            mlp_grad_any = True
            break
    assert mlp_grad_any, "All MLP grads are zero — gradient did not flow to MLP"


def test_nlm_empty_mask_returns_zeros():
    """Empty mask (no valid pixels) returns zero rgb without error."""
    model = _make_nlm_model(resolution=8)

    rast_out = torch.zeros(1, 8, 8, 4, device=model.device)  # all background
    texc = torch.rand(1, 8, 8, 2, device=model.device)
    view_dirs = torch.zeros(1, 8, 8, 3, device=model.device)

    rgb, mask = model.shade(rast_out, texc, torch.zeros_like(texc),
                            torch.zeros_like(texc), view_dirs, None, 8)
    assert rgb.shape == (1, 8, 8, 3)
    assert (rgb == 0).all()
    assert (mask == 0).all()
```

- [ ] **Step 2: Run tests**

```powershell
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -m pytest tests/test_nlm.py::test_nlm_gradient_connectivity tests/test_nlm.py::test_nlm_empty_mask_returns_zeros -v
```
Expected: 2 PASS (both already implemented in Task 7)

If either FAILS, debug — this is the closed-loop verification.

- [ ] **Step 3: Commit**

```powershell
git add tests/test_nlm.py
git commit -m "test(nlm): add gradient connectivity + empty mask tests"
```

---

## Task 9: Shading Model Factory Registration

**Files:**
- Modify: `src/shading/__init__.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nlm.py`:

```python
def test_create_shading_model_nlm():
    """Factory creates NLM model for render_mode='nlm'."""
    from src.config import Config
    from src.shading import create_shading_model
    from src.shading.nlm_model import NeuralLightmapShadingModel

    cfg = Config()
    cfg.render_mode = "nlm"
    model = create_shading_model("nlm", cfg)
    assert isinstance(model, NeuralLightmapShadingModel)
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -m pytest tests/test_nlm.py::test_create_shading_model_nlm -v
```
Expected: FAIL with `ValueError: Unknown render_mode: 'nlm'`

- [ ] **Step 3: Modify the factory**

In `src/shading/__init__.py`, replace the entire contents with:

```python
"""着色模型可插拔层 — 工厂函数。"""
from __future__ import annotations

from src.config import Config


def create_shading_model(render_mode: str, config: Config):
    """根据 render_mode 创建对应的着色模型。"""
    if render_mode == "sh":
        from src.shading.sh_model import SHShadingModel
        return SHShadingModel(config)
    elif render_mode == "pbr":
        from src.shading.pbr_model import PBRShadingModel
        return PBRShadingModel(config)
    elif render_mode == "nlm":
        from src.shading.nlm_model import NeuralLightmapShadingModel
        return NeuralLightmapShadingModel(config)
    else:
        raise ValueError(f"Unknown render_mode: {render_mode!r}")
```

- [ ] **Step 4: Run tests**

```powershell
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -m pytest tests/test_nlm.py -v
```
Expected: 22 PASS

- [ ] **Step 5: Commit**

```powershell
git add src/shading/__init__.py tests/test_nlm.py
git commit -m "feat(shading): register 'nlm' render_mode in factory"
```

---

## Task 10: NLMLogger

**Files:**
- Create: `src/shading/nlm_logger.py`
- Modify: `src/shading/logger.py`

Logger handles checkpoint saving + compare image export + orbit video. NLM has no diffuse/specular split, so compare atlas shows: GT / Rendered / Feature[ch0:3] / Feature[ch3:6].

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nlm.py`:

```python
def test_create_logger_nlm():
    """Logger factory returns NLMLogger for render_mode='nlm'."""
    from src.config import Config
    from src.shading.logger import create_logger
    from src.shading.nlm_logger import NLMLogger

    cfg = Config()
    cfg.render_mode = "nlm"
    logger = create_logger("nlm", cfg)
    assert isinstance(logger, NLMLogger)
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -m pytest tests/test_nlm.py::test_create_logger_nlm -v
```
Expected: FAIL with `ValueError: Unknown render_mode: 'nlm'`

- [ ] **Step 3: Create NLMLogger**

Create `src/shading/nlm_logger.py`:

```python
"""Neural Lightmap 调试日志 — compare atlas + checkpoint + 视频。"""
from __future__ import annotations

import os

import numpy as np
import torch

from src.config import Config
from src.shading.logger import ShadingLogger


class NLMLogger(ShadingLogger):
    """NLM 着色模型日志。"""

    def save_checkpoint(
        self, model, output_dir: str, epoch: int, loss: float, resolution: int,
    ) -> str:
        ckpt = model.state_dict()
        ckpt["epoch"] = epoch
        ckpt["loss"] = loss
        ckpt["resolution"] = resolution
        path = os.path.join(output_dir, "nlm_checkpoint.pt")
        torch.save(ckpt, path)
        return path

    def export_debug(
        self, model, renderer, dataset, output_dir: str, epoch: int,
        history: dict, device: str, current_resolution: int,
        **kwargs,
    ) -> None:
        import cv2
        from src.mesh import load_mesh
        from src.video import render_video, render_video_multi

        # 1. Export feature maps + MLP weights
        model.export(output_dir)
        print(f"  [Debug] NLM feature maps + MLP weights → {output_dir}")

        is_multi = kwargs.get("is_multi", False)
        renderers = kwargs.get("renderers", None)
        submesh_names = kwargs.get("submesh_names", None)

        # 2. Compare images
        if is_multi and renderers is not None:
            self._export_compare_multi(
                model, renderers, submesh_names, dataset, output_dir, device, current_resolution
            )
        else:
            self._export_compare(model, renderer, dataset, output_dir, device, current_resolution)

        # 3. Orbit video
        mesh = load_mesh(self.config.data.mesh_path)
        cfg = self.config
        vk = dict(
            center=cfg.video.center, radius=cfg.video.radius,
            height=cfg.video.height, num_frames=cfg.video.num_frames,
            fov_deg=cfg.video.fov_deg, resolution=cfg.video.resolution, fps=cfg.video.fps,
        )
        if is_multi:
            render_video_multi(
                mesh=mesh, shading_model=model, submesh_names=submesh_names,
                output_path=os.path.join(output_dir, "orbit.mp4"), **vk,
            )
        else:
            render_video(
                mesh=mesh, shading_model=model,
                output_path=os.path.join(output_dir, "orbit.mp4"), **vk,
            )

        print(f"  [Debug] compare + video → {output_dir}")

    def _export_compare(self, model, renderer, dataset, output_dir, device, resolution):
        self._export_compare_impl(model, renderer, dataset, output_dir, device, resolution, is_multi=False)

    def _export_compare_multi(self, model, renderers, submesh_names, dataset, output_dir, device, resolution):
        self._export_compare_impl(model, renderers, dataset, output_dir, device, resolution,
                                  is_multi=True, submesh_names=submesh_names)

    def _export_compare_impl(
        self, model, renderer_or_renderers, dataset, output_dir, device, resolution,
        is_multi=False, submesh_names=None,
    ):
        import cv2

        num_views = len(dataset)
        indices = [int(i * num_views / min(4, num_views)) for i in range(min(4, num_views))]

        for ci, idx in enumerate(indices):
            img_np, camera = dataset[idx]

            with torch.no_grad():
                if is_multi:
                    rendered = torch.zeros(1, resolution, resolution, 3, device=device)
                    depth_buf = torch.full((1, resolution, resolution), float("inf"), device=device)
                    mask = torch.zeros(1, resolution, resolution, device=device)
                    for sub_name in submesh_names:
                        sub_renderer = renderer_or_renderers[sub_name]
                        rast, texc, wpos, inorm, vdir, tang, btang = sub_renderer.rasterize_and_interpolate(camera)
                        rgb_sub, mask_sub = model.shade_submesh(
                            sub_name, rast, texc, wpos, inorm, vdir, camera, resolution, tang, btang
                        )
                        sub_depth = rast[..., 2]
                        write = (mask_sub > 0.5) & (sub_depth < depth_buf)
                        rendered = torch.where(write.unsqueeze(-1), rgb_sub, rendered)
                        depth_buf = torch.where(write, sub_depth, depth_buf)
                        mask = torch.max(mask, mask_sub)
                else:
                    rast, texc, wpos, inorm, vdir, tang, btang = renderer_or_renderers.rasterize_and_interpolate(camera)
                    rendered, mask = model.shade(rast, texc, wpos, inorm, vdir, camera, resolution)

            # Feature visualization (first 3 channels of __default__ or first submesh)
            debug = model.get_debug_info()
            feature = debug.get("feature", rendered * 0)
            feat_vis = feature[..., :3].clamp(-1, 1)
            feat_vis = (feat_vis + 1) * 0.5  # to [0,1]

            mask = mask.flip(1)
            mask_np = mask[0].cpu().numpy()

            def to_bgr(t, gamma=True):
                img = t[0].flip(0).clamp(0, 1).detach().cpu().numpy()
                if gamma:
                    img = img ** (1 / 2.2)
                img = (img * 255).astype(np.uint8)
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                bgr[mask_np < 0.5] = 0
                return bgr

            gt = cv2.cvtColor((img_np.transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            panels = [
                (gt, "GT"),
                (to_bgr(rendered), "NLM"),
                (to_bgr(feat_vis, gamma=False), "Feature[0:3]"),
                (to_bgr(rendered * 0 + 0.5, gamma=False), "Residual"),
            ]

            th = min(p[0].shape[0] for p in panels)
            rs = []
            for img, label in panels:
                h, w = img.shape[:2]
                r = cv2.resize(img, (w * th // h, th))
                cv2.putText(r, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                rs.append(r)

            canvas = np.concatenate([
                np.concatenate([rs[0], rs[1]], axis=1),
                np.concatenate([rs[2], rs[3]], axis=1),
            ], axis=0)
            cv2.imwrite(os.path.join(output_dir, f"compare_{ci:04d}.png"), canvas)
```

- [ ] **Step 4: Register in logger factory**

In `src/shading/logger.py`, add the `"nlm"` branch (after the `"pbr"` branch):

```python
    elif render_mode == "nlm":
        from src.shading.nlm_logger import NLMLogger
        return NLMLogger(config)
```

- [ ] **Step 5: Run tests**

```powershell
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -m pytest tests/test_nlm.py -v
```
Expected: 23 PASS

- [ ] **Step 6: Commit**

```powershell
git add src/shading/nlm_logger.py src/shading/logger.py tests/test_nlm.py
git commit -m "feat(nlm): add NLMLogger for compare/checkpoint/video"
```

---

## Task 11: Video Dispatch for NLM

**Files:**
- Modify: `src/video.py`

The `render_video_multi` function is already generic (calls `shade_submesh`). We need to verify it works with NLM and add the dispatch if any render_mode-specific branching exists.

- [ ] **Step 1: Inspect video.py for any PBR-specific dispatch**

```powershell
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -c "import src.video; import inspect; print(inspect.getsource(src.video.render_video_multi))"
```

Review the function for any PBR-specific branches. If `render_video_multi` only calls `shade_submesh` generically (no `render_mode` check), no change is needed — skip to Step 4.

- [ ] **Step 2: If dispatch is PBR-only, generalize it**

If you find code like:

```python
if render_mode == "pbr":
    # multi path
```

Change to:

```python
if render_mode in ("pbr", "nlm"):
    # multi path
```

- [ ] **Step 3: Run all NLM tests**

```powershell
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -m pytest tests/test_nlm.py -v
```
Expected: 23 PASS

- [ ] **Step 4: Commit (if changes were made)**

```powershell
git add src/video.py
git commit -m "feat(nlm): generalize video dispatch to support nlm"
```

If no changes were needed, simply note "video.py already generic — no changes needed" and proceed to next task.

---

## Task 12: Generalize trainer.py — Refactor _train_step_multi_pbr → _train_step_multi

**Files:**
- Modify: `src/trainer.py`

This is the core refactor. The existing `_train_step_multi_pbr` is renamed to `_train_step_multi` and made model-agnostic by using the new hook methods. The training step dispatch is updated.

- [ ] **Step 1: Read current _train_step_multi_pbr**

```powershell
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -c "from src.trainer import Trainer; import inspect; print(inspect.getsource(Trainer._train_step_multi_pbr))"
```

Note the exact current implementation (it uses `self.model.env_map.raw` and `self.model.mat_textures[sub_name]` directly).

- [ ] **Step 2: Refactor the method**

In `src/trainer.py`, rename `_train_step_multi_pbr` to `_train_step_multi` and replace its body. The new body uses the hook methods:

```python
    def _train_step_multi(self, camera, gt: torch.Tensor) -> float:
        """Multi-mesh gradient accumulation step (PBR + NLM)."""
        from src.losses import ssim_loss, tv_loss

        res = self.current_resolution

        # Composite mask + ownership via depth buffer
        rendered = torch.zeros(1, res, res, 3, device=self.device)
        depth_buf = torch.full((1, res, res), float("inf"), device=self.device)
        mask = torch.zeros(1, res, res, device=self.device)
        ownership = torch.zeros(1, res, res, dtype=torch.long, device=self.device)

        for k, sub_name in enumerate(self.submesh_names):
            sub_renderer = self.renderers[sub_name]
            with torch.no_grad():
                rast, _, _, _, _, _, _ = sub_renderer.rasterize_and_interpolate(camera)
                sub_depth = rast[..., 2]
                sub_mask = (rast[..., 3] > 0).float()
                write = (sub_mask > 0.5) & (sub_depth < depth_buf)
                ownership = torch.where(write, torch.tensor(k, device=self.device), ownership)
                depth_buf = torch.where(write, sub_depth, depth_buf)
                mask = torch.max(mask, sub_mask)

        mask = mask.flip(1)
        ownership = ownership.flip(1)

        # GT prep
        gt_hw = gt.permute(0, 1, 2, 3)
        gt_resized = F.interpolate(
            gt_hw, size=(res, res), mode="bilinear", align_corners=False
        )
        gt_resized = gt_resized.squeeze(0).permute(1, 2, 0).unsqueeze(0)
        gt_linear = gt_resized.clamp(0, 1).pow(2.2)

        # Phase 2: Global regularization (PBR: env TV/L2; NLM: 0)
        reg_loss = self.model.regularization_loss()
        if reg_loss.requires_grad:
            reg_loss.backward()
        total_loss = reg_loss.item()

        # Phase 3: Per-submesh gradient accumulation
        n_valid = mask.sum() * 3 + 1e-8

        for k, sub_name in enumerate(self.submesh_names):
            sub_mask = (ownership == k).float()
            if sub_mask.sum() < 1:
                continue

            rast, texc, wpos, inorm, vdir, tang, btang = (
                self.renderers[sub_name].rasterize_and_interpolate(camera)
            )
            rgb_sub, _ = self.model.shade_submesh(
                sub_name, rast, texc, wpos, inorm, vdir, camera, res, tang, btang
            )
            rgb_sub = rgb_sub.flip(1)

            pixel_mask = (sub_mask * mask).unsqueeze(-1)

            # L1
            abs_diff = (rgb_sub - gt_linear).abs() * pixel_mask
            l1 = abs_diff.sum() / n_valid

            # SSIM
            sub_rendered_full = rgb_sub * pixel_mask + gt_linear * (1 - pixel_mask)
            rendered_chw = sub_rendered_full.permute(0, 3, 1, 2)
            gt_chw = gt_linear.permute(0, 3, 1, 2)
            ssim = ssim_loss(rendered_chw, gt_chw)

            # TV on submesh texture (PBR: mat_texture, NLM: feature_map)
            tv = tv_loss(self.model.get_submesh_texture(sub_name))
            # NLM uses its own TV weight if configured
            if self.config.render_mode == "nlm":
                tv = tv * (self.config.nlm.feature_tv_weight / max(self.config.loss.lambda_tv, 1e-12))
                loss = self.config.loss.lambda_l1 * l1 + self.config.loss.lambda_ssim * ssim + tv
            else:
                loss = (
                    self.config.loss.lambda_l1 * l1
                    + self.config.loss.lambda_ssim * ssim
                    + self.config.loss.lambda_tv * tv
                )

            loss.backward()
            total_loss += loss.item()

        # Phase 4: NaN cleanup (general, all params)
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad = torch.nan_to_num(p.grad, nan=0.0)

        # Phase 5: Post-backward hook (PBR freezes normals; NLM noop)
        self.model.post_backward_hook()

        return total_loss
```

**Important note on TV weight**: The existing `self.config.loss.lambda_tv` is shared across all models. For NLM, we want `feature_tv_weight` (default 1e-5) which is much smaller than PBR's `lambda_tv` (default 0.005). The code above scales NLM's TV by `feature_tv_weight / lambda_tv` to preserve the intended magnitude. This is a pragmatic solution that avoids restructuring the loss config. If `lambda_tv` is 0, fall back to direct `feature_tv_weight`.

- [ ] **Step 3: Update the dispatch in train()**

In `src/trainer.py`, around line 516, find:

```python
                if self.is_multi and self.config.render_mode == "pbr":
                    # Multi-mesh PBR: per-submesh gradient accumulation
                    step_loss = self._train_step_multi_pbr(camera, gt)
```

Replace with:

```python
                if self.is_multi and self.config.render_mode in ("pbr", "nlm"):
                    # Multi-mesh gradient accumulation (PBR or NLM)
                    step_loss = self._train_step_multi(camera, gt)
```

- [ ] **Step 4: Update the frozen-normal guard (single mesh PBR)**

Around line 552, the existing code:

```python
                if self._frozen_normal_submeshes and not self.is_multi:
                    if self.model.mat_texture.grad is not None:
                        self.model.mat_texture.grad[..., 5:8].zero_()
```

This is PBR-only single-mesh path. Leave it unchanged (NLM doesn't use this path because NLM single-mesh also goes through `_train_step_multi` if `is_multi=False`... wait, check the dispatch logic).

**Review dispatch logic**: The dispatch at line 516 checks `self.is_multi`. For NLM single-mesh (helmet loaded as single mesh), `is_multi` would be `False`, and it would fall into the single-mesh path. But NLM single-mesh should also work through `shade()`.

For simplicity and consistency with the existing PBR single-mesh path, the single-mesh NLM case will fall into the existing else branch. The existing else branch calls `self.model.shade(...)`. NLM's `shade()` delegates to `shade_submesh("__default__", ...)`. This works.

However, the else branch has PBR-specific env regularization code (lines 541-546). We need to guard it. Find:

```python
                    if self.config.render_mode == "pbr":
                        from src.losses import tv_loss
                        env_tv = tv_loss(self.model.env_map.raw) * self.config.pbr.env_tv_weight
                        env_decoded = self.model.env_map.decode()
                        env_l2 = (env_decoded ** 2).mean() * self.config.pbr.env_l2_weight
                        loss = loss + env_tv + env_l2
```

This is already guarded by `if self.config.render_mode == "pbr":`, so NLM will skip it. **No change needed.**

- [ ] **Step 5: Update the PSNR computation (around line 577)**

Find:

```python
                elif self.is_multi:
```

This handles multi-mesh PSNR for any model. It already calls `shade_submesh` generically. **No change needed** for NLM multi-mesh.

For NLM single-mesh, the `else` branch at line 592 calls `self.model.shade(...)`. NLM's `shade()` works. **No change needed.**

- [ ] **Step 6: Run regression tests**

```powershell
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -m pytest tests/ -v
```
Expected: All tests PASS (NLM and existing)

- [ ] **Step 7: Commit**

```powershell
git add src/trainer.py
git commit -m "refactor(trainer): generalize _train_step_multi_pbr → _train_step_multi for PBR+NLM"
```

---

## Task 13: NLM Optimizer with TTUR

**Files:**
- Modify: `src/trainer.py`

The optimizer must use two parameter groups for NLM: feature maps (lr=0.1) and MLP (lr=0.001).

- [ ] **Step 1: Modify _rebuild_optimizer**

In `src/trainer.py`, find the `_rebuild_optimizer` method (around line 148-159):

```python
    def _rebuild_optimizer(self) -> None:
        """根据 model.parameters() 重建优化器，保持特殊 lr 比例。"""
        base_lr = self.config.training.lr
        param_groups = []
        for i, p in enumerate(self.model.parameters()):
            if self.config.render_mode == "sh" and i == 1:
                param_groups.append({"params": [p], "lr": base_lr * self.config.training.rest_lr_ratio})
            elif self.config.render_mode == "pbr" and i == 1:
                param_groups.append({"params": [p], "lr": base_lr * self.config.pbr.env_lr_ratio})
            else:
                param_groups.append({"params": [p], "lr": base_lr})
        self.optimizer = Adam(param_groups)
```

Replace the entire method body with:

```python
    def _rebuild_optimizer(self) -> None:
        """根据 model.parameters() 重建优化器，保持特殊 lr 比例。"""
        base_lr = self.config.training.lr

        if self.config.render_mode == "nlm":
            # TTUR: feature maps (high lr) + MLP (low lr)
            feat_params = list(self.model.feature_maps.values())
            mlp_params = list(self.model.mlp.parameters())
            param_groups = [
                {"params": feat_params, "lr": self.config.nlm.feature_lr},
                {"params": mlp_params, "lr": self.config.nlm.mlp_lr},
            ]
        else:
            param_groups = []
            for i, p in enumerate(self.model.parameters()):
                if self.config.render_mode == "sh" and i == 1:
                    param_groups.append({"params": [p], "lr": base_lr * self.config.training.rest_lr_ratio})
                elif self.config.render_mode == "pbr" and i == 1:
                    param_groups.append({"params": [p], "lr": base_lr * self.config.pbr.env_lr_ratio})
                else:
                    param_groups.append({"params": [p], "lr": base_lr})

        self.optimizer = Adam(param_groups)
```

- [ ] **Step 2: Run tests**

```powershell
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -m pytest tests/ -v
```
Expected: All PASS

- [ ] **Step 3: Commit**

```powershell
git add src/trainer.py
git commit -m "feat(nlm): TTUR dual learning rate for feature maps + MLP"
```

---

## Task 14: NLM Resolution Resize

**Files:**
- Modify: `src/trainer.py`

When resolution changes during coarse-to-fine, NLM feature maps must be bilinearly resized. The existing `_resize_textures` already calls `get_material_texture` / `set_material_texture` which NLM implements. But NLM has no MLP weight resize (MLP weights are resolution-independent). We need to verify the existing path works.

- [ ] **Step 1: Review _resize_textures**

Read lines 161-190 of `src/trainer.py`. The existing code:

1. Calls `self.model.get_material_texture()` → returns dict (NLM returns feature maps dict)
2. Resizes each tensor via `F.interpolate`
3. Calls `self.model.set_material_texture(new_textures)` → NLM accepts dict

This should work for NLM. The MLP weights are not touched by `_resize_textures`, which is correct.

**However**, lines 185-190 call `_bake_normal_maps()` if `_frozen_normal_submeshes` is set. For NLM, `_frozen_normal_submeshes` is never set (only PBR sets it). So this is skipped. **No change needed.**

- [ ] **Step 2: Write an integration test for resize**

Append to `tests/test_nlm.py`:

```python
def test_nlm_get_set_material_texture_resize():
    """NLM texture get/set roundtrip preserves shape after resize."""
    model = _make_nlm_model(resolution=32, submesh_names=["A"])

    # Get textures
    tex_dict = model.get_material_texture()
    assert "A" in tex_dict
    assert tex_dict["A"].shape == (1, 32, 32, 12)

    # Resize via interpolate
    import torch.nn.functional as F
    resized = {}
    for name, tex in tex_dict.items():
        t = tex.permute(0, 3, 1, 2)
        t = F.interpolate(t, size=(64, 64), mode="bilinear", align_corners=False)
        t = t.permute(0, 2, 3, 1)
        resized[name] = t.contiguous()

    model.set_material_texture(resized)
    assert model.feature_maps["A"].shape == (1, 64, 64, 12)
```

- [ ] **Step 3: Run tests**

```powershell
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -m pytest tests/test_nlm.py::test_nlm_get_set_material_texture_resize -v
```
Expected: PASS

- [ ] **Step 4: Commit**

```powershell
git add tests/test_nlm.py
git commit -m "test(nlm): verify feature map resize roundtrip"
```

---

## Task 15: Helmet NLM Config

**Files:**
- Create: `configs/train_nlm_helmet.yaml`

- [ ] **Step 1: Inspect the helmet data path**

```powershell
Get-ChildItem data\helmet_260604\scene\*.glb | Select-Object Name
```

Note: helmet loads as single mesh (lowpoly.glb) or multi-mesh (original_with_mats.glb). Per spec, NLM follows the same data organization as PBR. For helmet, the PBR config uses `original_with_mats.glb` (multi-mesh with 1 submesh).

- [ ] **Step 2: Create the config**

Create `configs/train_nlm_helmet.yaml`:

```yaml
render_mode: nlm

data:
  mesh_path: data/helmet_260604/scene/original_with_mats.glb
  gt_dir: data/helmet_260604/gt
  camera_path: data/helmet_260604/cameras.json

texture:
  base_resolution: 512
  target_resolution: 2048

nlm:
  feature_dim: 12
  pe_level: 2
  mlp_hidden_dim: 32
  feature_lr: 0.1
  mlp_lr: 0.001
  feature_tv_weight: 0.00001
  feature_init_std: 0.1

training:
  num_epochs: 2000
  batch_size: 4
  lr: 0.01
  lr_decay: 0.5
  lr_decay_epochs: [500, 1000, 1500]
  resolution_schedule:
    - {epoch: 0, resolution: 512}
    - {epoch: 300, resolution: 1024}
    - {epoch: 700, resolution: 2048}

loss:
  lambda_l1: 1.0
  lambda_ssim: 0.2
  lambda_tv: 0.005

seam_padding:
  dilation_radius: 3
  apply_every_n_epochs: 50

video:
  num_frames: 120
  resolution: 1024
  fps: 30
```

- [ ] **Step 3: Commit**

```powershell
git add configs/train_nlm_helmet.yaml
git commit -m "feat(nlm): add helmet NLM config"
```

---

## Task 16: Piano Multi-Mesh NLM Config

**Files:**
- Create: `configs/train_nlm_piano_multi.yaml`

- [ ] **Step 1: Create the config**

Create `configs/train_nlm_piano_multi.yaml`:

```yaml
render_mode: nlm

data:
  mesh_path: data/piano_260604/scene/original_with_mats.glb
  gt_dir: data/piano_260604/gt
  camera_path: data/piano_260604/cameras.json

texture:
  base_resolution: 512
  target_resolution: 1024

nlm:
  feature_dim: 12
  pe_level: 2
  mlp_hidden_dim: 32
  feature_lr: 0.1
  mlp_lr: 0.001
  feature_tv_weight: 0.00001
  feature_init_std: 0.1

training:
  num_epochs: 2000
  batch_size: 4
  lr: 0.01
  lr_decay: 0.5
  lr_decay_epochs: [500, 1000, 1500]
  resolution_schedule:
    - {epoch: 0, resolution: 512}
    - {epoch: 300, resolution: 1024}

loss:
  lambda_l1: 1.0
  lambda_ssim: 0.2
  lambda_tv: 0.005

seam_padding:
  dilation_radius: 3
  apply_every_n_epochs: 50

video:
  num_frames: 120
  resolution: 1024
  fps: 30
```

- [ ] **Step 2: Commit**

```powershell
git add configs/train_nlm_piano_multi.yaml
git commit -m "feat(nlm): add piano multi-mesh NLM config"
```

---

## Task 17: Full Test Suite Regression

**Files:**
- None (verification only)

- [ ] **Step 1: Run the entire test suite**

```powershell
$env:PYTHONPATH="C:\Users\yangfei\Code\differentiable"
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -m pytest tests/ -v
```
Expected: All tests PASS (NLM unit tests + any existing SH/PBR tests)

- [ ] **Step 2: Smoke test — instantiate Trainer with NLM config**

```powershell
$env:PYTHONPATH="C:\Users\yangfei\Code\differentiable"
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -c "from src.config import load_config; from src.trainer import Trainer; cfg = load_config('configs/train_nlm_helmet.yaml'); t = Trainer(cfg); print('Trainer init OK, is_multi=', t.is_multi, 'submeshes=', t.submesh_names); print('Optimizer groups:', len(t.optimizer.param_groups))"
```
Expected: prints `Trainer init OK, is_multi=True submeshes=[...] Optimizer groups: 2`

- [ ] **Step 3: Commit if any fixes were needed**

If no fixes needed, proceed.

---

## Task 18: Helmet NLM Training — Smoke Test (200 epochs)

**Files:**
- None (manual integration test)

- [ ] **Step 1: Run 200 epoch smoke training**

```powershell
$env:PYTHONPATH="C:\Users\yangfei\Code\differentiable"
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe main.py --config configs/train_nlm_helmet.yaml --mode train --output output/helmet_nlm_smoke
```

Expected output over 200 epochs:
- No NaN errors
- PSNR steadily increasing (target: > 12 dB by epoch 200)
- Memory stays under 8GB (NLM is light — feature maps are small, MLP is tiny)

If PSNR is 0 or NaN, debug:
- Check that gradients flow (Task 8 test)
- Check TTUR learning rates (Task 13)
- Check that `gt_linear` is computed correctly

- [ ] **Step 2: Verify output files**

```powershell
Get-ChildItem output\helmet_nlm_smoke\epoch200 | Select-Object Name
```
Expected: `curves.png`, `compare_0000.png`, `feature_map_*.png`, `feature_map_*.pt`, `mlp_weights.pt`, `nlm_checkpoint.pt`, `orbit.mp4`

- [ ] **Step 3: Inspect compare image**

Open `output/helmet_nlm_smoke/epoch200/compare_0000.png`. The "NLM" panel should show a recognizable helmet shape with colors approximating GT (not pure noise).

If smoke test passes, proceed to Task 19.

---

## Task 19: Piano Multi-Mesh NLM Training — Smoke Test (200 epochs)

**Files:**
- None (manual integration test)

- [ ] **Step 1: Run 200 epoch smoke training**

```powershell
$env:PYTHONPATH="C:\Users\yangfei\Code\differentiable"
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe main.py --config configs/train_nlm_piano_multi.yaml --mode train --output output/piano_nlm_smoke
```

Expected: PSNR > 12 dB by epoch 200, no NaN, all 6 submeshes produce output.

- [ ] **Step 2: Verify output files**

```powershell
Get-ChildItem output\piano_nlm_smoke\epoch200 | Select-Object Name
```
Expected: per-submesh feature maps + shared `mlp_weights.pt`

- [ ] **Step 3: Commit (no code changes, just verification)**

If both smoke tests pass, the NLM feature is functionally complete. Proceed to Task 20 for full training.

---

## Task 20: Full Helmet NLM Training (2000 epochs)

**Files:**
- None (long-running training)

- [ ] **Step 1: Run full training**

```powershell
$env:PYTHONPATH="C:\Users\yangfei\Code\differentiable"
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe main.py --config configs/train_nlm_helmet.yaml --mode train --output output/helmet_nlm
```

Expected: PSNR > 15 dB (architecture connectivity threshold per spec). Compare with PBR's 20.81 dB — NLM may be lower initially.

- [ ] **Step 2: Inspect final results**

Review `output/helmet_nlm/epoch2000/compare_0000.png` and `curves.png`. Document PSNR.

- [ ] **Step 3: Commit results metadata (optional)**

No commit needed for training outputs (they're in gitignored `output/`).

---

## Task 21: Full Piano NLM Training (2000 epochs)

**Files:**
- None (long-running training)

- [ ] **Step 1: Run full training**

```powershell
$env:PYTHONPATH="C:\Users\yangfei\Code\differentiable"
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe main.py --config configs/train_nlm_piano_multi.yaml --mode train --output output/piano_nlm
```

Expected: PSNR > 15 dB.

- [ ] **Step 2: Inspect results**

Review output. Document PSNR.

---

## Task 22: Final Integration & Branch Merge Prep

**Files:**
- None (verification + summary)

- [ ] **Step 1: Final full test suite**

```powershell
$env:PYTHONPATH="C:\Users\yangfei\Code\differentiable"
C:\Users\yangfei\miniconda3\envs\differentiable\python.exe -m pytest tests/ -v
```
Expected: All PASS

- [ ] **Step 2: Review all commits on branch**

```powershell
git log --oneline feature/neural-lightmap ^master
```

Verify clean, atomic commit history.

- [ ] **Step 3: Push branch**

```powershell
git push -u origin feature/neural-lightmap
```

- [ ] **Step 4: Report results**

Summarize:
- Helmet NLM final PSNR vs PBR (20.81 dB) vs SH (13.19 dB)
- Piano NLM final PSNR vs PBR (28.80 dB) vs SH (20.37 dB)
- Any ablation needed (per spec Section 9)

---

## Self-Review Checklist (Post-Plan)

**Spec coverage:**
- ✅ Section 2.2 Infrastructure reuse — Tasks 12, 13 (trainer generalization), Tasks 9, 10 (factories)
- ✅ Section 2.3 Data flow — Task 7 (shade_submesh with mask indexing)
- ✅ Section 3.2 FeatureMap — Task 3
- ✅ Section 3.3 Positional Encoding — Task 1
- ✅ Section 3.4 TinyMLP — Task 2
- ✅ Section 3.5 shade_submesh mask indexing — Task 7
- ✅ Section 3.7 TTUR — Task 13
- ✅ Section 3.8 state_dict/load — Task 7
- ✅ Section 3.9 export — Task 7
- ✅ Section 3.10 Hooks — Tasks 5, 6, 7
- ✅ Section 4 Config — Task 4
- ✅ Section 5 Trainer generalization — Tasks 12, 13, 14
- ✅ Section 6 Logger/Video — Tasks 10, 11
- ✅ Section 8 Testing — Tasks 1-8 (unit), Tasks 18-21 (integration)
- ✅ Section 10 Export format — Task 7

**Type consistency check:**
- `positional_encode(d, level)` — same signature in Task 1 test and Task 7 usage ✅
- `TinyMLP(in_dim, hidden_dim, out_dim)` — same in Task 2 and Task 7 ✅
- `init_feature_map(resolution, feature_dim, init_std)` — same in Task 3 and Task 7 ✅
- `NeuralLightmapShadingModel(config)` — same in Task 7 and Task 9 ✅
- `regularization_loss()`, `get_submesh_texture(name)`, `post_backward_hook()` — same in Tasks 5, 6, 7, 12 ✅
- `feature_maps` dict key `"__default__"` for single mesh — consistent across Task 7 ✅

**Placeholder scan:** No TBD/TODO. All steps contain actual code or commands. ✅
