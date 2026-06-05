# Split-Sum PBR 可微烘焙管线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 split-sum PBR 替代 SH 参数化，支持金属/镜面材质烘焙，与现有 SH 路径共存。

**Architecture:** 新增 `src/shading/` 子目录，定义 `ShadingModel` 协议，SH 和 PBR 各实现一个模型。泛化 `trainer.py`/`exporter.py`/`video.py` 通过协议与着色模型交互。PBR 模型内部用 `shading/pbr/` 子模块实现材质参数化、环境贴图预滤波、BRDF LUT。

**Tech Stack:** PyTorch, nvdiffrast, numpy, OpenCV, PIL

**Spec:** `docs/superpowers/specs/2026-06-05-split-sum-pbr-design.md`

**Branch:** `feature/split-sum-pbr` from `master`

---

## File Structure

```
src/
├── shading/                    # [新建] 着色模型可插拔层
│   ├── __init__.py             # 工厂函数
│   ├── base.py                 # ShadingModel 基类
│   ├── sh_model.py             # SH 着色模型包装
│   ├── pbr_model.py            # PBR 着色模型
│   └── pbr/
│       ├── __init__.py
│       ├── material.py         # 材质参数化
│       ├── env_map.py          # Equirect 环境贴图
│       └── brdf_lut.py         # BRDF LUT
├── renderer.py                 # [修改] 提取法线插值
├── mesh.py                     # [修改] 加载法线
├── config.py                   # [修改] render_mode + PBRConfig
├── trainer.py                  # [修改] 接受 ShadingModel
├── exporter.py                 # [修改] 接受 ShadingModel
├── video.py                    # [修改] 接受 ShadingModel
├── main.py                     # [修改] render_mode 分发
├── sh.py                       # [不变]
├── dataset.py                  # [不变]
├── camera.py                   # [不变]
├── losses.py                   # [不变]
├── seam_padding.py             # [不变]
└── utils.py                    # [不变]
tests/
├── test_shading_base.py        # [新建]
├── test_sh_model.py            # [新建]
├── test_pbr_material.py        # [新建]
├── test_env_map.py             # [新建]
├── test_brdf_lut.py            # [新建]
├── test_pbr_model.py           # [新建]
├── test_mesh.py                # [修改] 增加法线测试
└── test_renderer.py            # [修改] 增加法线插值测试
configs/
└── train_pbr.yaml              # [新建] PBR 训练配置
```

---

## Task 1: 扩展 Config — render_mode + PBRConfig

**Files:**
- Modify: `src/config.py`
- Test: `tests/test_config.py` (新建)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
"""测试配置系统 — render_mode + PBRConfig。"""
import tempfile
import pytest
from src.config import Config, PBRConfig, load_config


def test_default_config_has_render_mode():
    cfg = Config()
    assert cfg.render_mode == "sh"


def test_default_config_has_pbr():
    cfg = Config()
    assert isinstance(cfg.pbr, PBRConfig)
    assert cfg.pbr.env_map_res == (64, 128)
    assert cfg.pbr.n_mip_levels == 5
    assert cfg.pbr.brdf_lut_size == 256
    assert cfg.pbr.env_lr_ratio == 1.0
    assert cfg.pbr.env_tv_weight == 0.001
    assert cfg.pbr.init_env_map is None


def test_load_config_with_pbr():
    yaml_content = """
render_mode: pbr
pbr:
  env_map_res: [128, 256]
  n_mip_levels: 7
  env_lr_ratio: 0.5
  init_env_map: /path/to/env.hdr
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(yaml_content)
        f.flush()
        cfg = load_config(f.name)

    assert cfg.render_mode == "pbr"
    assert cfg.pbr.env_map_res == [128, 256]
    assert cfg.pbr.n_mip_levels == 7
    assert cfg.pbr.env_lr_ratio == 0.5
    assert cfg.pbr.init_env_map == "/path/to/env.hdr"


def test_load_config_default_render_mode():
    yaml_content = """
data:
  mesh_path: test.obj
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(yaml_content)
        f.flush()
        cfg = load_config(f.name)

    assert cfg.render_mode == "sh"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — `ImportError: cannot import name 'PBRConfig' from 'src.config'`

- [ ] **Step 3: Write minimal implementation**

在 `src/config.py` 中添加 `PBRConfig` dataclass 和 `render_mode` 字段，修改 `load_config` 函数。

在 `from typing import List` 后添加 `from typing import List, Optional`。

在 `VideoConfig` 之前添加:

```python
@dataclass
class PBRConfig:
    env_map_res: list = field(default_factory=lambda: [64, 128])
    n_mip_levels: int = 5
    brdf_lut_size: int = 256
    env_lr_ratio: float = 1.0
    env_tv_weight: float = 0.001
    init_env_map: Optional[str] = None
```

在 `Config` dataclass 的第一个字段位置添加:

```python
    render_mode: str = "sh"  # "sh" | "pbr"
    pbr: PBRConfig = field(default_factory=PBRConfig)
```

在 `load_config` 函数中，`cfg = Config()` 之后添加:

```python
    if "render_mode" in raw:
        cfg.render_mode = raw["render_mode"]
    if "pbr" in raw:
        cfg.pbr = PBRConfig(**raw["pbr"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: add render_mode + PBRConfig to config system"
```

---

## Task 2: Mesh 法线加载

**Files:**
- Modify: `src/mesh.py`
- Test: `tests/test_mesh.py`

- [ ] **Step 1: Write the failing test**

在 `tests/test_mesh.py` 中添加:

```python
def test_mesh_data_has_normals():
    """MeshData 应包含法线字段。"""
    import numpy as np
    from src.mesh import MeshData

    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    uvs = np.zeros((3, 2), dtype=np.float64)
    uv_idx = np.array([[0, 1, 2]], dtype=np.int64)
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float64)

    mesh = MeshData(vertices=verts, faces=faces, uvs=uvs, uv_idx=uv_idx, normals=normals)
    assert mesh.normals is not None
    assert mesh.normals.shape == (3, 3)


def test_mesh_data_compute_vertex_normals():
    """compute_vertex_normals 应返回单位法线。"""
    import numpy as np
    from src.mesh import MeshData

    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    uvs = np.zeros((3, 2), dtype=np.float64)
    uv_idx = np.array([[0, 1, 2]], dtype=np.int64)
    normals = np.zeros_like(verts)

    mesh = MeshData(vertices=verts, faces=faces, uvs=uvs, uv_idx=uv_idx, normals=normals)
    vn = mesh.compute_vertex_normals()
    assert vn.shape == (3, 3)
    # 面法线应为 (0, 0, 1)，三个顶点法线相同
    for i in range(3):
        assert abs(vn[i, 2] - 1.0) < 0.01


def test_load_mesh_includes_normals(tmp_path):
    """load_mesh 应提取顶点法线。"""
    import numpy as np
    from src.mesh import load_mesh

    # 创建一个简单的 OBJ 文件
    obj_content = """v 0 0 0
v 1 0 0
v 0 1 0
vn 0 0 1
vn 0 0 1
vn 0 0 1
vt 0 0
vt 1 0
vt 0 1
f 1/1/1 2/2/2 3/3/3
"""
    obj_path = tmp_path / "test.obj"
    obj_path.write_text(obj_content)

    mesh = load_mesh(str(obj_path))
    assert mesh.normals is not None
    assert mesh.normals.shape[0] == 3
    assert mesh.normals.shape[1] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mesh.py::test_mesh_data_has_normals tests/test_mesh.py::test_load_mesh_includes_normals -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'normals'`

- [ ] **Step 3: Write minimal implementation**

修改 `src/mesh.py`:

1. `MeshData` dataclass 添加 `normals` 字段和 `normal_idx` 字段:

在 `uv_idx: np.ndarray` 后添加:

```python
    normals: np.ndarray = None           # 顶点法线 [N, 3]
    normal_idx: np.ndarray = None        # 面-法线索引 [M, 3]
```

2. 修改 `load_mesh` 函数提取法线。在 `uvs[:, 1] = uvs[:, 1] % 1.0` 之后、`return MeshData(...)` 之前添加法线提取逻辑:

```python
    # 提取顶点法线
    if hasattr(mesh_obj.visual, 'vertex_normals'):
        # trimesh 可能通过 visual 访问
        normals = np.array(mesh_obj.visual.vertex_normals, dtype=np.float64)
    elif hasattr(mesh_obj, 'vertex_normals'):
        normals = np.array(mesh_obj.vertex_normals, dtype=np.float64)
    else:
        # 无顶点法线 → 计算面法线转顶点法线
        normals = None

    normal_idx = np.array(mesh_obj.faces, dtype=np.int64)

    if normals is None:
        # 创建临时 MeshData 计算法线
        temp = MeshData(vertices=vertices, faces=faces, uvs=uvs, uv_idx=uv_idx,
                        normals=np.zeros_like(vertices), normal_idx=normal_idx)
        normals = temp.compute_vertex_normals()
```

3. 修改 return 语句:

```python
    return MeshData(vertices=vertices, faces=faces, uvs=uvs, uv_idx=uv_idx,
                    normals=normals, normal_idx=normal_idx)
```

4. 修改 `to_torch` 方法返回值:

```python
    def to_torch(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """将网格数据转换为 PyTorch 张量。

        Returns:
            (vertices, faces, uvs, uv_idx, normals, normal_idx)
        """
        v = torch.from_numpy(self.vertices.astype(np.float32))
        f = torch.from_numpy(self.faces.astype(np.int64))
        uv = torch.from_numpy(self.uvs.astype(np.float32))
        uvi = torch.from_numpy(self.uv_idx.astype(np.int64))
        if self.normals is not None:
            n = torch.from_numpy(self.normals.astype(np.float32))
            ni = torch.from_numpy(self.normal_idx.astype(np.int64))
        else:
            n = torch.zeros_like(v)
            ni = torch.zeros_like(f)
        return v, f, uv, uvi, n, ni
```

- [ ] **Step 4: Fix all callers of `mesh.to_torch()`**

修改 `src/renderer.py` 的 `__init__` 签名和构造:

```python
    def __init__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        uvs: torch.Tensor,
        uv_idx: torch.Tensor,
        normals: torch.Tensor = None,
        normal_idx: torch.Tensor = None,
        resolution: int = 512,
        device: str = "cuda",
    ):
```

在 `self.uv_idx = uv_idx.to(device).int()` 之后添加:

```python
        # normals: [N, 3]
        if normals is not None:
            self.normals = normals.to(device).float()
        else:
            self.normals = None
        # normal_idx: [F, 3] int32
        if normal_idx is not None:
            self.normal_idx = normal_idx.to(device).int()
        else:
            self.normal_idx = None
```

修改 `src/trainer.py` 中 `to_torch()` 调用处（约 line 68）:

```python
        self.vertices, self.faces, self.uvs, self.uv_idx, self.normals, self.normal_idx = mesh.to_torch()
```

修改 `self.renderer = self._create_renderer(...)` 传入法线（`_create_renderer` 方法和 `__init__` 中都改）:

```python
    def _create_renderer(self, resolution: int) -> DifferentiableRenderer:
        return DifferentiableRenderer(
            vertices=self.vertices,
            faces=self.faces,
            uvs=self.uvs,
            uv_idx=self.uv_idx,
            normals=self.normals,
            normal_idx=self.normal_idx,
            resolution=resolution,
            device=self.device,
        )
```

修改 `src/video.py` 中 `mesh.to_torch()` 调用处（约 line 154）:

```python
    verts, faces, uvs, uv_idx, normals, normal_idx = mesh.to_torch()
    renderer = DifferentiableRenderer(
        vertices=verts,
        faces=faces,
        uvs=uvs,
        uv_idx=uv_idx,
        normals=normals,
        normal_idx=normal_idx,
        resolution=resolution,
        device=device,
    )
```

- [ ] **Step 5: Run all existing tests**

Run: `python -m pytest tests/ -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/mesh.py src/renderer.py src/trainer.py src/video.py tests/test_mesh.py
git commit -m "feat: mesh normals loading + renderer normal interpolation prep"
```

---

## Task 3: Renderer 法线插值

**Files:**
- Modify: `src/renderer.py`
- Test: `tests/test_renderer.py`

- [ ] **Step 1: Write the failing test**

在 `tests/test_renderer.py` 中添加:

```python
def test_render_returns_normals():
    """渲染应返回插值后的法线。"""
    import torch
    import numpy as np
    from src.renderer import DifferentiableRenderer
    from src.camera import Camera

    # 简单三角形
    verts = torch.tensor([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=torch.float32)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    uvs = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.float32)
    uv_idx = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    normals = torch.tensor([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=torch.float32)
    normal_idx = torch.tensor([[0, 1, 2]], dtype=torch.int64)

    renderer = DifferentiableRenderer(
        vertices=verts, faces=faces, uvs=uvs, uv_idx=uv_idx,
        normals=normals, normal_idx=normal_idx,
        resolution=64, device="cuda",
    )

    # 简单正交相机
    cam = Camera(
        position=np.array([0, 0, 5]),
        look_at=np.array([0, 0, 0]),
        up=np.array([0, 1, 0]),
        fov_deg=45.0, image_width=64, image_height=64,
    )

    # 创建最小 SH 纹理 (order 0: 3 channels)
    dc = torch.ones(1, 16, 16, 3, device="cuda") * 0.5
    rest = torch.zeros(1, 16, 16, 0, device="cuda")

    rgb, mask, interp_normals = renderer.render(dc, rest, cam)
    assert interp_normals.shape == (1, 64, 64, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_renderer.py::test_render_returns_normals -v`
Expected: FAIL — `ValueError: too many values to unpack`

- [ ] **Step 3: Write minimal implementation**

修改 `src/renderer.py` 的 `render` 方法：

1. 在 `# ---- 5. 插值世界坐标 ----` 之后添加:

```python
        # ---- 5b. 插值法线 ----
        if self.normals is not None and self.normal_idx is not None:
            interp_normals, _ = dr.interpolate(self.normals, rast, self.normal_idx)  # [1, H, W, 3]
            interp_normals = interp_normals / (interp_normals.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            interp_normals = torch.zeros_like(world_pos)
```

2. 修改 return 语句:

```python
        return rgb, mask, interp_normals
```

3. 修复所有现有 `render` 调用点（`trainer.py`, `video.py`），将 `rgb, mask = renderer.render(...)` 改为 `rgb, mask, _ = renderer.render(...)`:

在 `src/trainer.py` 中，全局替换:
- `rendered, mask = self.renderer.render(` → `rendered, mask, _ = self.renderer.render(`
- `_rendered, _mask = self.renderer.render(` → `_rendered, _mask, _ = self.renderer.render(`

在 `src/video.py` 中:
- `rgb, mask = renderer.render(dc_param, rest_param, cam)` → `rgb, mask, _ = renderer.render(dc_param, rest_param, cam)`
- `rgb_sub, _ = renderer.render(sub_dc_param, sub_rest_param, cam)` → `rgb_sub, _, _ = renderer.render(sub_dc_param, sub_rest_param, cam)`

4. 修复现有测试中 `render` 的调用点（`tests/test_renderer.py`），将返回值解包为 3 个。

- [ ] **Step 4: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/renderer.py src/trainer.py src/video.py tests/test_renderer.py
git commit -m "feat: renderer returns interpolated normals"
```

---

## Task 4: ShadingModel 基类 + 工厂函数

**Files:**
- Create: `src/shading/__init__.py`
- Create: `src/shading/base.py`
- Test: `tests/test_shading_base.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_shading_base.py
"""测试 ShadingModel 基类。"""
import pytest
from src.shading.base import ShadingModel


def test_shading_model_has_required_methods():
    """ShadingModel 基类应定义所有必要的方法签名。"""
    required = [
        "parameters", "init_textures", "shade",
        "get_material_texture", "set_material_texture",
        "get_debug_info", "export", "state_dict", "load_state_dict",
    ]
    for name in required:
        assert hasattr(ShadingModel, name), f"Missing method: {name}"


def test_create_shading_model_sh():
    """工厂函数 render_mode='sh' 应返回 SHShadingModel 实例。"""
    from src.shading import create_shading_model
    from src.config import Config

    cfg = Config()
    model = create_shading_model("sh", cfg)
    assert model is not None
    assert hasattr(model, "parameters")


def test_create_shading_model_pbr():
    """工厂函数 render_mode='pbr' 应返回 PBRShadingModel 实例。"""
    from src.shading import create_shading_model
    from src.config import Config

    cfg = Config()
    cfg.render_mode = "pbr"
    model = create_shading_model("pbr", cfg)
    assert model is not None


def test_create_shading_model_invalid():
    """工厂函数应拒绝未知的 render_mode。"""
    from src.shading import create_shading_model
    from src.config import Config

    cfg = Config()
    with pytest.raises(ValueError, match="Unknown render_mode"):
        create_shading_model("invalid", cfg)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_shading_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.shading'`

- [ ] **Step 3: Create the module files**

`src/shading/__init__.py`:

```python
"""着色模型可插拔层 — 工厂函数。"""
from __future__ import annotations

from src.config import Config


def create_shading_model(render_mode: str, config: Config):
    """根据 render_mode 创建对应的着色模型。

    Args:
        render_mode: "sh" 或 "pbr"。
        config: 全局配置。

    Returns:
        ShadingModel 实例。
    """
    if render_mode == "sh":
        from src.shading.sh_model import SHShadingModel
        return SHShadingModel(config)
    elif render_mode == "pbr":
        from src.shading.pbr_model import PBRShadingModel
        return PBRShadingModel(config)
    else:
        raise ValueError(f"Unknown render_mode: {render_mode!r}")
```

`src/shading/base.py`:

```python
"""ShadingModel 基类 — 定义着色模型的接口协议。"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class ShadingModel:
    """着色模型基类。

    所有着色模型（SH, PBR 等）必须继承此类并实现所有方法。
    """

    def parameters(self) -> list[nn.Parameter]:
        """返回可优化参数列表。"""
        raise NotImplementedError

    def init_textures(self, resolution: int) -> None:
        """初始化纹理到指定分辨率。"""
        raise NotImplementedError

    def shade(
        self,
        rast_out: torch.Tensor,
        texc: torch.Tensor,
        world_pos: torch.Tensor,
        normals: torch.Tensor,
        view_dirs: torch.Tensor,
        camera,
        resolution: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """执行着色计算。

        Args:
            rast_out: 光栅化输出 [1, H, W, 4]。
            texc: 插值 UV 坐标 [1, H, W, 2]。
            world_pos: 插值世界坐标 [1, H, W, 3]。
            normals: 插值法线 [1, H, W, 3] (归一化)。
            view_dirs: 视角方向 [1, H, W, 3] (归一化)。
            camera: Camera 对象。
            resolution: 渲染分辨率。

        Returns:
            (rgb [1, H, W, 3], mask [1, H, W])
        """
        raise NotImplementedError

    def get_material_texture(self) -> torch.Tensor:
        """返回材质贴图张量（用于 seam padding 等）。"""
        raise NotImplementedError

    def set_material_texture(self, texture: torch.Tensor) -> None:
        """设置材质贴图张量。"""
        raise NotImplementedError

    def get_debug_info(self) -> dict:
        """返回调试信息（用于可视化）。"""
        return {}

    def export(self, output_dir: str) -> list[str]:
        """导出所有产物。

        Returns:
            导出文件路径列表。
        """
        raise NotImplementedError

    def state_dict(self) -> dict:
        """返回序列化状态。"""
        raise NotImplementedError

    def load_state_dict(self, state: dict) -> None:
        """加载序列化状态。"""
        raise NotImplementedError
```

占位 `src/shading/sh_model.py`（先空壳，Task 6 完善）:

```python
"""SH 着色模型包装。"""
from __future__ import annotations

from src.shading.base import ShadingModel


class SHShadingModel(ShadingModel):
    def __init__(self, config):
        self.config = config
```

占位 `src/shading/pbr_model.py`（先空壳，Task 7 完善）:

```python
"""PBR 着色模型。"""
from __future__ import annotations

from src.shading.base import ShadingModel


class PBRShadingModel(ShadingModel):
    def __init__(self, config):
        self.config = config
```

占位 `src/shading/pbr/__init__.py`:

```python
"""PBR 子模块。"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_shading_base.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/shading/ tests/test_shading_base.py
git commit -m "feat: ShadingModel base class + factory function"
```

---

## Task 5: PBR Material 模块

**Files:**
- Create: `src/shading/pbr/material.py`
- Test: `tests/test_pbr_material.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pbr_material.py
"""测试 PBR 材质参数化。"""
import torch
from src.shading.pbr.material import (
    init_material_texture,
    decode_material,
    compute_F0,
)


def test_init_material_texture_shape():
    tex = init_material_texture(64)
    assert tex.shape == (1, 64, 64, 5)


def test_init_material_texture_is_parameter():
    tex = init_material_texture(64)
    assert isinstance(tex, torch.nn.Parameter)


def test_decode_material_base_color_range():
    """sigmoid 约束后 base_color 应在 [0, 1]。"""
    tex = init_material_texture(32)
    base_color, roughness, metallic = decode_material(tex)
    assert base_color.min() >= 0.0
    assert base_color.max() <= 1.0
    assert roughness.min() >= 0.0
    assert roughness.max() <= 1.0
    assert metallic.min() >= 0.0
    assert metallic.max() <= 1.0


def test_decode_material_shapes():
    tex = init_material_texture(16)
    base_color, roughness, metallic = decode_material(tex)
    assert base_color.shape == (1, 16, 16, 3)
    assert roughness.shape == (1, 16, 16, 1)
    assert metallic.shape == (1, 16, 16, 1)


def test_compute_F0_dielectric():
    """非金属 (metallic=0) 的 F0 应为 ~0.04。"""
    base_color = torch.ones(1, 4, 4, 3) * 0.5
    metallic = torch.zeros(1, 4, 4, 1)
    F0 = compute_F0(base_color, metallic)
    assert torch.allclose(F0, torch.ones(1, 4, 4, 3) * 0.04, atol=1e-5)


def test_compute_F0_metallic():
    """金属 (metallic=1) 的 F0 应为 base_color。"""
    base_color = torch.ones(1, 4, 4, 3) * 0.8
    metallic = torch.ones(1, 4, 4, 1)
    F0 = compute_F0(base_color, metallic)
    assert torch.allclose(F0, base_color, atol=1e-5)


def test_material_gradient_flows():
    """decode_material 应保持梯度。"""
    tex = init_material_texture(16)
    base_color, roughness, metallic = decode_material(tex)
    loss = base_color.sum() + roughness.sum() + metallic.sum()
    loss.backward()
    assert tex.grad is not None
    assert tex.grad.abs().sum() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pbr_material.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

`src/shading/pbr/material.py`:

```python
"""PBR 材质参数化 — 单张 5 通道纹理 + sigmoid 约束。"""
from __future__ import annotations

import torch
import torch.nn as nn


def init_material_texture(resolution: int) -> nn.Parameter:
    """初始化材质贴图。

    5 通道: [base_color_R, base_color_G, base_color_B, roughness, metallic]
    存储为 nn.Parameter，初始值经过 inverse-sigmoid 昖射以使得
    sigmoid(raw) ≈ 期望初始值。

    Args:
        resolution: 纹理分辨率 (正方形)。

    Returns:
        nn.Parameter [1, resolution, resolution, 5]
    """
    # 期望初始值: base_color=0.5, roughness=0.5, metallic=0.0
    # inverse_sigmoid(y) = log(y / (1-y))
    # sigmoid_inv(0.5) = 0.0, sigmoid_inv(0.0) → -∞, 用 -5.0 近似 (sigmoid(-5)≈0.007)
    init_vals = torch.tensor([0.0, 0.0, 0.0, 0.0, -5.0])  # [5]
    data = init_vals.reshape(1, 1, 1, 5).expand(1, resolution, resolution, 5).clone()

    return nn.Parameter(data.float())


def decode_material(
    raw_texture: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从原始纹理张量解码材质参数。

    Args:
        raw_texture: 原始纹理 [1, H, W, 5]。

    Returns:
        (base_color, roughness, metallic) —
        base_color [1, H, W, 3], roughness [1, H, W, 1], metallic [1, H, W, 1]
    """
    decoded = torch.sigmoid(raw_texture)  # [1, H, W, 5]

    base_color = decoded[..., :3]  # [1, H, W, 3]
    roughness = decoded[..., 3:4]  # [1, H, W, 1]
    metallic = decoded[..., 4:5]  # [1, H, W, 1]

    return base_color, roughness, metallic


def compute_F0(
    base_color: torch.Tensor,
    metallic: torch.Tensor,
    dielectric_F0: float = 0.04,
) -> torch.Tensor:
    """计算菲涅尔 F0。

    F0 = lerp(dielectric_F0, base_color, metallic)

    Args:
        base_color: [1, H, W, 3]
        metallic: [1, H, W, 1]
        dielectric_F0: 非金属 F0 默认值。

    Returns:
        F0 [1, H, W, 3]
    """
    return dielectric_F0 * (1.0 - metallic) + base_color * metallic
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pbr_material.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/shading/pbr/material.py tests/test_pbr_material.py
git commit -m "feat: PBR material parameterization with sigmoid constraint"
```

---

## Task 6: Equirect 环境贴图 + 可导预滤波

**Files:**
- Create: `src/shading/pbr/env_map.py`
- Test: `tests/test_env_map.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_env_map.py
"""测试 equirect 环境贴图。"""
import torch
from src.shading.pbr.env_map import (
    init_env_map,
    direction_to_equirect,
    sample_env_map,
    prefilter_env_map,
    sample_prefiltered,
)


def test_init_env_map_default():
    env = init_env_map(32, 64)
    assert env.shape == (1, 32, 64, 3)
    assert isinstance(env, torch.nn.Parameter)


def test_direction_to_equirect_forward():
    """正前方 (+Z) 应映射到 equirect 中心。"""
    dirs = torch.tensor([[0.0, 0.0, 1.0]])  # [1, 3]
    u, v = direction_to_equirect(dirs)
    # atan2(1, 0) / (2π) + 0.5 = 0.5 + 0.5 = 1.0, but we wrap to [0,1]
    # atan2(1, 0) = π/2, u = π/2 / (2π) + 0.5 = 0.25 + 0.5 = 0.75
    # Actually: atan2(z=1, x=0) = π/2, u = π/2/(2π) + 0.5 = 0.75
    assert abs(u[0].item() - 0.75) < 0.01
    assert abs(v[0].item() - 0.5) < 0.01  # asin(0)/π + 0.5 = 0.5


def test_direction_to_equirect_up():
    """正上方 (+Y) 应映射到 equirect 顶部。"""
    dirs = torch.tensor([[0.0, 1.0, 0.0]])
    u, v = direction_to_equirect(dirs)
    assert abs(v[0].item() - 1.0) < 0.01  # asin(1)/π + 0.5 = 1.0


def test_direction_to_equirect_gradient():
    """坐标变换应保持梯度。"""
    dirs = torch.tensor([[0.0, 0.0, 1.0]], requires_grad=True)
    u, v = direction_to_equirect(dirs)
    u.sum().backward()
    assert dirs.grad is not None


def test_sample_env_map():
    """应能从环境贴图采样。"""
    env = init_env_map(16, 32)
    dirs = torch.tensor([[0.0, 0.0, 1.0]])  # [1, 3]
    color = sample_env_map(env, dirs)
    assert color.shape == (1, 3)


def test_prefilter_env_map_shape():
    """预滤波应输出正确的 mipmap 形状。"""
    env = init_env_map(16, 32)
    n_levels = 5
    prefiltered = prefilter_env_map(env, n_levels=n_levels)
    assert prefiltered.shape[0] == 1
    assert prefiltered.shape[1] == n_levels
    assert prefiltered.shape[2] == 16
    assert prefiltered.shape[3] == 32
    assert prefiltered.shape[4] == 3


def test_prefilter_env_map_gradient():
    """预滤波应保持对 env_map 的梯度。"""
    env = init_env_map(16, 32)
    prefiltered = prefilter_env_map(env, n_levels=3)
    loss = prefiltered.sum()
    loss.backward()
    assert env.grad is not None


def test_sample_prefiltered():
    """应能从预滤波贴图按 roughness 采样。"""
    env = init_env_map(16, 32)
    prefiltered = prefilter_env_map(env, n_levels=5)
    dirs = torch.tensor([[0.0, 0.0, 1.0]])  # [1, 3]
    roughness = torch.tensor([0.5])
    color = sample_prefiltered(prefiltered, dirs, roughness, n_levels=5)
    assert color.shape == (1, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_env_map.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

`src/shading/pbr/env_map.py`:

```python
"""Equirectangular 环境贴图 — 存储、采样、可导 mipmap 预滤波。"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_env_map(height: int, width: int, init_image: torch.Tensor | None = None) -> nn.Parameter:
    """初始化环境贴图。

    Args:
        height: 贴图高度。
        width: 贴图宽度。
        init_image: 可选初始图像 [1, H, W, 3]。None 则用均匀灰色。

    Returns:
        nn.Parameter [1, H, W, 3]
    """
    if init_image is not None:
        data = init_image.clone().float()
    else:
        # 均匀灰色 0.5, inverse-softplus(0.5) ≈ -0.1267
        data = torch.ones(1, height, width, 3) * 0.5
        # softplus_inv(x) = log(exp(x) - 1), 对于 x=0.5:
        data = torch.log(torch.exp(data) - 1.0 + 1e-6)

    return nn.Parameter(data)


def _decode_env_map(raw: torch.Tensor) -> torch.Tensor:
    """Softplus 约束保证非负。"""
    return F.softplus(raw)


def direction_to_equirect(dirs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """将方向向量转换为 equirect UV 坐标。

    Args:
        dirs: 归一化方向 [..., 3] (x, y, z)

    Returns:
        (u, v) — 各为 [...] 形状，值域 [0, 1]
    """
    x = dirs[..., 0]
    y = dirs[..., 1]
    z = dirs[..., 2]

    u = torch.atan2(z, x) / (2.0 * math.pi) + 0.5  # [0, 1]
    v = torch.asin(y.clamp(-0.999, 0.999)) / math.pi + 0.5  # [0, 1]

    return u, v


def sample_env_map(env_map: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """从环境贴图沿方向采样。

    Args:
        env_map: 原始环境贴图参数 [1, Eh, Ew, 3]
        dirs: 方向向量 [..., 3]

    Returns:
        颜色 [...] → [..., 3]
    """
    decoded = _decode_env_map(env_map)  # [1, Eh, Ew, 3]

    u, v = direction_to_equirect(dirs)

    # 构建 grid_sample 输入: [N, H, W, 2]
    orig_shape = u.shape
    grid = torch.stack([u, v], dim=-1)  # [..., 2]
    # 展平为 [1, 1, N, 2] 供 grid_sample 使用
    n_pixels = u.numel()
    grid = grid.reshape(1, 1, n_pixels, 2)

    # decoded: [1, Eh, Ew, 3] → [N, 3, Eh, Ew] → [1, 3, Eh, Ew]
    tex = decoded.permute(0, 3, 1, 2)  # [1, 3, Eh, Ew]

    sampled = F.grid_sample(tex, grid, mode="bilinear", padding_mode="border", align_corners=True)
    # sampled: [1, 3, 1, N] → [N, 3]
    sampled = sampled.reshape(3, n_pixels).T  # [N, 3]

    return sampled.reshape(*orig_shape, 3)


def prefilter_env_map(env_map: torch.Tensor, n_levels: int) -> torch.Tensor:
    """对环境贴图做可导的 2D 高斯卷积生成 mipmap 链。

    Args:
        env_map: 原始参数 [1, Eh, Ew, 3]
        n_levels: mipmap 级别数（包含 level 0）

    Returns:
        prefiltered [1, n_levels, Eh, Ew, 3]
    """
    decoded = _decode_env_map(env_map)  # [1, Eh, Ew, 3]
    H, W = decoded.shape[1], decoded.shape[2]

    levels = []
    for level in range(n_levels):
        roughness = level / max(n_levels - 1, 1)  # 0.0 → 1.0
        if level == 0:
            levels.append(decoded)
        else:
            # σ = roughness * max_kernel_sigma
            # kernel size 偶数取整，确保 σ 覆盖
            sigma = roughness * min(H, W) * 0.25
            kernel_size = int(sigma * 4) | 1  # 确保 odd
            kernel_size = max(kernel_size, 3)
            kernel_size = min(kernel_size, min(H, W))
            if kernel_size % 2 == 0:
                kernel_size += 1

            # 创建 2D 高斯核
            k = kernel_size
            ax = torch.arange(k, dtype=torch.float32, device=decoded.device) - k // 2
            xx, yy = torch.meshgrid(ax, ax, indexing="ij")
            kernel_2d = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2 + 1e-8))
            kernel_2d = kernel_2d / kernel_2d.sum()

            # 对每个通道做卷积
            # [1, Eh, Ew, 3] → [3, 1, Eh, Ew]
            inp = decoded[0].permute(2, 0, 1).unsqueeze(1)  # [3, 1, Eh, Ew]
            kernel_3ch = kernel_2d.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)  # [3, 1, k, k]
            pad = k // 2
            blurred = F.conv2d(inp, kernel_3ch, padding=pad, groups=1)  # [3, 1, Eh, Ew]
            blurred = blurred.squeeze(1).permute(1, 2, 0).unsqueeze(0)  # [1, Eh, Ew, 3]
            levels.append(blurred)

    # Stack: [1, n_levels, Eh, Ew, 3]
    return torch.stack(levels, dim=1)


def sample_prefiltered(
    prefiltered: torch.Tensor,
    dirs: torch.Tensor,
    roughness: torch.Tensor,
    n_levels: int,
) -> torch.Tensor:
    """从预滤波 mipmap 按 roughness 采样。

    在两个相邻 mipmap 级别之间做线性插值。

    Args:
        prefiltered: [1, n_levels, Eh, Ew, 3]
        dirs: 方向 [..., 3]
        roughness: [..., 1] 值域 [0, 1]
        n_levels: mipmap 级别数

    Returns:
        颜色 [..., 3]
    """
    u, v = direction_to_equirect(dirs)
    orig_shape = u.shape
    n_pixels = u.numel()

    # roughness → 连续 mip level
    mip_level = roughness.reshape(-1) * (n_levels - 1)  # [N]
    mip_level = mip_level.clamp(0, n_levels - 1)

    # 取 floor 和 ceil 两个级别
    level_lo = mip_level.floor().long().clamp(0, n_levels - 1)
    level_hi = (level_lo + 1).clamp(max=n_levels - 1)
    frac = (mip_level - level_lo.float()).reshape(-1, 1)  # [N, 1]

    grid = torch.stack([u.reshape(-1), v.reshape(-1)], dim=-1).reshape(1, 1, n_pixels, 2)

    Eh, Ew = prefiltered.shape[2], prefiltered.shape[3]
    colors_lo = []
    colors_hi = []

    for i in range(n_pixels):
        lo = level_lo[i]
        hi = level_hi[i]
        tex_lo = prefiltered[0, lo].permute(2, 0, 1).unsqueeze(0)  # [1, 3, Eh, Ew]
        tex_hi = prefiltered[0, hi].permute(2, 0, 1).unsqueeze(0)
        g = grid[:, :, i:i+1, :]
        c_lo = F.grid_sample(tex_lo, g, mode="bilinear", padding_mode="border", align_corners=True)
        c_hi = F.grid_sample(tex_hi, g, mode="bilinear", padding_mode="border", align_corners=True)
        colors_lo.append(c_lo.reshape(3))
        colors_hi.append(c_hi.reshape(3))

    colors_lo = torch.stack(colors_lo)  # [N, 3]
    colors_hi = torch.stack(colors_hi)  # [N, 3]
    color = colors_lo * (1.0 - frac) + colors_hi * frac  # [N, 3]

    return color.reshape(*orig_shape, 3)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_env_map.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/shading/pbr/env_map.py tests/test_env_map.py
git commit -m "feat: equirect env map with differentiable prefilter"
```

---

## Task 7: BRDF LUT

**Files:**
- Create: `src/shading/pbr/brdf_lut.py`
- Test: `tests/test_brdf_lut.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_brdf_lut.py
"""测试 BRDF LUT。"""
import torch
from src.shading.pbr.brdf_lut import generate_brdf_lut, sample_brdf


def test_generate_brdf_lut_shape():
    lut = generate_brdf_lut(64)
    assert lut.shape == (64, 64, 2)


def test_generate_brdf_lut_range():
    """LUT 值应合理 (scale 在 [0,1], bias 在 [0,1])。"""
    lut = generate_brdf_lut(64)
    assert lut.min() >= 0.0
    assert lut.max() <= 1.5  # scale 可能略超 1


def test_sample_brdf_shape():
    lut = generate_brdf_lut(64)
    NdotV = torch.tensor([0.5, 0.8, 0.1])
    roughness = torch.tensor([0.3, 0.7, 0.0])
    scale, bias = sample_brdf(lut, NdotV, roughness)
    assert scale.shape == (3,)
    assert bias.shape == (3,)


def test_sample_brdf_perfect_mirror():
    """roughness=0, NdotV=1 的 scale 应接近 1。"""
    lut = generate_brdf_lut(256)
    NdotV = torch.tensor([1.0])
    roughness = torch.tensor([0.0])
    scale, bias = sample_brdf(lut, NdotV, roughness)
    assert scale.item() > 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_brdf_lut.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

`src/shading/pbr/brdf_lut.py`:

```python
"""固定 BRDF LUT — GGX BRDF 积分表生成与采样。"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _ggx_importance_sample(NdotV: float, roughness: float, num_samples: int = 1024) -> tuple[float, float]:
    """对 GGX BRDF 做 importance sampling 积分，返回 (scale, bias)。

    参考: Epic Games 2013, "Real Shading in Unreal Engine 4"
    """
    V = NdotV  # 简化: 假设 view 在 normal 平面内
    a = roughness * roughness
    a2 = a * a

    scale = 0.0
    bias = 0.0

    for i in range(num_samples):
        # Hammersley sequence
        bits = i
        bits = ((bits << 16) & 0xFFFF0000) | ((bits >> 16) & 0x0000FFFF)
        bits = ((bits << 8) & 0xFF00FF00) | ((bits >> 8) & 0x00FF00FF)
        bits = ((bits << 4) & 0xF0F0F0F0) | ((bits >> 4) & 0x0F0F0F0F)
        bits = ((bits << 2) & 0xCCCCCCCC) | ((bits >> 2) & 0x33333333)
        bits = ((bits << 1) & 0xAAAAAAAA) | ((bits >> 1) & 0x55555555)
        xi_1 = float(bits) / float(0x100000000)

        bits2 = i + 1
        bits2 = ((bits2 << 16) & 0xFFFF0000) | ((bits2 >> 16) & 0x0000FFFF)
        bits2 = ((bits2 << 8) & 0xFF00FF00) | ((bits2 >> 8) & 0x00FF00FF)
        bits2 = ((bits2 << 4) & 0xF0F0F0F0) | ((bits2 >> 4) & 0x0F0F0F0F)
        bits2 = ((bits2 << 2) & 0xCCCCCCCC) | ((bits2 >> 2) & 0x33333333)
        bits2 = ((bits2 << 1) & 0xAAAAAAAA) | ((bits2 >> 1) & 0x55555555)
        xi_2 = float(bits2) / float(0x100000000)

        # GGX importance sample
        a2_safe = max(a2, 1e-7)
        phi = 2.0 * math.pi * xi_1
        cos_theta = math.sqrt((1.0 - xi_2) / (1.0 + (a2_safe - 1.0) * xi_2))
        sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)

        # Half vector in tangent space
        Hx = sin_theta * math.cos(phi)
        Hy = sin_theta * math.sin(phi)
        Hz = cos_theta

        # View direction: V = (sqrt(1 - NdotV^2), 0, NdotV)
        Vx = math.sqrt(max(1.0 - V * V, 0.0))
        Vy = 0.0
        Vz = V

        # NdotH
        NdotH = Hz
        # VdotH = NdotV * NdotH + Vx * Hx (简化)
        VdotH = Vz * Hz + Vx * Hx
        VdotH = max(VdotH, 1e-7)

        # G (Smith)
        k = (roughness + 1) ** 2 / 8.0
        G_V = NdotV / (NdotV * (1.0 - k) + k + 1e-7)
        G_L = cos_theta / (cos_theta * (1.0 - k) + k + 1e-7)
        G = G_V * G_L

        # F = (1 - (1 - VdotH)^5)
        F = (1.0 - (1.0 - VdotH)) ** 5  # placeholder for F0-independent part
        F_weight = G * VdotH / (NdotH * NdotV + 1e-7)

        scale += F_weight
        bias += F_weight * (1.0 - (1.0 - VdotH) ** 5)  # simplified

    scale /= num_samples
    bias /= num_samples

    return scale, bias


def generate_brdf_lut(size: int = 256) -> torch.Tensor:
    """生成 GGX BRDF 积分查找表。

    Args:
        size: LUT 分辨率 (正方形)。

    Returns:
        Tensor [size, size, 2]: 通道 0 = scale, 通道 1 = bias。
    """
    lut = torch.zeros(size, size, 2)

    for y in range(size):
        roughness = (y + 0.5) / size
        roughness = max(roughness, 0.04)  # 避免 0 roughness 奇异

        for x in range(size):
            NdotV = (x + 0.5) / size
            NdotV = max(NdotV, 0.001)  # 避免 NdotV=0 奇异

            scale, bias = _ggx_importance_sample(NdotV, roughness, num_samples=512)
            lut[y, x, 0] = scale
            lut[y, x, 1] = bias

    return lut


def sample_brdf(
    lut: torch.Tensor,
    NdotV: torch.Tensor,
    roughness: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从 BRDF LUT 采样。

    Args:
        lut: [size, size, 2]
        NdotV: [...] 值域 [0, 1]
        roughness: [...] 值域 [0, 1]

    Returns:
        (scale, bias) — 各为 [...] 形状
    """
    size = lut.shape[0]
    device = lut.device

    # Normalize to [0, 1]
    u = NdotV.clamp(0, 1)
    v = roughness.clamp(0, 1)

    # grid_sample: [1, 1, N, 2]
    orig_shape = u.shape
    n = u.numel()
    grid = torch.stack([u.reshape(-1), v.reshape(-1)], dim=-1).reshape(1, 1, n, 2)

    # lut: [size, size, 2] → [1, 2, size, size]
    lut_tex = lut.permute(2, 0, 1).unsqueeze(0).to(device)

    sampled = F.grid_sample(
        lut_tex, grid.to(device),
        mode="bilinear", padding_mode="border", align_corners=True,
    )  # [1, 2, 1, n]
    sampled = sampled.reshape(2, n).T  # [n, 2]

    scale = sampled[:, 0].reshape(*orig_shape)
    bias = sampled[:, 1].reshape(*orig_shape)

    return scale, bias
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_brdf_lut.py -v`
Expected: PASS (注意 `test_generate_brdf_lut_shape` 用 size=64 较快)

- [ ] **Step 5: Commit**

```bash
git add src/shading/pbr/brdf_lut.py tests/test_brdf_lut.py
git commit -m "feat: GGX BRDF LUT generation and sampling"
```

---

## Task 8: SH 着色模型包装

**Files:**
- Create: `src/shading/sh_model.py` (替换占位)
- Test: `tests/test_sh_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sh_model.py
"""测试 SH 着色模型包装。"""
import torch
from src.config import Config
from src.shading.sh_model import SHShadingModel


def test_sh_model_init():
    cfg = Config()
    model = SHShadingModel(cfg)
    model.init_textures(16)
    params = model.parameters()
    assert len(params) == 2  # features_dc, features_rest


def test_sh_model_state_dict():
    cfg = Config()
    model = SHShadingModel(cfg)
    model.init_textures(16)
    state = model.state_dict()
    assert "features_dc" in state
    assert "features_rest" in state
    assert "render_mode" in state
    assert state["render_mode"] == "sh"


def test_sh_model_get_material_texture():
    cfg = Config()
    model = SHShadingModel(cfg)
    model.init_textures(16)
    tex = model.get_material_texture()
    # DC + Rest 拼接
    n_sh = (cfg.texture.sh_order + 1) ** 2
    assert tex.shape[-1] == n_sh * 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sh_model.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

`src/shading/sh_model.py`:

```python
"""SH 着色模型包装 — 将现有 SH 管线封装为 ShadingModel 接口。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config
from src.sh import (
    init_sh_texture,
    cat_sh_features,
    eval_sh_basis,
)
from src.shading.base import ShadingModel


class SHShadingModel(ShadingModel):
    """SH 着色模型。"""

    def __init__(self, config: Config):
        self.config = config
        self.sh_order = config.texture.sh_order
        self.features_dc: nn.Parameter | None = None
        self.features_rest: nn.Parameter | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def parameters(self) -> list[nn.Parameter]:
        return [self.features_dc, self.features_rest]

    def init_textures(self, resolution: int) -> None:
        _dc, _rest = init_sh_texture(
            resolution,
            sh_order=self.sh_order,
            init_dc=self.config.texture.init_dc_value,
        )
        self.features_dc = nn.Parameter(_dc.data.to(self.device))
        self.features_rest = nn.Parameter(_rest.data.to(self.device))

    def shade(
        self,
        rast_out: torch.Tensor,
        texc: torch.Tensor,
        world_pos: torch.Tensor,
        normals: torch.Tensor,
        view_dirs: torch.Tensor,
        camera,
        resolution: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import nvdiffrast.torch as dr

        # 拼接完整 SH 纹理并采样
        full_tex = torch.cat([self.features_dc, self.features_rest], dim=-1)
        tex = dr.texture(full_tex, texc, filter_mode="linear", boundary_mode="clamp")

        # SH 解码
        n_sh = tex.shape[-1] // 3
        sh_order = int(n_sh ** 0.5) - 1
        sh_nx3 = tex.reshape(*tex.shape[:-1], n_sh, 3)
        basis = eval_sh_basis(view_dirs, order=sh_order)
        basis_exp = basis.unsqueeze(-1)
        rgb = (sh_nx3 * basis_exp).sum(dim=-2)
        rgb = rgb + 0.5
        rgb = rgb.clamp(0.0, 1.0)

        mask = (rast_out[..., 3] > 0).float()
        rgb = rgb * mask.unsqueeze(-1)

        return rgb, mask

    def get_material_texture(self) -> torch.Tensor:
        return cat_sh_features(self.features_dc, self.features_rest).detach().cpu()

    def set_material_texture(self, texture: torch.Tensor) -> None:
        self.features_dc = nn.Parameter(texture[..., :3].to(self.device).contiguous())
        self.features_rest = nn.Parameter(texture[..., 3:].to(self.device).contiguous())

    def state_dict(self) -> dict:
        return {
            "render_mode": "sh",
            "features_dc": self.features_dc.data.detach().cpu(),
            "features_rest": self.features_rest.data.detach().cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        if "features_dc" in state:
            self.features_dc = nn.Parameter(state["features_dc"].to(self.device))
            self.features_rest = nn.Parameter(state["features_rest"].to(self.device))

    def export(self, output_dir: str) -> list[str]:
        from src.exporter import export_diffuse_texture, export_sh_channels, export_gltf
        import os

        tex = self.get_material_texture()
        paths = []

        diffuse_path = os.path.join(output_dir, "diffuse.png")
        export_diffuse_texture(tex, diffuse_path, self.sh_order)
        paths.append(diffuse_path)

        sh_dir = os.path.join(output_dir, "sh_channels")
        paths.extend(export_sh_channels(tex, sh_dir, self.sh_order))

        return paths
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sh_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/shading/sh_model.py tests/test_sh_model.py
git commit -m "feat: SH shading model wrapper implementing ShadingModel"
```

---

## Task 9: PBR 着色模型

**Files:**
- Create: `src/shading/pbr_model.py` (替换占位)
- Test: `tests/test_pbr_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pbr_model.py
"""测试 PBR 着色模型。"""
import torch
import numpy as np
from src.config import Config
from src.shading.pbr_model import PBRShadingModel
from src.shading.base import ShadingModel


def test_pbr_model_is_shading_model():
    cfg = Config()
    model = PBRShadingModel(cfg)
    assert isinstance(model, ShadingModel)


def test_pbr_model_init_textures():
    cfg = Config()
    model = PBRShadingModel(cfg)
    model.init_textures(32)
    params = model.parameters()
    assert len(params) == 2  # mat_texture, env_map


def test_pbr_model_mat_texture_shape():
    cfg = Config()
    model = PBRShadingModel(cfg)
    model.init_textures(32)
    mat = model.get_material_texture()
    assert mat.shape == (1, 32, 32, 5)


def test_pbr_model_state_dict():
    cfg = Config()
    model = PBRShadingModel(cfg)
    model.init_textures(16)
    state = model.state_dict()
    assert state["render_mode"] == "pbr"
    assert "mat_texture" in state
    assert "env_map" in state


def test_pbr_model_shade_requires_cuda():
    """shade 方法需要 CUDA 和完整的 raster 输出，仅验证接口。"""
    cfg = Config()
    model = PBRShadingModel(cfg)
    model.init_textures(16)
    assert hasattr(model, "shade")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pbr_model.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

`src/shading/pbr_model.py`:

```python
"""PBR 着色模型 — Split-Sum 近似。"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config
from src.shading.base import ShadingModel
from src.shading.pbr.material import init_material_texture, decode_material, compute_F0
from src.shading.pbr.env_map import (
    init_env_map,
    prefilter_env_map,
    sample_prefiltered,
    sample_env_map,
)
from src.shading.pbr.brdf_lut import generate_brdf_lut, sample_brdf


class PBRShadingModel(ShadingModel):
    """PBR Split-Sum 着色模型。"""

    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.mat_texture: nn.Parameter | None = None
        self.env_map: nn.Parameter | None = None
        self.brdf_lut: torch.Tensor | None = None

        # 预计算 BRDF LUT (不参与优化)
        pbr_cfg = config.pbr
        self.brdf_lut = generate_brdf_lut(pbr_cfg.brdf_lut_size)
        self.n_mip_levels = pbr_cfg.n_mip_levels

    def parameters(self) -> list[nn.Parameter]:
        return [self.mat_texture, self.env_map]

    def init_textures(self, resolution: int) -> None:
        pbr_cfg = self.config.pbr
        eh, ew = pbr_cfg.env_map_res

        self.mat_texture = init_material_texture(resolution).to(self.device)
        self.env_map = init_env_map(eh, ew).to(self.device)

    def shade(
        self,
        rast_out: torch.Tensor,
        texc: torch.Tensor,
        world_pos: torch.Tensor,
        normals: torch.Tensor,
        view_dirs: torch.Tensor,
        camera,
        resolution: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """PBR split-sum 着色。"""
        import nvdiffrast.torch as dr

        # ---- 1. 采样材质贴图 ----
        mat_raw = dr.texture(
            self.mat_texture, texc, filter_mode="linear", boundary_mode="clamp"
        )  # [1, H, W, 5]
        base_color, roughness, metallic = decode_material(mat_raw)

        # ---- 2. 计算反射方向 ----
        NdotV = (normals * view_dirs).sum(dim=-1, keepdim=True).clamp(0, 1)  # [1, H, W, 1]
        reflect_dir = 2.0 * NdotV * normals - view_dirs  # [1, H, W, 3]
        reflect_dir = reflect_dir / (reflect_dir.norm(dim=-1, keepdim=True) + 1e-8)

        # ---- 3. 预滤波环境贴图 ----
        prefiltered = prefilter_env_map(self.env_map, self.n_mip_levels)  # [1, M, Eh, Ew, 3]

        # ---- 4. Diffuse 项 ----
        # irradiance 用 level 0 (最大模糊) 沿法线方向采样
        irradiance = sample_prefiltered(prefiltered, normals, torch.zeros_like(NdotV), self.n_mip_levels)
        F0 = compute_F0(base_color, metallic)
        kd = (1.0 - metallic) * (1.0 - F0)
        diffuse = kd * base_color * irradiance  # [1, H, W, 3]

        # ---- 5. Specular 项 ----
        # 采样 prefiltered env map
        prefiltered_color = sample_prefiltered(prefiltered, reflect_dir, roughness, self.n_mip_levels)
        # 采样 BRDF LUT
        NdotV_flat = NdotV.reshape(-1)
        roughness_flat = roughness.reshape(-1)
        scale, bias = sample_brdf(self.brdf_lut, NdotV_flat, roughness_flat)
        scale = scale.reshape(*NdotV.shape)  # [1, H, W, 1]
        bias = bias.reshape(*NdotV.shape)    # [1, H, W, 1]
        specular = (F0 * scale + bias) * prefiltered_color  # [1, H, W, 3]

        # ---- 6. 合成 ----
        rgb = diffuse + specular
        rgb = rgb.clamp(0.0, 1.0)

        # ---- 7. 遮罩 ----
        mask = (rast_out[..., 3] > 0).float()
        rgb = rgb * mask.unsqueeze(-1)

        # 保存调试信息
        self._last_debug = {
            "diffuse": diffuse.detach(),
            "specular": specular.detach(),
            "base_color": base_color.detach(),
            "roughness": roughness.detach(),
            "metallic": metallic.detach(),
        }

        return rgb, mask

    def get_material_texture(self) -> torch.Tensor:
        return self.mat_texture.data.detach().cpu()

    def set_material_texture(self, texture: torch.Tensor) -> None:
        self.mat_texture = nn.Parameter(texture.to(self.device).contiguous())

    def get_debug_info(self) -> dict:
        return getattr(self, "_last_debug", {})

    def state_dict(self) -> dict:
        return {
            "render_mode": "pbr",
            "mat_texture": self.mat_texture.data.detach().cpu(),
            "env_map": self.env_map.data.detach().cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        if "mat_texture" in state:
            self.mat_texture = nn.Parameter(state["mat_texture"].to(self.device))
        if "env_map" in state:
            self.env_map = nn.Parameter(state["env_map"].to(self.device))

    def export(self, output_dir: str) -> list[str]:
        import numpy as np
        from PIL import Image

        os.makedirs(output_dir, exist_ok=True)
        paths = []

        # 解码材质
        base_color, roughness, metallic = decode_material(self.mat_texture)

        # base_color.png
        bc = base_color[0].clamp(0, 1).pow(1.0 / 2.2).cpu().numpy()
        bc = (bc * 255).astype(np.uint8)
        p = os.path.join(output_dir, "base_color.png")
        Image.fromarray(bc, "RGB").save(p)
        paths.append(p)

        # roughness.png
        r = roughness[0].clamp(0, 1).cpu().numpy().repeat(3, axis=-1)  # RGB 灰度
        r = (r * 255).astype(np.uint8)
        p = os.path.join(output_dir, "roughness.png")
        Image.fromarray(r, "RGB").save(p)
        paths.append(p)

        # metallic.png
        m = metallic[0].clamp(0, 1).cpu().numpy().repeat(3, axis=-1)
        m = (m * 255).astype(np.uint8)
        p = os.path.join(output_dir, "metallic.png")
        Image.fromarray(m, "RGB").save(p)
        paths.append(p)

        # env_map.png (简单预览)
        from src.shading.pbr.env_map import _decode_env_map
        env_decoded = _decode_env_map(self.env_map)
        env_img = env_decoded[0].clamp(0, 1).cpu().numpy()
        env_img = (env_img * 255).astype(np.uint8)
        p = os.path.join(output_dir, "env_map.png")
        Image.fromarray(env_img, "RGB").save(p)
        paths.append(p)

        return paths
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pbr_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/shading/pbr_model.py tests/test_pbr_model.py
git commit -m "feat: PBR shading model with split-sum"
```

---

## Task 10: 泛化 Trainer 接受 ShadingModel

**Files:**
- Modify: `src/trainer.py`
- Test: `tests/test_trainer.py` (修改)

这是最复杂的任务。需要将现有 SH 硬编码逻辑替换为通过 ShadingModel 接口交互。

- [ ] **Step 1: 修改 Trainer.__init__ 接受可选 ShadingModel**

修改 `src/trainer.py` 的 `Trainer.__init__` 签名:

```python
    def __init__(self, config: Config, shading_model=None) -> None:
```

在 `__init__` 中，将现有的 SH 初始化块替换为:

```python
        # ---- 3. 着色模型 ----
        if shading_model is not None:
            self.model = shading_model
        else:
            from src.shading import create_shading_model
            self.model = create_shading_model(config.render_mode, config)

        self.model.init_textures(config.texture.base_resolution)
```

删除原来的 SH 初始化块 (features_dc/features_rest 初始化) 和优化器初始化。

修改优化器初始化为通用形式:

```python
        # ---- 4. 优化器 ----
        base_lr = config.training.lr
        param_groups = []
        for i, p in enumerate(self.model.parameters()):
            if config.render_mode == "sh" and i == 1:
                # SH: 高阶用 rest_lr_ratio
                param_groups.append({"params": [p], "lr": base_lr * config.training.rest_lr_ratio})
            elif config.render_mode == "pbr" and i == 1:
                # PBR: env_map 用 env_lr_ratio
                param_groups.append({"params": [p], "lr": base_lr * config.pbr.env_lr_ratio})
            else:
                param_groups.append({"params": [p], "lr": base_lr})
        self.optimizer = Adam(param_groups)
```

- [ ] **Step 2: 修改 render 调用为通过渲染器基础 + 模型 shade**

将 `self.renderer.render(features_dc, features_rest, camera)` 替换为两步:
1. 渲染器做光栅化 + 插值（需要暴露这些中间结果）
2. 模型做着色

修改 `src/renderer.py` 添加 `rasterize_and_interpolate` 方法:

```python
    def rasterize_and_interpolate(
        self, camera
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """光栅化 + 插值，返回中间结果供着色模型使用。

        Returns:
            (rast_out, texc, world_pos, normals, view_dirs)
        """
        dr = _get_dr()
        h = w = self.resolution

        mvp = camera.mvp_torch().to(self.device)
        verts = self.vertices
        ones = torch.ones_like(verts[..., :1])
        verts_h = torch.cat([verts, ones], dim=-1)
        clip = torch.bmm(verts_h, mvp.transpose(1, 2))

        rast, _ = dr.rasterize(self.glctx, clip, self.faces, resolution=[h, w])
        texc, _ = dr.interpolate(self.uvs, rast, self.uv_idx)
        world_pos, _ = dr.interpolate(self.vertices, rast, self.faces)

        if self.normals is not None and self.normal_idx is not None:
            interp_normals, _ = dr.interpolate(self.normals, rast, self.normal_idx)
            interp_normals = interp_normals / (interp_normals.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            interp_normals = torch.zeros_like(world_pos)

        cam_pos = (
            torch.tensor(camera.position, dtype=torch.float32, device=self.device)
            .reshape(1, 1, 1, 3)
        )
        view_dirs = cam_pos - world_pos
        view_dirs = view_dirs / (view_dirs.norm(dim=-1, keepdim=True) + 1e-8)

        return rast, texc, world_pos, interp_normals, view_dirs
```

保留原 `render` 方法（SH 模型兼容性），在 `trainer.py` 中根据 render_mode 选择调用方式。

- [ ] **Step 3: 修改训练循环**

将训练循环中的 render 调用替换为:

```python
                # 渲染
                if self.config.render_mode == "sh":
                    rendered, mask, _ = self.renderer.render(
                        self.model.features_dc, self.model.features_rest, camera,
                    )
                else:
                    rast, texc, wpos, normals, vdirs = self.renderer.rasterize_and_interpolate(camera)
                    rendered, mask = self.model.shade(rast, texc, wpos, normals, vdirs, camera, self.current_resolution)
```

PSNR 计算中的渲染调用同样修改。

- [ ] **Step 4: 修改纹理 resize/seam padding**

```python
    def _resize_textures(self, new_res: int) -> None:
        if self.config.render_mode == "sh":
            self._resize_sh_texture(new_res)
        else:
            # PBR: resize mat_texture, env_map 不变
            old_res = self.model.mat_texture.shape[1]
            if old_res == new_res:
                return
            tex = self.model.mat_texture.data.permute(0, 3, 1, 2)
            tex = F.interpolate(tex, size=(new_res, new_res), mode="bilinear", align_corners=False)
            tex = tex.permute(0, 2, 3, 1)
            self.model.mat_texture = nn.Parameter(tex.contiguous().to(self.device))
            self._rebuild_optimizer()

    def _apply_seam_padding(self) -> None:
        radius = self.config.seam_padding.dilation_radius
        tex = self.model.get_material_texture().to(self.device)
        H, W = tex.shape[1], tex.shape[2]
        valid_mask = torch.ones(1, H, W, 1, device=self.device)
        tex = dilate_texture(tex, valid_mask, radius=radius).contiguous()
        self.model.set_material_texture(tex)

        if self.config.render_mode == "sh":
            # SH 模型需要重建优化器
            self._rebuild_optimizer()

    def _rebuild_optimizer(self) -> None:
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

- [ ] **Step 5: 修改 checkpoint**

```python
                # Checkpoint
                ckpt = self.model.state_dict()
                ckpt["epoch"] = epoch + 1
                ckpt["loss"] = avg_loss
                ckpt["resolution"] = self.current_resolution
                torch.save(ckpt, ckpt_path)
```

断点续训:

```python
        if resume_from is not None:
            ckpt = torch.load(resume_from, map_location=self.device)
            if isinstance(ckpt, dict) and "render_mode" in ckpt:
                if ckpt["render_mode"] == "pbr":
                    self.model.load_state_dict(ckpt)
                elif ckpt["render_mode"] == "sh":
                    self.model.load_state_dict(ckpt)
                start_epoch = ckpt.get("epoch", 0)
            elif isinstance(ckpt, dict) and "features_dc" in ckpt:
                # 旧格式 SH checkpoint
                self.model.load_state_dict(ckpt)
                start_epoch = ckpt.get("epoch", 0)
            elif isinstance(ckpt, dict) and "sh_texture" in ckpt:
                # 最旧格式
                tex = ckpt["sh_texture"]
                state = {
                    "render_mode": "sh",
                    "features_dc": tex[..., :3],
                    "features_rest": tex[..., 3:],
                }
                self.model.load_state_dict(state)
                start_epoch = ckpt.get("epoch", 0)
```

- [ ] **Step 6: 修改 debug export**

在 `_export_debug` 中，根据 render_mode 选择不同的导出逻辑:

```python
        if self.config.render_mode == "pbr":
            self._export_debug_pbr(output_dir, epoch)
        else:
            self._export_debug_sh(output_dir, epoch)
```

将现有 SH debug 逻辑搬到 `_export_debug_sh`，PBR 版本在 `_export_debug_pbr` 中实现:

```python
    def _export_debug_pbr(self, output_dir: str, epoch: int) -> None:
        import cv2
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 导出材质贴图
        self.model.export(output_dir)

        # Compare 图
        debug_info = self.model.get_debug_info()
        num_views = len(self.dataset)
        compare_count = min(4, num_views)
        compare_indices = [int(i * num_views / compare_count) for i in range(compare_count)]

        for ci, idx in enumerate(compare_indices):
            img_np, camera = self.dataset[idx]

            with torch.no_grad():
                rast, texc, wpos, normals, vdirs = self.renderer.rasterize_and_interpolate(camera)
                rgb, mask = self.model.shade(rast, texc, wpos, normals, vdirs, camera, self.current_resolution)

            mask = mask.flip(1)
            mask_np = mask[0].cpu().numpy()

            def to_srgb_bgr(rgb_tensor):
                img = rgb_tensor[0].flip(0).clamp(0, 1).pow(1.0 / 2.2).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                bgr[mask_np < 0.5] = 0
                return bgr

            gt_bgr = cv2.cvtColor((img_np.transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

            diffuse_bgr = to_srgb_bgr(debug_info.get("diffuse", rgb))
            specular_bgr = to_srgb_bgr(debug_info.get("specular", rgb * 0))

            panels = [
                (gt_bgr, "GT"),
                (to_srgb_bgr(rgb), "Rendered"),
                (diffuse_bgr, "Diffuse"),
                (specular_bgr, "Specular"),
            ]

            target_h = min(p[0].shape[0] for p in panels)
            resized = []
            for img, label in panels:
                h, w = img.shape[:2]
                r = cv2.resize(img, (w * target_h // h, target_h))
                cv2.putText(r, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                resized.append(r)

            top = np.concatenate([resized[0], resized[1]], axis=1)
            bottom = np.concatenate([resized[2], resized[3]], axis=1)
            canvas = np.concatenate([top, bottom], axis=0)
            cv2.imwrite(os.path.join(output_dir, f"compare_{ci:04d}.png"), canvas)
```

- [ ] **Step 7: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All pass

- [ ] **Step 8: Commit**

```bash
git add src/trainer.py src/renderer.py tests/test_trainer.py
git commit -m "feat: generalize trainer to accept ShadingModel"
```

---

## Task 11: 泛化 Video 和 Exporter

**Files:**
- Modify: `src/video.py`
- Modify: `src/exporter.py`
- Modify: `main.py`

- [ ] **Step 1: 修改 video.py**

修改 `render_video` 函数签名，接受 `ShadingModel`:

```python
def render_video(
    shading_model,
    mesh: MeshData,
    output_path: str,
    ...
) -> str:
```

在函数内部替换 SH 特定逻辑:

```python
    verts, faces, uvs, uv_idx, normals, normal_idx = mesh.to_torch()
    renderer = DifferentiableRenderer(
        vertices=verts, faces=faces, uvs=uvs, uv_idx=uv_idx,
        normals=normals, normal_idx=normal_idx,
        resolution=resolution, device=device,
    )

    for i, cam in enumerate(cameras):
        with torch.no_grad():
            rast, texc, wpos, interp_normals, vdirs = renderer.rasterize_and_interpolate(cam)
            rgb, mask = shading_model.shade(rast, texc, wpos, interp_normals, vdirs, cam, resolution)
            # ... 其余不变
```

为了向后兼容，保留 `sh_texture` 参数路径作为 fallback。

- [ ] **Step 2: 修改 main.py**

```python
    # 创建着色模型
    from src.shading import create_shading_model
    model = create_shading_model(cfg.render_mode, cfg)

    if args.mode == "train":
        trainer = Trainer(cfg, shading_model=model)
        trainer.train(output_dir=output_dir, resume_from=args.resume)
    elif args.mode == "export":
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        model.export(output_dir)
    elif args.mode == "video":
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        from src.video import render_video
        mesh = load_mesh(cfg.data.mesh_path)
        render_video(
            shading_model=model, mesh=mesh,
            output_path=os.path.join(output_dir, "orbit.mp4"),
            ...
        )
```

- [ ] **Step 3: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add src/video.py src/exporter.py main.py
git commit -m "feat: generalize video/exporter/main to use ShadingModel"
```

---

## Task 12: PBR 训练配置

**Files:**
- Create: `configs/train_pbr.yaml`

- [ ] **Step 1: Create PBR config file**

`configs/train_pbr.yaml`:

```yaml
render_mode: pbr

data:
  mesh_path: data/helmet_260604/lowpoly.glb
  gt_dir: data/helmet_260604/gt
  camera_path: data/helmet_260604/cameras.json

texture:
  sh_order: 2  # PBR 模式忽略此值
  base_resolution: 512
  target_resolution: 2048

training:
  num_epochs: 2000
  lr: 0.01
  batch_size: 4
  lr_decay: 0.5
  lr_decay_epochs: [500, 1000, 1500]
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
  env_map_res: [64, 128]
  n_mip_levels: 5
  brdf_lut_size: 256
  env_lr_ratio: 1.0
  env_tv_weight: 0.001

video:
  num_frames: 120
  resolution: 1024
  fps: 30
```

- [ ] **Step 2: Commit**

```bash
git add configs/train_pbr.yaml
git commit -m "feat: PBR training config for helmet dataset"
```

---

## Task 13: 端到端验证 — 头盔数据集 PBR 训练

**Files:** 无新文件

- [ ] **Step 1: 创建分支**

```bash
git checkout -b feature/split-sum-pbr
```

- [ ] **Step 2: 运行所有测试**

```bash
python -m pytest tests/ -v
```
Expected: All pass

- [ ] **Step 3: 头盔数据集 PBR 训练 (quick test)**

```bash
python main.py train --config configs/train_pbr.yaml --output output/helmet_pbr_test
```

观察:
- 前 10 个 epoch loss 是否下降
- PSNR 是否合理
- 是否有 CUDA OOM

- [ ] **Step 4: 检查调试输出**

验证 `output/helmet_pbr_test/` 下:
- compare 图: GT vs Rendered vs Diffuse vs Specular
- 材质贴图: base_color.png, roughness.png, metallic.png
- env_map.png

- [ ] **Step 5: 完整训练 (如果 quick test OK)**

```bash
python main.py train --config configs/train_pbr.yaml --output output/helmet_pbr --checkpoint-every 200
```

对比 SH 结果 (13.19 dB) — PBR 目标 > 15 dB。

- [ ] **Step 6: Commit results (if tracked)**

```bash
git add -A
git commit -m "feat: complete split-sum PBR pipeline with ShadingModel architecture"
```
