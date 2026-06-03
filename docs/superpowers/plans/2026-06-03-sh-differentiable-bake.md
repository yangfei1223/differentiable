# 可微烘焙管线 MVP 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于 PyTorch + nvdiffrast 构建完整的可微渲染烘焙管线 MVP，将高模多视角 GT 渲染图反向优化为低模 2 阶球谐辐射场纹理。

**Architecture:** 模块化库架构，每个关注点一个模块（camera/mesh/dataset/sh/renderer/losses/seam_padding/trainer/exporter），YAML 配置驱动，CLI 入口。前向管线使用 nvdiffrast 光栅化 + SH 解码，反向通过 PyTorch autograd 优化 SH 纹理参数。Coarse-to-Fine 多分辨率训练策略。

**Tech Stack:** Python 3.10+, PyTorch 2.x, nvdiffrast, numpy, opencv-python, trimesh, Pillow, PyYAML, conda

---

## 文件结构

```
differentiable-bake/
├── data/                       # 训练数据 (由 Blender 导出, gitignore)
│   ├── scene/
│   │   └── lowpoly.obj         # 低模 (已展UV)
│   ├── gt/
│   │   ├── view_0000.png       # GT 渲染图
│   │   └── ...
│   └── cameras.json            # 相机参数
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── camera.py
│   ├── mesh.py
│   ├── dataset.py
│   ├── sh.py
│   ├── renderer.py
│   ├── losses.py
│   ├── seam_padding.py
│   ├── trainer.py
│   └── exporter.py
├── scripts/
│   ├── blender_export.py       # Blender 内运行的数据导出脚本
│   └── README.md               # Blender MCP 操作 SOP
├── tests/
│   ├── __init__.py
│   ├── test_camera.py
│   ├── test_mesh.py
│   ├── test_dataset.py
│   ├── test_sh.py
│   ├── test_renderer.py
│   ├── test_losses.py
│   ├── test_seam_padding.py
│   ├── test_trainer.py
│   └── test_exporter.py
├── configs/
│   └── default.yaml
├── output/                     # 导出结果 (gitignore)
├── main.py
├── requirements.txt
├── setup.py
├── environment.yml
└── .gitignore
```

---

### Task 1: conda 环境与项目脚手架

**Files:**
- Create: `environment.yml`
- Create: `requirements.txt`
- Create: `setup.py`
- Create: `configs/default.yaml`
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `main.py`
- Create: `tests/__init__.py`
- Create: `.gitignore`
- Create: `data/.gitkeep`

- [ ] **Step 1: 创建 environment.yml**

```yaml
name: differentiable
channels:
  - pytorch
  - nvidia
  - defaults
dependencies:
  - python=3.10
  - pytorch::pytorch>=2.0
  - pytorch::torchvision
  - pytorch::pytorch-cuda=12.1
  - pip
  - pip:
    - numpy>=1.24.0
    - opencv-python>=4.8.0
    - trimesh>=4.0.0
    - PyYAML>=6.0
    - Pillow>=10.0.0
    - tqdm>=4.65.0
    - imageio>=2.31.0
```

- [ ] **Step 2: 创建 requirements.txt**

```txt
torch>=2.0.0
numpy>=1.24.0
opencv-python>=4.8.0
trimesh>=4.0.0
PyYAML>=6.0
Pillow>=10.0.0
tqdm>=4.65.0
imageio>=2.31.0
```

- [ ] **Step 3: 创建 setup.py**

```python
from setuptools import setup, find_packages

setup(
    name="differentiable-bake",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0", "numpy>=1.24.0", "opencv-python>=4.8.0",
        "trimesh>=4.0.0", "PyYAML>=6.0", "Pillow>=10.0.0",
        "tqdm>=4.65.0", "imageio>=2.31.0",
    ],
)
```

- [ ] **Step 4: 创建 configs/default.yaml**

```yaml
data:
  mesh_path: "data/scene/lowpoly.obj"
  gt_dir: "data/gt"
  camera_path: "data/cameras.json"

texture:
  sh_order: 2
  base_resolution: 512
  target_resolution: 4096
  init_dc_value: 0.5

training:
  num_epochs: 2000
  lr: 0.01
  lr_decay: 0.5
  lr_decay_epochs: [500, 1000, 1500]
  batch_size: 4
  resolution_schedule:
    - { epoch: 0, resolution: 512 }
    - { epoch: 300, resolution: 1024 }
    - { epoch: 700, resolution: 2048 }
    - { epoch: 1100, resolution: 4096 }

loss:
  lambda_l1: 1.0
  lambda_ssim: 0.2
  lambda_tv: 0.005

seam_padding:
  dilation_radius: 3
  apply_every_n_epochs: 50

export:
  output_dir: "output"
  format: "gltf"
  sh_truncate_order: -1
```

- [ ] **Step 5: 创建 src/__init__.py**

```python
"""可微烘焙管线 — 高面片数 3D 场景光照与辐射场可微烘焙。"""
```

- [ ] **Step 6: 创建 src/config.py** — 完整的配置 dataclass 系统，从 YAML 加载。包含 DataConfig, TextureConfig, TrainingConfig, LossConfig, SeamPaddingConfig, ExportConfig, Config, load_config()。实现与之前计划中 Task 1 Step 6 完全一致。

- [ ] **Step 7: 创建 main.py** — CLI 入口骨架 (train/export 双模式, --config, --checkpoint 参数)。

- [ ] **Step 8: 创建 .gitignore**

```
data/gt/
data/scene/
data/cameras.json
output/
__pycache__/
*.pyc
*.egg-info/
```

- [ ] **Step 9: 创建 data/.gitkeep 和 tests/__init__.py (空文件)**

- [ ] **Step 10: 初始化 git 并提交**

```bash
git init
git add -A
git commit -m "chore: 项目脚手架 — conda 环境、配置系统、CLI 入口"
```

> **注意:** Task 1 中 config.py 和 main.py 的完整代码见之前计划文件，此处省略以控制篇幅。实现时需包含完整代码。

---

### Task 2: Blender 数据导出 SOP 文档

**Files:**
- Create: `scripts/README.md`

- [ ] **Step 1: 编写 Blender MCP 数据导出 SOP**

内容涵盖:
1. 前置条件 (Blender 3.x+, 已完成减模和 UV 展开)
2. Step 1: 导出低模 OBJ → `data/scene/lowpoly.obj`
3. Step 2: 半球采样相机阵列设置 (50-200 视角, Fibonacci 均匀分布, 含顶部)
4. Step 3: Cycles 渲染 GT 图像 → `data/gt/view_XXXX.png` (256+ 采样)
5. Step 4: 导出 cameras.json (`blender_coordinate: true`, 含 image_path/position/look_at/up/fov_deg/image_size)
6. Step 5: 验证数据完整性
7. 快捷方式: 引用 `scripts/blender_export.py`

cameras.json 格式:
```json
{
  "blender_coordinate": true,
  "cameras": [
    {
      "image_path": "gt/view_0000.png",
      "position": [1.23, 4.56, 7.89],
      "look_at": [0, 0, 0],
      "up": [0, 0, 1],
      "fov_deg": 45.0,
      "image_size": [1920, 1080]
    }
  ]
}
```

- [ ] **Step 2: 提交**

```bash
git add scripts/README.md
git commit -m "docs: Blender MCP 数据导出 SOP"
```

---

### Task 3: Blender 自动化导出脚本

**Files:**
- Create: `scripts/blender_export.py`

- [ ] **Step 1: 编写 Blender 导出脚本**

在 Blender 内运行的 Python 脚本，功能:
1. 自动计算选中对象包围盒 → 确定相机半径
2. Fibonacci 半球采样生成相机阵列
3. 逐相机 Cycles 渲染 → 保存 PNG 到 `data/gt/`
4. 导出 cameras.json (Blender Z-up 坐标系)
5. 导出选中对象为 OBJ → `data/scene/lowpoly.obj`

用户可调参数 (脚本头部):
- `OUTPUT_DIR`: 默认项目 `data/` 目录
- `NUM_VIEWS`: 默认 100
- `CAMERA_RADIUS`: 默认自动计算
- `RENDER_SAMPLES`: 默认 256
- `RESOLUTION`: 默认 1024×1024
- `FOV_DEG`: 默认 45°

用法: Blender Script Editor 运行, 或 `blender --background --python scripts/blender_export.py`

关键实现: `fibonacci_hemisphere()` 函数生成上半球均匀采样点; 每个相机用 `direction.to_track_quat('-Z', 'Y')` 设置朝向; 渲染用 `bpy.ops.render.render(write_still=True)`。

- [ ] **Step 2: 验证脚本可被 Blender 加载无语法错误**

- [ ] **Step 3: 提交**

```bash
git add scripts/blender_export.py
git commit -m "feat: Blender 自动化数据导出脚本"
```

---

### Task 4: 相机模块

**Files:** `src/camera.py`, `tests/test_camera.py`

- [ ] **Step 1: 编写测试** — 7 个测试: roundtrip 坐标转换, Z→Y 映射, from_dict(look_at/rotation), MVP 形状与行列式, torch 版与 numpy 版一致, load_cameras JSON 加载
- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现** — `blender_to_opengl()` 坐标转换, `Camera` dataclass (from_dict/view_matrix/projection_matrix/mvp/mvp_torch), `_quat_rotate()`, `load_cameras()` 从 JSON 加载并自动转换坐标系
- [ ] **Step 4: 运行确认通过** — 7 passed
- [ ] **Step 5: 提交**

---

### Task 5: 网格加载模块

**Files:** `src/mesh.py`, `tests/test_mesh.py`

- [ ] **Step 1: 编写测试** — 3 个测试: load_obj 形状验证, compute_vertex_normals 单位化, to_torch 数据类型
- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现** — `MeshData` dataclass (vertices/faces/uvs/uv_idx, compute_vertex_normals, to_torch), `load_mesh()` 使用 trimesh 加载 OBJ/GLB
- [ ] **Step 4: 运行确认通过** — 3 passed
- [ ] **Step 5: 提交**

---

### Task 6: 球谐(SH)基函数模块

**Files:** `src/sh.py`, `tests/test_sh.py`

- [ ] **Step 1: 编写测试** — 6 个测试: order0/order2 形状, DC 常数验证, decode_sh 形状, DC 视角无关, init_sh_texture DC 非零高阶为零
- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现** — `_C0/_C1/_C2` 常量, `eval_sh_basis(dirs, order)` 返回 [..., 9], `decode_sh(sh, dirs, order)` 内积求和, `init_sh_texture(res, order, dc)` 返回 nn.Parameter [1,H,W,27]
- [ ] **Step 4: 运行确认通过** — 6 passed
- [ ] **Step 5: 提交**

---

### Task 7: GT 图像数据集

**Files:** `src/dataset.py`, `tests/test_dataset.py`

- [ ] **Step 1: 编写测试** — 3 个测试: 长度, item 形状, 遍历全部。测试数据通过 cameras.json 的 image_path 字段关联
- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现** — `GTDataset(Dataset)`: 从 cameras.json 加载相机列表, 通过 image_path 字段精确关联 GT 图像 (不依赖文件名排序), `__getitem__` 返回 (image[3,H,W], Camera), 图像归一化 [0,1]
- [ ] **Step 4: 运行确认通过** — 3 passed
- [ ] **Step 5: 提交**

---

### Task 8: 可微渲染器

**Files:** `src/renderer.py`, `tests/test_renderer.py`

- [ ] **Step 1: 编写测试** — 2 个测试 (需 CUDA, skip if no GPU): render 输出形状 + mask, 梯度回传验证
- [ ] **Step 2: 运行确认失败或 skip**
- [ ] **Step 3: 实现** — `DifferentiableRenderer`: 构造时传入 numpy 网格 + 分辨率, 持有 nvdiffrast RasterizeGLContext. `render(sh_texture, camera)`: MVP 变换 → dr.rasterize → dr.interpolate(uvs + world_pos) → 计算视线方向 → dr.texture 采样 27 通道 SH → eval_sh_basis + 内积解码 RGB → mask 背景
- [ ] **Step 4: 运行确认通过** — 2 passed (CUDA) 或 2 skipped
- [ ] **Step 5: 提交**

---

### Task 9: 损失函数

**Files:** `src/losses.py`, `tests/test_losses.py`

- [ ] **Step 1: 编写测试** — 7 个测试: L1 同/异, SSIM 同/异, TV 常数/噪声, CombinedLoss 标量+梯度
- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现** — `l1_loss()`, `ssim_loss()` (11×11 高斯窗), `tv_loss()` (相邻像素差), `CombinedLoss(nn.Module)`: forward(rendered[H,W,3], gt, mask, sh_texture) → λ₁·L1 + λ₂·(1-SSIM) + λ₃·TV
- [ ] **Step 4: 运行确认通过** — 7 passed
- [ ] **Step 5: 提交**

---

### Task 10: UV Seam Padding

**Files:** `src/seam_padding.py`, `tests/test_seam_padding.py`

- [ ] **Step 1: 编写测试** — 3 个测试: 膨胀填充邻居, 保持原始值不变, 更大半径填充更多
- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现** — `dilate_texture(texture[1,H,W,C], mask[1,H,W,1], radius)`: 用 conv2d 做邻域平均, 仅在空白区域填入膨胀值, 有效区域保持不变
- [ ] **Step 4: 运行确认通过** — 3 passed
- [ ] **Step 5: 提交**

---

### Task 11: 训练主循环 (Coarse-to-Fine)

**Files:** `src/trainer.py`, `tests/test_trainer.py`

- [ ] **Step 1: 编写集成测试** (需 CUDA) — 创建最小训练数据 (平面 OBJ + 4 视角随机图 + cameras.json), Config 用 16×16 分辨率跑 3 epoch, 验证 SH 纹理已更新
- [ ] **Step 2: 运行确认失败或 skip**
- [ ] **Step 3: 实现** — `Trainer`: 加载网格和数据集, init_sh_texture, Adam + MultiStepLR. `_current_resolution(epoch)` 查表, `_resize_sh_texture(new_res)` 双线性上采样 + 重建优化器, `_apply_seam_padding()` 每 N epoch. `train()`: 主循环 (检查分辨率→随机采样视角→render→GT 缩放→loss→backward→step→scheduler→padding)
- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: 提交**

---

### Task 12: 资产导出

**Files:** `src/exporter.py`, `tests/test_exporter.py`

- [ ] **Step 1: 编写测试** — 3 个测试: diffuse PNG, 9 张 SH 通道 PNG, glTF .glb 文件
- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现** — `export_diffuse_texture()` DC×C₀ → PNG, `export_sh_channels()` 拆 9 张 PNG, `export_gltf()` trimesh 打包 .glb
- [ ] **Step 4: 运行确认通过** — 3 passed
- [ ] **Step 5: 提交**

---

### Task 13: CLI 入口完整集成

**Files:** 修改 `main.py`

- [ ] **Step 1: 完善 main.py** — train 模式: Trainer(cfg) → train() → 保存 sh_texture.pt. export 模式: 加载 .pt → 按 format (gltf/custom) 导出
- [ ] **Step 2: 验证 `python main.py --help` 无报错**
- [ ] **Step 3: 提交**

---

## Spec 覆盖度自查

| 设计文档章节 | 对应 Task |
|---|---|
| 1. 数据接口 (目录结构/cameras.json) | Task 4 (camera), Task 5 (mesh), Task 7 (dataset) |
| 1.3 Blender 数据导出 SOP | Task 2 (scripts/README.md) |
| 1.3 Blender 自动化导出脚本 | Task 3 (scripts/blender_export.py) |
| 2. 模块架构 (10 个模块) | Task 4-12 |
| 2.2 SH 纹理表示 [1,H,W,27] | Task 6 (sh.py init_sh_texture) |
| 3. 前向渲染管线 | Task 8 (renderer.py) |
| 4.1 训练主循环 | Task 11 (trainer.py) |
| 4.2 Coarse-to-Fine 分辨率调度 | Task 11 (trainer.py _resize_sh_texture) |
| 4.3 UV Seam Padding | Task 10 (seam_padding.py) |
| 4.4 损失函数 L1+SSIM+TV | Task 9 (losses.py) |
| 5. 导出 glTF / custom | Task 12 (exporter.py) |
| 6. 配置系统 | Task 1 (config.py) |
| 7. 环境准备 (conda) | Task 1 (environment.yml) |
