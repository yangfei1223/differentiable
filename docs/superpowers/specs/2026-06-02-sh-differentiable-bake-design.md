# 可微烘焙管线 MVP 设计文档

## 概述

基于 PyTorch + nvdiffrast 构建可微渲染烘焙管线，将高模多视角 GT 渲染图反向优化为绑定在低模上的 2 阶球谐(SH)辐射场纹理。最终资产面向移动端部署，但训练阶段保留完整高分辨率 + 2阶 SH。

**MVP 范围：** 核心训练管线 + 资产导出。不含几何偏差掩码、训练监控面板、分布式训练。

**环境：** Windows + NVIDIA GPU (CUDA)，团队熟悉 nvdiffrast。

---

## 1. 数据接口

### 1.1 目录结构

Blender MCP 导出数据按以下结构组织：

```
data/
├── scene/
│   └── lowpoly.obj          # 低模 (已展UV, 无重叠)
├── gt/
│   ├── view_0000.png        # GT 渲染图 (Blender Cycles)
│   ├── view_0001.png
│   └── ...
└── cameras.json             # 相机参数
```

### 1.2 cameras.json 格式

```json
{
  "blender_coordinate": true,
  "cameras": [
    {
      "image_path": "gt/view_0000.png",
      "position": [x, y, z],
      "look_at": [x, y, z],
      "up": [0, 1, 0],
      "fov_deg": 45.0,
      "image_size": [1920, 1080]
    }
  ]
}
```

- `blender_coordinate: true` 表示相机坐标是 Blender Z-up 坐标系，`camera.py` 自动转为 OpenGL Y-up (视线指向 -Z)
- `image_path` 相对于 JSON 文件所在目录
- 也支持 `position + rotation (quaternion wxyz)` 格式

### 1.3 低模网格

- 格式：OBJ 或 GLB，由 Blender MCP 导出
- 要求：UV 完全展平且无重叠
- 提取属性：vertices `[N,3]`、faces `[M,3]`、uvs `[K,2]`、uv_idx `[M,3]`

---

## 2. 模块架构

```
src/
├── camera.py          # 相机加载 + Blender→OpenGL 坐标转换 + MVP 矩阵
├── mesh.py            # OBJ/GLB 网格加载, 提取 vertices/faces/uvs/uv_idx
├── dataset.py         # GT 图像 + 相机关联的 PyTorch Dataset
├── sh.py              # 球谐基函数评估 + 颜色解码 + 纹理初始化
├── renderer.py        # nvdiffrast 可微渲染器 (光栅化 + SH 采样解码)
├── losses.py          # L1 + SSIM + TV Loss
├── seam_padding.py    # UV 边界膨胀算子
├── trainer.py         # Coarse-to-Fine 训练主循环
├── exporter.py        # 资产导出 (Diffuse PNG / 9x SH PNG)
└── config.py          # YAML 配置加载 + dataclass
```

辅助文件：
- `main.py` — CLI 入口 (train / export 双模式)
- `configs/default.yaml` — 默认配置

### 2.1 模块接口

**camera.py → Camera**
- `Camera.from_dict(d)` — 从 JSON 字典构建
- `camera.mvp()` — 返回 4x4 MVP numpy 矩阵
- `camera.mvp_torch()` — 返回 `[1, 4, 4]` torch tensor
- `load_cameras(json_path)` — 加载完整相机列表，自动处理坐标系转换

**mesh.py → MeshData**
- `load_mesh(path)` — 加载 OBJ/GLB
- `mesh.compute_vertex_normals()` — 计算顶点法线
- `mesh.to_torch()` — 转为 torch tensor 元组

**sh.py**
- `eval_sh_basis(dirs, order)` — 评估 SH 基函数，返回 `[..., 9]`
- `decode_sh(sh_texture, view_dirs, order)` — 解码 RGB，返回 `[..., 3]`
- `init_sh_texture(resolution, sh_order, init_dc)` — 返回 `nn.Parameter [1, H, W, 27]`

**dataset.py → GTDataset**
- 从 cameras.json 加载相机列表，每条记录含 `image_path` 字段
- 通过 `image_path` 精确关联 GT 图像与相机参数（不依赖文件名排序）
- `__getitem__(idx)` 返回 `(image_tensor [3,H,W], Camera)`
- 图像统一归一化到 `[0, 1]` float32

**renderer.py → DifferentiableRenderer**
- 构造：传入 numpy 网格数据 + 分辨率
- `render(sh_texture, camera)` — 前向渲染，返回 `(rgb [1,H,W,3], mask [1,H,W])`

**losses.py → CombinedLoss**
- `forward(rendered, gt, mask, sh_texture)` — 返回标量损失
- 组成：`λ₁·L1 + λ₂·(1-SSIM) + λ₃·TV`

**trainer.py → Trainer**
- 构造：传入 `Config`
- `train()` — 执行完整训练循环
- `get_sh_texture()` — 获取最终 SH 纹理

**exporter.py**
- `export_diffuse_texture(sh_tex, path)` — 导出 DC 分量 PNG
- `export_sh_channels(sh_tex, dir)` — 导出 9 张独立 PNG
- `export_gltf(vertices, faces, sh_tex, path)` — 导出 .glb

### 2.2 SH 纹理表示

MVP 采用单张 `[1, H, W, 27]` 的 `nn.Parameter`，27 通道 = 9 个 SH 系数 × 3 RGB 通道。

渲染时通过 nvdiffrast `dr.texture` 一次性采样 27 通道，然后 `decode_sh` 将其与视线方向的 SH 基函数内积得到 RGB。

**初始化：** 0阶 DC 分量 (前3通道) 初始化为 `init_dc / C0`（默认 0.5），其余 24 通道初始化为 0。

> V2 优化：拆为 9 张 `[1,H,W,3]` 分批采样以降低显存峰值。

---

## 3. 前向渲染管线

```
vertices ──→ transform_pos(mvp) ──→ clip space
                                        │
                                   dr.rasterize
                                        │
                              ┌─────────┴─────────┐
                              │                   │
                        dr.interpolate        dr.interpolate
                        (uvs → texcoords)    (verts → world_pos)
                              │                   │
                              │              camera_pos - world_pos
                              │                   │
                              │              normalize → view_dir
                              │                   │
                        dr.texture(SH_tex)         │
                              │                   │
                        sampled_sh [1,H,W,27]     │
                              │                   │
                              └──────┬────────────┘
                                     │
                            eval_sh_basis(view_dir)
                            + sum(coeff × basis)
                                     │
                              rgb [1,H,W,3]
                                     │
                            × mask (rast[...,3] > 0)
                                     │
                              output [1,H,W,3]
```

---

## 4. 训练流程

### 4.1 主循环

```
for epoch in range(num_epochs):
  ① 检查分辨率调度, 必要时双线性上采样 SH 纹理
  ② 随机采样 batch_size 个视角
  ③ for each view:
     ├─ 前向渲染: render(sh_tex, camera)
     ├─ 缩放 GT 到当前渲染分辨率
     ├─ 计算联合损失 (L1 + SSIM + TV)
     ├─ 反向传播 + Adam 更新
  ④ 学习率衰减 (MultiStepLR)
  ⑤ 每 N epoch: UV seam padding 膨胀
```

### 4.2 Coarse-to-Fine 分辨率调度

| Epoch | 分辨率 | 说明 |
|-------|--------|------|
| 0-299 | 512×512 | 低频光照收敛 |
| 300-699 | 1024×1024 | 中频细节 |
| 700-1099 | 2048×2048 | 高频纹理 |
| 1100+ | 4096×4096 | 最终质量 |

分辨率升级时：
- SH 纹理通过 `F.interpolate(bilinear)` 上采样
- 优化器重建（不继承动量，避免大分辨率下初始震荡）

### 4.3 UV Seam Padding

每 50 epoch 执行一次：
- 计算有效像素掩码（SH 纹理非零区域）
- 对空白区域用邻域有效像素的平均颜色膨胀填充
- 防止 mipmap 采样在 UV 岛边界产生黑边

### 4.4 损失函数

$$L_{total} = \lambda_1 \cdot L_1(I_{render}, I_{gt}) + \lambda_2 \cdot (1 - \text{SSIM}(I_{render}, I_{gt})) + \lambda_3 \cdot \text{TV}(S)$$

默认权重：`λ₁=1.0, λ₂=0.2, λ₃=0.005`

- L1：像素级绝对误差，掩码内平均
- SSIM：结构相似性，11×11 高斯窗口
- TV：Total Variation 正则化，平滑未观测区域

> V2 新增：几何偏差掩码（高模法线 vs 低模法线 > 30° 时屏蔽 loss）

---

## 5. 导出策略

### 模式 A: `gltf` (标准兼容)

- 提取 SH 0阶 DC 分量 (前3通道 × C0)，输出 `diffuse.png`
- 低模 + diffuse 贴图打包为 `.glb`，写入 `emissiveTexture`
- 任何标准 glTF 渲染器可打开

### 模式 B: `custom` (高保真)

- 27 通道拆分为 9 张 PNG (`sh_coeff_00.png` ~ `sh_coeff_08.png`)
- 输出低模 OBJ + 9 张贴图
- 运行时需自定义 Shader 完成 SH 重建

### SH 截断

`sh_truncate_order` 参数：
- `-1`：全部导出 (默认)
- `0`：仅 DC 分量 (3 通道 → 1 张贴图)
- `1`：DC + 一阶 (12 通道 → 4 张贴图)

---

## 6. 配置系统

YAML 驱动，通过 `config.py` 的 dataclass 加载。主要配置项：

- `data.mesh_path` / `data.gt_dir` / `data.camera_path`
- `texture.sh_order` / `texture.base_resolution` / `texture.target_resolution`
- `training.num_epochs` / `training.lr` / `training.batch_size` / `training.resolution_schedule`
- `loss.lambda_l1` / `loss.lambda_ssim` / `loss.lambda_tv`
- `seam_padding.dilation_radius` / `seam_padding.apply_every_n_epochs`
- `export.format` / `export.sh_truncate_order`

---

## 7. 环境准备

使用 cona 创建独立虚拟环境：

```bash
conda create -n differentiable python=3.10 -y
conda activate differentiable
```

### 7.1 CUDA 与 PyTorch 安装

根据本机 CUDA 版本安装对应 PyTorch（以 CUDA 12.1 为例）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 7.2 nvdiffrast 安装

nvdiffrast 对 CUDA 与 C++ 编译器版本敏感。Windows 上需要 Visual Studio Build Tools：

```bash
pip install git+https://github.com/NVlabs/nvdiffrast.git
```

若编译失败，检查：
- CUDA toolkit 版本与 PyTorch CUDA 版本一致
- Visual Studio Build Tools 已安装（C++ 桌面开发工作负载）

### 7.3 其余依赖

```bash
pip install numpy opencv-python trimesh PyYAML Pillow tqdm imageio
```

### 7.4 验证

```python
import torch
print(f"PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}")

import nvdiffrast.torch as dr
print("nvdiffrast OK")
```

---

## 8. 技术栈

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | 3.10+ | 运行时 |
| PyTorch | 2.x | 自动微分 + GPU 计算 |
| nvdiffrast | latest | 可微光栅化 |
| numpy | 1.24+ | 数值计算 |
| opencv-python | 4.8+ | 图像读写 |
| trimesh | 4.0+ | 网格加载 |
| Pillow | 10.0+ | PNG 导出 |
| PyYAML | 6.0+ | 配置加载 |

---

## 9. V2 路线图 (MVP 不含)

- 几何偏差掩码（需 Blender MCP 同步导出高模法线图）
- TensorBoard / WandB 训练监控
- SH 纹理拆分（9×3 通道分批采样以降低显存）
- 多 GPU 分布式训练
- Blender MCP 集成脚本（自动化数据导出流水线）
- 断点续训 (checkpoint save/load)
- WebUI — 基于 Gradio/Streamlit 构建，提供训练参数调整、实时 loss 曲线预览、结果对比等功能，降低非技术用户的使用门槛
