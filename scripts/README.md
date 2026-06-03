# Blender MCP 数据制备 SOP

## 概述

本文档描述如何通过 Blender 准备可微烘焙（Differentiable Baking）管线所需的训练数据。

核心流程：在 Blender 中对高模进行多视角 Cycles 渲染，导出低模几何、GT 渲染图和相机参数，供下游 PyTorch + nvdiffrast 训练管线使用。

> **快捷方式**：如果希望一键完成所有步骤，可直接运行 [`blender_export.py`](./blender_export.py) 脚本。详见文末 [快捷方式](#快捷方式) 章节。

---

## 前置条件

| 项目 | 要求 |
|------|------|
| Blender | 3.x 及以上版本（推荐 3.6 LTS 或 4.x） |
| 高模 | 已完成材质、灯光、场景搭建，可直接渲染 |
| 低模 | 已完成减面（Decimate）、UV 展开（无重叠），与高模位置对齐 |
| 磁盘空间 | 每个视角约 3-4 MB（1024×1024 PNG），100 视角 ≈ 350 MB |

### 文件夹结构

```
data/
├── scene/
│   └── lowpoly.obj          # 导出的低模
├── gt/
│   ├── view_0000.png        # GT 渲染图
│   ├── view_0001.png
│   └── ...
└── cameras.json             # 相机参数
```

---

## Step 1: 导出低模

### 操作步骤

1. 在 Blender **对象模式**下，选中已减面并 UV 展开的低模对象
2. 菜单栏 → `File` → `Export` → `Wavefront (.obj)`
3. 保存路径设为 `data/scene/lowpoly.obj`
4. 在导出选项中确认以下设置：

| 选项 | 设置 | 说明 |
|------|------|------|
| Selection Only | ✅ 勾选 | 仅导出选中的低模 |
| Apply Modifiers | ✅ 勾选 | 应用修改器后的最终几何 |
| Write Materials | ✅ 勾选 | 导出材质信息 |
| Objects as OBJ Objects | ✅ | 保持对象结构 |
| Forward | Y Forward | 默认即可 |
| Up | Z Up | Blender 坐标系 |

5. 点击 `Export OBJ` 完成导出

### 注意事项

- 确保 UV 岛之间没有重叠，否则纹理烘焙会出现采样冲突
- 低模的缩放变换（Scale）需要应用（`Ctrl+A` → Apply Scale），否则 OBJ 顶点坐标不一致

---

## Step 2: 半球采样相机

### 采样策略

使用 **Fibonacci 半球均匀采样** 生成相机位置。该策略确保：

- 在上半球面上均匀分布相机，避免观测盲区
- 包含正上方（Top）视角，覆盖模型顶部
- 适用于任意闭合模型的全方位数据采集

### 参数设置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 视角数量 | 50 - 200 | 100 为推荐默认值；模型细节越多需要越多视角 |
| FOV | 45° - 60° | 45° 最接近人眼，60° 适合较宽场景 |
| 采样半径 | 包围模型 | 自动计算为模型包围盒半径的 1.5-2.0 倍 |
| 分布方式 | Fibonacci 螺旋 | 上半球面均匀分布 |
| 额外视角 | 顶部 1 个 | 沿 -Y 轴正上方俯视 |

### Fibonacci 半球采样原理

```
黄金角 = π * (3 - √5)
对 i = 0, 1, ..., n-1:
    θ = arccos(1 - (i + 0.5) / n)     # 极角，仅上半球
    φ = i * 黄金角                       # 方位角
    x = r * sin(θ) * cos(φ)
    y = r * sin(θ) * sin(φ)
    z = r * cos(θ)
```

> ⚠️ **切勿**仅使用固定高度的环绕拍摄，这会导致低模顶部/底部因缺乏观测而产生严重伪影。

---

## Step 3: Cycles 渲染 GT

### 渲染设置

1. 切换渲染引擎为 **Cycles**
2. 配置以下参数：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 采样数 | ≥ 256 | 512 更佳；低于 128 会出现明显噪点 |
| 分辨率 | 1024 × 1024 | 正方形分辨率，便于训练 |
| 降噪 | OptiX 或 OpenImage | 可选，减少 GT 噪点 |
| 设备 | GPU Compute | 显著加速渲染 |
| 色彩管理 | sRGB | 标准线性工作流 |

### 渲染流程

对每个相机位置执行：

1. 创建临时相机对象，放置在采样点上
2. 设置相机朝向模型中心（Look-At）
3. 设置 FOV 和分辨率
4. 渲染当前视角
5. 保存渲染结果为 PNG：
   ```
   data/gt/view_XXXX.png    # XXXX 为四位零填充索引，如 view_0000.png
   ```
6. 删除临时相机

### 注意事项

- 使用透明背景（Film → Transparent）或统一背景色，便于后续训练时生成掩码
- 渲染结果保持 **sRGB 色彩空间**，与训练管线保持一致

---

## Step 4: 导出相机参数

### 需要记录的参数

对每个相机视角记录以下信息：

| 字段 | 类型 | 说明 |
|------|------|------|
| `position` | `[x, y, z]` | 相机世界坐标位置 |
| `look_at` | `[x, y, z]` | 相机注视点（通常为模型中心） |
| `up` | `[x, y, z]` | 相机上方方向（通常为 `[0, 0, 1]`） |
| `fov_deg` | `float` | 垂直视场角（度） |
| `image_size` | `[w, h]` | 图像分辨率 |
| `image_path` | `string` | 对应渲染图的相对路径 |

### 保存位置

```
data/cameras.json
```

---

## cameras.json 格式

```json
{
  "blender_coordinate": true,
  "cameras": [
    {
      "position": [1.234, -0.567, 0.891],
      "look_at": [0.0, 0.0, 0.0],
      "up": [0.0, 0.0, 1.0],
      "fov_deg": 45.0,
      "image_size": [1024, 1024],
      "image_path": "gt/view_0000.png"
    },
    {
      "position": [-0.321, 1.456, 0.234],
      "look_at": [0.0, 0.0, 0.0],
      "up": [0.0, 0.0, 1.0],
      "fov_deg": 45.0,
      "image_size": [1024, 1024],
      "image_path": "gt/view_0001.png"
    }
  ]
}
```

### 字段说明

- **`blender_coordinate`**: `true` 表示使用 Blender 坐标系（Z-up, 右手系）。下游训练管线会据此自动转换为 OpenGL 标准坐标（Y-up, 视线指向 -Z）。
- **`image_path`**: 相对于 `data/` 目录的路径，方便训练脚本统一加载。
- **`position`**: 相机在世界空间中的位置（Blender 坐标系）。
- **`look_at`**: 相机注视的目标点，通常为模型包围盒中心。

---

## Step 5: 验证数据完整性

导出完成后，逐项检查以下清单：

### ✅ 检查清单

- [ ] **低模 OBJ 存在**：`data/scene/lowpoly.obj` 文件存在且非空
- [ ] **低模 UV 有效**：在 Blender 中重新导入 OBJ，确认 UV 展开保留完整
- [ ] **GT 图像数量匹配**：`data/gt/` 下 PNG 文件数量等于相机数量
- [ ] **GT 图像命名规范**：文件名为 `view_XXXX.png`（四位零填充）
- [ ] **GT 图像内容正确**：随机抽查几张渲染图，确认模型居中、无全黑/全白
- [ ] **cameras.json 存在且格式正确**：JSON 解析无报错
- [ ] **cameras.json 条目数量**：`cameras` 数组长度等于 GT 图像数量
- [ ] **image_path 对应正确**：每条相机记录的 `image_path` 指向实际存在的 PNG 文件
- [ ] **blender_coordinate 标记**：值为 `true`
- [ ] **相机覆盖均匀**：无大片空白区域（可用脚本可视化相机位置分布）

### 快速验证脚本

```bash
# 检查文件数量
python -c "
import json, os
cams = json.load(open('data/cameras.json'))
gt_files = [f for f in os.listdir('data/gt') if f.endswith('.png')]
print(f'Cameras: {len(cams[\"cameras\"])}, GT images: {len(gt_files)}')
assert len(cams['cameras']) == len(gt_files), 'Mismatch!'
print('OK ✓')
"
```

---

## 快捷方式

上述所有步骤可通过 [`blender_export.py`](./blender_export.py) 脚本一键完成。

### 使用方法

```bash
# 在 Blender 中运行（推荐通过命令行调用）
blender --background your_scene.blend --python scripts/blender_export.py
```

或在 Blender 内部脚本编辑器中直接运行。

### 脚本参数

脚本顶部有可配置参数：

```python
OUTPUT_DIR = "data"       # 输出目录
NUM_VIEWS = 100           # 采样视角数量
CAMERA_RADIUS = 0         # 0 = 自动计算（推荐）
RENDER_SAMPLES = 256      # Cycles 采样数
RESOLUTION = 1024         # 渲染分辨率
FOV_DEG = 45              # 视场角
TARGET_OBJECT = None      # None = 使用当前选中对象
```

根据实际场景调整以上参数后运行即可。
