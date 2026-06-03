# 可微烘焙管线 (Differentiable Baking Pipeline)

基于 PyTorch + nvdiffrast 的可微渲染烘焙管线，将高模多视角 GT 渲染图反向优化为绑定在低模上的 2 阶球谐(SH)辐射场纹理。

## 功能

- **半球采样相机生成** — Fibonacci 均匀分布，支持随机 FOV 波动
- **Cycles GT 渲染** — 自动逐相机渲染高质量 GT 图像
- **2 阶球谐纹理优化** — 27 通道 SH 参数，支持视角相关高光
- **Coarse-to-Fine 训练** — 512→1024→2048→4096 多分辨率渐进训练
- **UV Seam Padding** — 自动膨胀填充 UV 岛边界，消除黑边
- **双模式导出** — 标准 glTF (Diffuse) 或自定义 9 通道 SH PNG

## 项目结构

```
├── src/
│   ├── camera.py          # 相机加载 & Blender→OpenGL 坐标转换
│   ├── mesh.py             # OBJ/GLB 网格加载
│   ├── dataset.py          # GT 图像 + 相机关联 Dataset
│   ├── sh.py               # 球谐基函数评估 & 颜色解码
│   ├── renderer.py         # nvdiffrast 可微渲染器
│   ├── losses.py           # L1 + SSIM + TV Loss
│   ├── seam_padding.py     # UV 边界膨胀
│   ├── trainer.py          # Coarse-to-Fine 训练主循环
│   ├── exporter.py         # 资产导出
│   └── config.py           # YAML 配置系统
├── scripts/
│   ├── blender_export.py   # Blender 自动化数据导出脚本
│   └── README.md           # Blender MCP 操作 SOP
├── configs/default.yaml    # 默认训练配置
├── data/                   # 训练数据 (gitignore)
│   ├── scene/lowpoly.glb   # 低模
│   ├── gt/                  # 200 张 GT 渲染图
│   └── cameras.json        # 相机参数
├── tests/                  # 47 个单元测试
└── main.py                 # CLI 入口
```

## 环境搭建

```bash
# 1. 创建 conda 环境
conda env create -f environment.yml
conda activate differentiable

# 2. 安装 CUDA 版 PyTorch (根据本机 CUDA 版本调整)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. 安装 nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
```

## 数据准备

### 方式一：通过 Blender MCP

参考 `scripts/README.md` 中的 SOP，手动操作 Blender 导出低模、渲染 GT、导出相机参数。

### 方式二：自动化脚本

在 Blender Script Editor 中运行：

```python
exec(open("scripts/blender_export.py").read())
```

脚本会自动完成：导出低模 → 生成半球相机 → Cycles 渲染 → 导出 cameras.json。

## 训练

```bash
python main.py --config configs/default.yaml --mode train
```

训练过程中 SH 纹理会按分辨率阶梯逐步提升：

| Epoch | 分辨率 | 阶段 |
|-------|--------|------|
| 0-299 | 512×512 | 低频光照 |
| 300-699 | 1024×1024 | 中频细节 |
| 700-1099 | 2048×2048 | 高频纹理 |
| 1100+ | 4096×4096 | 最终质量 |

## 导出

```bash
# glTF 模式 (标准兼容)
python main.py --mode export --checkpoint output/sh_texture.pt --config configs/default.yaml

# 自定义模式 (9 张 SH 通道 PNG)
# 修改 configs/default.yaml 中 export.format 为 "custom"
python main.py --mode export --checkpoint output/sh_texture.pt
```

## 测试

```bash
pytest tests/ -v
```

## 技术栈

- Python 3.10, PyTorch 2.x, nvdiffrast
- Blender (数据制备)
- trimesh, OpenCV, Pillow

## License

Private
