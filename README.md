# 可微烘焙管线 (Differentiable Baking Pipeline)

基于 PyTorch + nvdiffrast 的可微渲染烘焙管线，将高模多视角 GT 渲染图反向优化为低模 SH 辐射场纹理，面向移动端部署。

## 功能

- **半球采样相机生成** — Fibonacci 均匀分布，支持随机 FOV 波动
- **Cycles GT 渲染** — 自动逐相机渲染高质量 GT 图像（Blender --background 批量渲染）
- **3DGS 风格 SH 参数化** — DC / 高阶分离存储，独立学习率（高阶 lr = DC lr × rest_lr_ratio）
- **动态 SH Order** — 支持 order 0 / 1 / 2，通过配置文件切换
- **Coarse-to-Fine 训练** — 512→1024→2048 多分辨率渐进训练
- **UV Seam Padding** — 自动膨胀填充 UV 岛边界，消除黑边
- **数据集管理** — 按 `{scene}_{yymmdd}` 组织训练数据和输出，自动关联
- **视频自动相机** — 根据 mesh bounding box 自适应计算相机参数
- **调试输出** — 2×2 compare atlas (GT / Full SH / DC / High Freq) + 多视频 (Full / DC / HF)

## 项目结构

```
├── src/
│   ├── camera.py          # 相机加载 & Blender→OpenGL 坐标转换
│   ├── mesh.py             # OBJ/GLB 网格加载 (含 UV 修正)
│   ├── dataset.py          # GT 图像 + 相机关联 Dataset
│   ├── sh.py               # 球谐基函数 & RGB2SH/SH2RGB (3DGS 约定)
│   ├── renderer.py         # nvdiffrast 可微渲染器
│   ├── losses.py           # L1 + Masked SSIM + TV Loss
│   ├── seam_padding.py     # UV 边界膨胀
│   ├── trainer.py          # 训练主循环 + 调试输出
│   ├── video.py            # 环绕视频渲染 (自动相机参数)
│   ├── exporter.py         # 资产导出 (Diffuse / SH 通道 / glTF)
│   ├── utils.py            # 可视化工具
│   └── config.py           # YAML 配置系统
├── scripts/
│   ├── blender_export.py   # Blender 数据导出脚本 (高模 GT + 低模 + cameras)
│   └── run_ablation.py     # SH0 vs SH2 对照实验脚本
├── configs/
│   ├── default.yaml        # 默认配置 (SH order 2)
│   ├── train_sh0.yaml      # SH order 0 配置
│   ├── train_1k.yaml       # 1K 分辨率快速训练
│   ├── quick_test.yaml     # 快速验证
│   └── train_helmet.yaml   # 头盔数据集配置
├── data/                   # 训练数据 (gitignore)
│   └── {scene}_{yymmdd}/
│       ├── scene/lowpoly.glb
│       ├── gt/              # 多视角 GT 渲染图
│       └── cameras.json     # 相机参数
├── output/                  # 训练输出 (gitignore)
│   └── {scene}_{yymmdd}/
│       ├── curves.png
│       ├── sh_texture.pt
│       └── epoch*/
│           ├── compare_*.png  (2×2: GT / Full / DC / HF)
│           ├── diffuse.png
│           ├── orbit.mp4      (Full SH)
│           ├── orbit_dc.mp4   (DC only)
│           └── orbit_hf.mp4   (High Freq = Full - DC)
├── asset/                   # Blender 工程文件
│   └── scene.blend
├── tests/                   # 50 个单元测试
└── main.py                  # CLI 入口
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

### 自动化脚本（推荐）

```bash
blender --background asset/scene.blend --python scripts/blender_export.py
```

脚本会自动完成：导出低模 → 生成半球相机 → Cycles 渲染 → 导出 cameras.json。

输出到 `data/{SCENE_NAME}_{yymmdd}/`，参数在脚本头部配置。

### 通过 Blender MCP

直接操作 Blender 导出低模、渲染 GT、导出相机参数。

## 训练

```bash
# 默认配置 (SH order 2)
python main.py --config configs/default.yaml --mode train

# SH order 0 (纯 diffuse)
python main.py --config configs/train_sh0.yaml --mode train

# 断点续训
python main.py --config configs/default.yaml --mode train --resume output/{dataset}/epoch{N}/sh_texture.pt
```

输出自动按数据集名分目录：`output/{scene}_{yymmdd}/`

## 导出

```bash
# Diffuse 贴图
python main.py --mode export --checkpoint output/{dataset}/sh_texture.pt

# 环绕视频
python main.py --mode video --checkpoint output/{dataset}/sh_texture.pt
```

## 测试

```bash
pytest tests/ -v
```

## V0.2 变更

- 3DGS 风格 SH 参数化：DC / 高阶分离存储，`RGB2SH` / `SH2RGB` 约定
- 独立学习率：`rest_lr_ratio` 可配置（默认 0.05，即 1/20）
- UV 修正：自动处理 glTF 模型 V 坐标在 [-1, 0] 的问题
- 渲染器 boundary_mode 改为 clamp
- 视频自动相机：根据 mesh bounding box 自适应 radius/height/FOV
- 调试输出：2×2 compare atlas + 3 个视频 (Full / DC / HF)
- HF 可视化：Full - DC 帧级差分，展示高频净贡献
- 数据集管理：输入输出按 `{scene}_{yymmdd}` 目录组织
- Blender 导出脚本：增加验证、自动选择高模、兼容集合不存在的情况
- 兼容旧格式 checkpoint 断点续训

## 技术栈

- Python 3.10, PyTorch 2.x, nvdiffrast
- Blender 5.1 (数据制备)
- trimesh, OpenCV, Pillow, matplotlib

## License

Private
