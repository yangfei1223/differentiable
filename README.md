# 可微烘焙管线 (Differentiable Baking Pipeline)

基于 PyTorch + nvdiffrast 的可微渲染烘焙管线，将高模多视角 GT 渲染图反向优化为低模纹理。支持 **SH 辐射场** 和 **Split-Sum PBR** 两种着色模型，面向移动端部署。

## 训练结果

| 场景 | 着色模型 | PSNR | 纹理分辨率 |
|------|---------|------|-----------|
| 钢琴 | SH (order 2) | 20.37 dB | 2048×2048 |
| 钢琴 | PBR (split-sum) | **21.41 dB** | 2048×2048 |
| 头盔 | SH (order 2) | 13.19 dB | 2048×2048 |
| 头盔 | PBR (split-sum) | **21.97 dB** | 2048×2048 |

> 头盔含金属面罩，PBR split-sum 捕捉镜面反射提升 +8.8 dB；钢琴以漫反射为主，PBR 仍有 +1.0 dB 增益。

## 功能

### 通用
- **半球采样相机生成** — Fibonacci 均匀分布，支持随机 FOV 波动
- **Cycles GT 渲染** — 自动逐相机渲染高质量 GT 图像（Blender `--background` 批量渲染）
- **Coarse-to-Fine 训练** — 512→1024→2048 多分辨率渐进训练
- **UV Seam Padding** — 自动膨胀填充 UV 岛边界，消除黑边
- **数据集管理** — 按 `{scene}_{yymmdd}` 组织训练数据和输出，自动关联
- **视频自动相机** — 根据 mesh bounding box 自适应计算相机参数

### SH 着色 (v0.2)
- **3DGS 风格 SH 参数化** — DC / 高阶分离存储，独立学习率（高阶 lr = DC lr × rest_lr_ratio）
- **动态 SH Order** — 支持 order 0 / 1 / 2，通过配置文件切换
- **调试输出** — 2×2 compare atlas (GT / Full SH / DC / High Freq) + 多视频 (Full / DC / HF)

### PBR Split-Sum (v0.3)
- **Split-Sum 近似** — diffuse irradiance + prefiltered specular + BRDF LUT (Karis 2014)
- **8ch PBR 材质贴图** — base_color (3) + roughness (1) + metallic (1) + normal_xyz (3)，sigmoid 约束
- **Tangent-Space 法线贴图** — Mikktspace 风格切线计算，TBN 变换到世界空间
- **HDR 环境贴图** — Equirect 参数化，softplus 解码，nvdiffrast `dr.texture` 内置 mipmap
- **GGX BRDF LUT** — 全 PyTorch 向量化 importance sampling，能量守恒 (A+B ≤ 1)
- **联合优化** — 材质贴图 + 环境贴图同时优化，TV + L2 正则化防爆炸
- **分量视频** — Diffuse / Specular 分离环绕视频
- **可插拔着色模型** — `ShadingModel` 协议 + 工厂函数，SH/PBR 透明切换

## 项目结构

```
├── src/
│   ├── camera.py              # 相机加载 & Blender→OpenGL 坐标转换
│   ├── mesh.py                # OBJ/GLB 网格加载 + Mikktspace 切线计算
│   ├── dataset.py             # GT 图像 + 相机关联 Dataset
│   ├── sh.py                  # 球谐基函数 & RGB2SH/SH2RGB
│   ├── renderer.py            # nvdiffrast 可微渲染器 (7 值输出含 TBN)
│   ├── losses.py              # L1 + Masked SSIM + TV Loss
│   ├── seam_padding.py        # UV 边界膨胀
│   ├── trainer.py             # 训练主循环 + env TV/L2 正则化
│   ├── video.py               # 环绕视频渲染
│   ├── exporter.py            # 资产导出 (Diffuse / SH 通道 / glTF)
│   ├── utils.py               # 可视化工具
│   ├── config.py              # YAML 配置系统
│   └── shading/
│       ├── __init__.py        # create_shading_model 工厂
│       ├── base.py            # ShadingModel 协议
│       ├── logger.py          # ShadingLogger 基类 + 工厂
│       ├── sh_model.py        # SHShadingModel
│       ├── sh_logger.py       # SH 调试日志
│       ├── pbr_model.py       # PBRShadingModel (split-sum + TBN)
│       ├── pbr_logger.py      # PBR 调试日志 + 分量视频
│       └── pbr/
│           ├── __init__.py
│           ├── material.py    # 8ch sigmoid 材质参数化
│           ├── env_map.py     # EnvironmentMap (nn.Module)
│           └── brdf_lut.py    # GGX BRDF LUT generation
├── scripts/
│   ├── blender_export.py      # Blender 数据导出脚本
│   ├── run_ablation.py        # SH0 vs SH2 对照实验
│   └── README.md              # 数据制备 SOP
├── configs/
│   ├── default.yaml           # 默认配置 (SH order 2)
│   ├── train_sh0.yaml         # SH order 0
│   ├── train_1k.yaml          # 1K 分辨率快速训练
│   ├── train_helmet.yaml      # 头盔 SH 配置
│   ├── train_pbr.yaml         # 头盔 PBR 配置
│   ├── train_pbr_piano.yaml   # 钢琴 PBR 配置
│   └── quick_test.yaml        # 快速验证
├── tests/                     # 单元测试
├── data/                      # 训练数据 (gitignore)
│   └── {scene}_{yymmdd}/
│       ├── scene/lowpoly.glb
│       ├── gt/
│       └── cameras.json
├── output/                    # 训练输出 (gitignore)
├── asset/                     # Blender 工程文件 (gitignore)
└── main.py                    # CLI 入口
```

## 环境搭建

```bash
conda env create -f environment.yml
conda activate differentiable

# 安装 nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
```

## 数据准备

```bash
blender --background asset/scene.blend --python scripts/blender_export.py
```

详见 [scripts/README.md](scripts/README.md)。

## 训练

```bash
# SH 着色 (默认)
python main.py --config configs/default.yaml --mode train

# PBR split-sum
python main.py --config configs/train_pbr.yaml --mode train

# 断点续训
python main.py --config configs/train_pbr.yaml --mode train --resume output/{dataset}/checkpoint.pt
```

## 导出 & 视频

```bash
# 导出贴图
python main.py --mode export --checkpoint output/{dataset}/checkpoint.pt

# 环绕视频
python main.py --mode video --checkpoint output/{dataset}/checkpoint.pt
```

## 测试

```bash
pytest tests/ -v
```

## 技术栈

- Python 3.10, PyTorch 2.x, nvdiffrast
- Blender 5.1 (数据制备)
- trimesh, OpenCV, Pillow, matplotlib

## License

Private
