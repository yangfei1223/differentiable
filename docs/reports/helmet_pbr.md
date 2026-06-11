# Helmet — PBR Split-Sum Rendering

DamagedHelmet 头盔场景，使用 PBR split-sum 着色。金属面罩 + 衬垫，测试 specular/reflection 处理能力。

## 实验配置

| 参数 | 值 |
|------|-----|
| 着色模型 | PBR (GGX split-sum) |
| 网格 | `data/helmet_260604/scene/lowpoly.glb`（14,588 顶点） |
| 材质纹理 | 8 通道（base_color 3 + roughness 1 + metallic 1 + normal 3） |
| 环境贴图 | 256×512 HDR（softplus 编码） |
| 纹理分辨率 | 512 → 1024 → 2048 |
| 训练轮数 | 2000 |
| 输出 | `output/helmet_260604_pbr/` |

## 结果

| 指标 | 值 |
|------|-----|
| **PSNR** | **21.97 dB** |
| 对比 SH | **+8.78 dB** |

## 渲染对比

左上 GT，右上渲染，左下 Diffuse，右下 Specular。

<p align="center">
<img src="../../resource/helmet_pbr/compare_0000.png" width="45%"/>
<img src="../../resource/helmet_pbr/compare_0001.png" width="45%"/>
</p>

## 训练曲线

<p align="center">
<img src="../../resource/helmet_pbr/curves.png" width="60%"/>
</p>

## 材质分解

<p align="center">
<img src="../../resource/helmet_pbr/base_color.png" width="22%"/>
<img src="../../resource/helmet_pbr/roughness.png" width="22%"/>
<img src="../../resource/helmet_pbr/metallic.png" width="22%"/>
<img src="../../resource/helmet_pbr/normal_map.png" width="22%"/>
</p>

- **base_color**：学习了面罩划痕细节
- **roughness**：面罩低粗糙度、衬垫高粗糙度
- **metallic**：面罩高金属度、衬垫低
- **normal_map**：细微表面起伏

## 环境贴图 & BRDF LUT

<p align="center">
<img src="../../resource/helmet_pbr/env_map.png" width="28%"/>
<img src="../../resource/helmet_pbr/brdf_lut.png" width="28%"/>
</p>

## 环绕视频

<p align="center">
<video src="../../resource/helmet_pbr/orbit.mp4" width="30%"/>
<video src="../../resource/helmet_pbr/orbit_diffuse.mp4" width="30%"/>
<video src="../../resource/helmet_pbr/orbit_specular.mp4" width="30%"/>
</p>

## 训练过程

| Epoch | PSNR | Resolution |
|-------|------|------------|
| 1 | ~10 dB | 512 |
| 200 | ~18 dB | 512 |
| 400 | ~19 dB | 1024 |
| 800 | ~21 dB | 2048 |
| 2000 | **21.97 dB** | 2048 |

## 上限诊断

| 实验 | PSNR |
|------|------|
| Random init（baseline） | 21.97 dB |
| GT 材质 + EXR envmap 初始化 | 22.17 dB |
| 差距 | **+0.20 dB** |

使用 DamagedHelmet 原始 albedo/metallicRoughness/normal（2048×2048）+ 场景 EXR 初始化环境光，仅提升 +0.20 dB。说明 ~22 dB 是 split-sum 模型的上限。

## 分析

PBR 在头盔上大幅优于 SH（+8.8 dB）。GGX split-sum 通过 roughness 控制 specular lobe 宽度，正确建模了金属面罩的镜面高光。SH 的 9 个基函数无法逼近这种尖锐反射。

Tone mapping 实验（Filmic/ACES、Reinhard、log1p）全部降低 PSNR，原版 softplus+clamp 管线最优。

## 相关文件

- 资源：`resource/helmet_pbr/`
- 输出：`output/helmet_260604_pbr/epoch2000/`
