# 01 — SH (Spherical Harmonics) Rendering Scheme

基于球谐系数的可微渲染方案。每个纹理 texel 编码 SH 系数，渲染时根据视角方向解码颜色，隐式表达视角依赖效果（如高光）。

## 技术参数

| 参数 | 值 |
|------|-----|
| SH Order | 2（9 系数 × 3 通道 = 27 per texel） |
| 纹理编码 | 原始 SH 系数 |
| 纹理分辨率 | 512 → 1024 → 2048 → 4096 |
| 训练轮数 | 2000 |
| Batch Size | 4 |
| 学习率 | 0.01（DC），0.0005（高阶） |
| 损失函数 | L1 + SSIM + TV |

## 结果总览

| 场景 | PSNR | 分析 |
|------|------|------|
| 头盔 | **13.19 dB** | SH 无法捕捉金属面罩的锐利镜面高光 |
| 钢琴 | **20.37 dB** | 钢琴以漫反射为主，SH 近似尚可 |

头盔的金属面罩需要高动态范围的镜面反射，2 阶 SH 只有 9 个基函数，无法逼近尖锐高光。钢琴暗色漫反射表面容易用低阶 SH 表达。

---

## 头盔 SH 结果

渲染对比（左上 GT，右上渲染，左下 Diffuse，右下 SH 高频分量）：

<p align="center">
<img src="../../resource/helmet_sh/compare_0000.png" width="45%"/>
<img src="../../resource/helmet_sh/compare_0001.png" width="45%"/>
</p>

训练曲线：

<p align="center">
<img src="../../resource/helmet_sh/curves.png" width="60%"/>
</p>

环绕视频：

<p align="center">
<video src="../../resource/helmet_sh/orbit.mp4" width="30%"/>
<video src="../../resource/helmet_sh/orbit_dc.mp4" width="30%"/>
<video src="../../resource/helmet_sh/orbit_hf.mp4" width="30%"/>
</p>

---

## 钢琴 SH 结果

<p align="center">
<img src="../../resource/piano_sh/compare_0000.png" width="45%"/>
<img src="../../resource/piano_sh/compare_0001.png" width="45%"/>
</p>

<p align="center">
<img src="../../resource/piano_sh/curves.png" width="60%"/>
</p>

<p align="center">
<video src="../../resource/piano_sh/orbit.mp4" width="30%"/>
<video src="../../resource/piano_sh/orbit_dc.mp4" width="30%"/>
<video src="../../resource/piano_sh/orbit_hf.mp4" width="30%"/>
</p>

## 局限

1. **无法捕捉尖锐高光**——2 阶 SH 仅 9 个基函数，对光滑/反射表面严重不足
2. **无材质分离**——颜色 bake 为 SH 系数，无显式粗糙度/金属度控制
3. **无环境光照**——SH 纹理隐式编码所有环境相关效果
4. **金属表面表现差**——头盔 13.19 dB vs 钢琴 20.37 dB 说明对 specular-heavy 场景严重不足

## 相关文件

- 输出：`output/helmet_260604/epoch2000/`，`output/piano_260604/epoch2000/`
- 资源：`resource/helmet_sh/`，`resource/piano_sh/`
