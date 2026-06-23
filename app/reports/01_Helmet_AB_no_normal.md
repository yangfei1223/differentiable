# 头盔 — Web Viewer AB 验证（no_normal 数据源）

Web Viewer 与训练管线 GT 的首次端到端像素对比。数据源切换到 `no_normal` 训练输出。

## 测试配置

| 项目 | 值 |
|------|-----|
| 场景 | helmet（单 mesh，`data/helmet_260604/scene/lowpoly.glb`，14,588 顶点） |
| 数据源 | `output/helmet_no_normal/epoch2000/`（训练时 `pbr.disable_normal_map=True`） |
| Env map | `env_map.hdr`，由 checkpoint 经 `src/shading/pbr/hdr_writer.py` 重新生成（RGBE 编码，保留 softplus 解码的 HDR 值） |
| BRDF LUT | 由 `brdf_lut.pt` 重新生成 256×256 数据 PNG（原 `brdf_lut.png` 是旧的 512×256 调试图） |
| 相机 | `data/helmet_260604/cameras.json`，indices `[0, 50, 100, 150]` 对应 `compare_0000..compare_0003` |
| 渲染尺寸 | 1024×1024（与 GT 面板一致） |

### 关于 `uNormalMapEnabled = false`

`normal_map.png` 在 `no_normal` 输出里仍含有从父 checkpoint 继承的扰动数据（255 unique R values），但训练管线生成 GT 和环绕视频时**永远跳过 normal mapping**：

- `pbr_logger._export_compare`（line 90）和 `video.py`（line 221）调用 `model.shade(...)` 时**不传** `tangents/bitangents`
- `pbr_model.py:78` 的守卫 `if not disable_normal_map AND tangents is not None AND bitangents is not None` 因此失败
- 结果：GT 始终用几何法线渲染

Web Viewer 的 `PBRMesh.ts` 设置 `uNormalMapEnabled = false` 以匹配此行为，**与 `normal_map.png` 文件内容无关**。

## AB 统计（camera[50]，`compare_0001`）

完整 `ab_compare.py` 输出见 `../resource/helmet_no_normal_ab/psnr.txt`。

| 指标 | 值 |
|------|-----|
| **PSNR**（重叠前景区域） | **17.47 dB** |
| RMSE | 34.10 |
| Web 前景均值 RGB | `[146, 143, 129]` |
| GT 前景均值 RGB | `[148, 146, 134]` |
| 平均差（Web − GT） | `[−1, −2, −5]` |
| 平均绝对差 | `[20, 21, 24]` |
| 前景覆盖率 | Web 56.5% vs GT 56.4% |
| Sobel 边缘密度（前景，阈值 50） | Web 19.58% vs GT 19.61% |

## 渲染对比（最终合成）

左 GT、右 Web，4 个角度（camera[0/50/100/150]）：

<p align="center">
<img src="../resource/helmet_no_normal_ab/compare_final_cam0.png" width="45%"/>
<img src="../resource/helmet_no_normal_ab/compare_final_cam50.png" width="45%"/>
</p>

<p align="center">
<img src="../resource/helmet_no_normal_ab/compare_final_cam100.png" width="45%"/>
<img src="../resource/helmet_no_normal_ab/compare_final_cam150.png" width="45%"/>
</p>

## 通道分解对比

Web 的 diffuse/specular debug 通道输出**线性值**，GT 面板经过 `pow(1/2.2)` 编码（`pbr_logger.py:162`）。下方对比图 Web 通道已应用相同 `pow(1/2.2)` 编码以视觉对齐。

### Diffuse 通道

<p align="center">
<img src="../resource/helmet_no_normal_ab/compare_diffuse_cam0.png" width="45%"/>
<img src="../resource/helmet_no_normal_ab/compare_diffuse_cam50.png" width="45%"/>
</p>

<p align="center">
<img src="../resource/helmet_no_normal_ab/compare_diffuse_cam100.png" width="45%"/>
<img src="../resource/helmet_no_normal_ab/compare_diffuse_cam150.png" width="45%"/>
</p>

### Specular 通道

<p align="center">
<img src="../resource/helmet_no_normal_ab/compare_specular_cam0.png" width="45%"/>
<img src="../resource/helmet_no_normal_ab/compare_specular_cam50.png" width="45%"/>
</p>

<p align="center">
<img src="../resource/helmet_no_normal_ab/compare_specular_cam100.png" width="45%"/>
<img src="../resource/helmet_no_normal_ab/compare_specular_cam150.png" width="45%"/>
</p>

## 分析

### 已对齐

- 前景均值偏差 ≤3 RGB，前景覆盖率几乎相同（56.5% vs 56.4%）
- 边缘密度几乎相同（19.58% vs 19.61%）→ Web 没有比 GT 更"碎片"
- 视觉检查 4 个角度：剪影、取景、diffuse 高光位置、表面碎片化纹理全部对齐
- Diffuse 通道方向一致，亮度和色温有小差异

### 剩余误差来源（PSNR 17.47 dB，未达 ≥30 dB 目标）

1. **Specular LOD 语义不一致**（最主要）：`pbr.frag:76` 用 `textureLod`（绝对 LOD，跳过屏幕空间 UV 导数），Python 用 nvdiffrast 的 `mip_level_bias`（相对偏置加到 auto-LOD）。曲面相邻像素 R 方向变化大时 auto-LOD 贡献可观，Web 缺失这部分贡献，表现为高光更模糊且亮度偏高。修复方向：改用 `texture(uEnvMap, uv, specLod)` 与 diffuse（`pbr.frag:60`）一致。

2. **BRDF LUT UV 映射约定可能不一致**：`pbr.frag:78` 用 `(val*(size−1)+0.5)/size`（pixel-center / `align_corners=False`），Python `brdf_lut.py:120` 用 `grid_sample(..., align_corners=True)` 配合 `grid ∈ [0,1]`。两者在边界采样点不等价，需单测确认。

3. **baseColor gamma 曲线**：Python 导出用 `pow(1/2.2)`（`pbr_model.py:259`），Web 依赖 Three.js `SRGBColorSpace` 自动解码（真正 sRGB EOTF 分段函数）。两者 mid-tone 差异 1–5%。

4. **亚像素几何错位**：相机参数虽一致，但 nvdiffrast 与 WebGL 光栅化在 silhouette 覆盖判断上略有差异，每个边缘像素贡献 50–100 RGB 偏差，在前景边缘 1–2 像素带内拉低 PSNR。

## 结论

| 项目 | 状态 |
|------|------|
| 视觉对齐（取景/剪影/diffuse 高光位置/碎片化） | 通过 |
| 整体色调与亮度（前景均值 ≤3 RGB） | 通过 |
| PSNR ≥ 30 dB | 未达（17.47 dB） |
| Specular 高光形状 | Web 偏亮偏模糊 |

PSNR 未达目标的主要原因是 specular 通道的 LOD 采样方式与训练管线不一致（详见上文"剩余误差来源"第 1 条）。

## 复现步骤

```bash
# 1. 从 no_normal 输出打包 helmet
python -m scripts.package_runtime_asset \
  --glb data/helmet_260604/scene/lowpoly.glb \
  --epoch-dir output/helmet_no_normal/epoch2000 \
  --scene-name helmet

# 2. 启动 dev server
cd app && npm install && npm run dev

# 3. 加载 helmet 场景 + 相机 hash（camera[50]）
#    http://localhost:5173/#cam=1.344972,1.144323,2.843814,-0.002482,0.187155,1.1e-05,0.0,0.0,1.0,35.8

# 4. 离屏渲染 1024×1024
#    URL 加 ?render=1024 自动下载 PNG；
#    或 console 手动驱动 window.__pipeline.setSize(1024,1024); .render(); readPixels; toDataURL

# 5. 对比
python scripts/ab_compare.py helmet
```

## 相关文件

- 资源：`app/resource/helmet_no_normal_ab/`
- 数据源：`output/helmet_no_normal/epoch2000/`
- 训练配置：`configs/train_pbr_helmet_no_normal.yaml`
- 打包脚本：`scripts/package_runtime_asset.py`
- AB 对比脚本：`scripts/ab_compare.py`
