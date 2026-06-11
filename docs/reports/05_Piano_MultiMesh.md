# 05 — Piano Scene (Multi-Mesh PBR)

钢琴场景，使用 **多 mesh PBR** 扩展。6 个几何子 mesh（琴身面板、琴键、琴弦、踏板等）各自拥有独立 1024×1024 的 8 通道材质纹理。使用原始高模几何（`original_with_mats.glb`，~99K 顶点）与 GT 完全对齐。

## 子 Mesh 结构

| 子 Mesh | 面数 | 描述 |
|---------|------|------|
| Object_0 | 62 | 小型组件 |
| Object_1 | 1,428 | 中型组件 |
| Object_2 | 1,800 | 中型组件 |
| Object_3 | 3,800 | 大型组件 |
| Object_4 | 44,351 | 琴身主体 |
| Object_5 | 47,831 | 琴身/琴盖 |
| **总计** | **~99K 顶点** | 高模原始几何 |

## 结果

| 指标 | 值 |
|------|-----|
| 峰值 PSNR | **21.95 dB**（epoch 1800） |
| 最终 PSNR | 20.83 dB（epoch 2000） |
| 纹理分辨率 | 1024×1024 × 6 |
| 训练时间 | ~28 分钟（2000 epochs） |
| 输出 | `output/piano_260604_pbr_multi/` |

## 训练过程

| Epoch | PSNR | Resolution |
|-------|------|------------|
| 1 | 12.88 dB | 512 |
| 200 | 19.42 dB | 512 |
| 600 | 20.21 dB | 1024 |
| 1000 | 20.66 dB | 1024 |
| 1400 | 20.90 dB | 1024 |
| **1800** | **21.95 dB** | 1024 |
| 2000 | 20.83 dB | 1024 |

## 渲染对比

<p align="center">
<img src="../resource/piano_pbr_multi/compare_0000.png" width="45%"/>
<img src="../resource/piano_pbr_multi/compare_0001.png" width="45%"/>
</p>

## 训练曲线

<p align="center">
<img src="../resource/piano_pbr_multi/curves.png" width="60%"/>
</p>

## 子 Mesh 材质贴图

### Object_4（琴身主体，最大子 mesh）

<p align="center">
<img src="../resource/piano_pbr_multi/Object_4/base_color.png" width="22%"/>
<img src="../resource/piano_pbr_multi/Object_4/roughness.png" width="22%"/>
<img src="../resource/piano_pbr_multi/Object_4/metallic.png" width="22%"/>
<img src="../resource/piano_pbr_multi/Object_4/normal_map.png" width="22%"/>
</p>

### Object_3（大型组件）

<p align="center">
<img src="../resource/piano_pbr_multi/Object_3/base_color.png" width="22%"/>
<img src="../resource/piano_pbr_multi/Object_3/roughness.png" width="22%"/>
<img src="../resource/piano_pbr_multi/Object_3/metallic.png" width="22%"/>
<img src="../resource/piano_pbr_multi/Object_3/normal_map.png" width="22%"/>
</p>

### Object_0（最小子 mesh，62 面）

<p align="center">
<img src="../resource/piano_pbr_multi/Object_0/base_color.png" width="22%"/>
<img src="../resource/piano_pbr_multi/Object_0/roughness.png" width="22%"/>
<img src="../resource/piano_pbr_multi/Object_0/metallic.png" width="22%"/>
<img src="../resource/piano_pbr_multi/Object_0/normal_map.png" width="22%"/>
</p>

## 环境贴图

<p align="center">
<img src="../resource/piano_pbr_multi/env_map.png" width="45%"/>
</p>

## 环绕视频

<p align="center">
<video src="../resource/piano_pbr_multi/orbit.mp4" width="50%"/>
</p>

## 渲染架构

```
glTF (original_with_mats.glb)
  └── gltf_loader / pygltflib
      └── 6 × SubMeshData → MultiMeshData

每帧渲染：
  for each submesh:
      renderer.rasterize_and_interpolate()
      → model.shade_submesh(name, ...)
      → rgb_sub, mask_sub (+ depth)

  Depth compositing:
      argmin(depth) → frontmost per pixel
      torch.gather → differentiable selection
      → 最终渲染图
```

### 关键优化

- **Depth-based 合成**（非加法）：防止半透明伪影
- **单次 `dr.texture()`**：从 12 次/epoch 降到 6 次
- **`torch.gather` 合成**：替换 6 层 `torch.where` 链，降低 autograd 开销

## 性能 Profile

| 分辨率 | 每 step | 每 epoch（4 views） | 2000 epochs |
|-------|---------|---------------------|-------------|
| 512 | 126 ms | 504 ms | 17 min |
| 1024 | 172 ms | 688 ms | **23 min** |
| 2048 | 7716 ms | 30.9 s | 17 hours |

2048 分辨率下 autograd 图构建开销非线性增长（6 submesh × gradient × 4M 像素）。当前配置 max resolution 限制在 1024。

## 待解决问题

1. **Specular 质量不足**——粗糙度初始值 0.5 对光洁钢琴表面收敛慢，高光不够锐利
2. **"水渍"伪影**——diffuse/specular 纹理中的斑点状噪声，小 submesh 梯度信号弱导致
3. **根节点 transform 兼容性**——gltf_loader 的场景图 transform 与 trimesh 坐标系不一致，头盔场景需要修复
4. **粗糙度初始化**——降低到 0.12 提升钢琴 PSNR（+0.9 dB），但对 diffuse 场景可能不稳定

## 与 Single-Mesh 对比

| 方面 | Single-Mesh | Multi-Mesh |
|------|------------|------------|
| PSNR | 21.41 dB @ 2048 | **21.95 dB** @ 1024 |
| 三角走样 | 可见 | **无** |
| 部件材质 | 平均化 | **独立** |
| 纹理数量 | 1 × 2048² | 6 × 1024² |
| 几何质量 | 低模 ~14K | 高模 ~99K |

## 相关文件

- 输出：`output/piano_260604_pbr_multi/epoch2000/`
- 资源：`resource/piano_pbr_multi/`
