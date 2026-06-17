太棒了，使用 Mermaid 流程图来替代实体图片，不仅完美符合“纯文本无图片”的要求，而且在 GitHub、GitLab 或各类 Markdown 编辑器中都能直接渲染，极具工程师极客感。

这是为你更新的**包含了 Mermaid 文本架构图**的完整技术方案文档。

---

# 神经光照贴图 (Neural Lightmap) POC 技术方案文档

## 1. 需求背景与核心目标 (Background & Objectives)

### 1.1 传统图形学痛点

在现代实时渲染管线中，光照质量与性能始终是一对矛盾体：

* **传统光照贴图 (Lightmap)：** 仅能烘焙与视角无关的漫反射（Diffuse），无法表现高光、镜面反射等复杂材质质感。
* **实时光线追踪 / SSR：** 能够渲染高频的视角相关（View-dependent）光照，但在移动端、Web 端或低算力平台上无法维持 60 FPS 的流畅体验。

### 1.2 技术演进与架构思路

近年来，以 MobileNeRF 为代表的“表面参数化”神经渲染路线证明了：**将神经辐射场降维，绑定到传统 Mesh 的 UV 表面上，能够完美复用 GPU 高度优化的硬件光栅化管线。**
本方案（Neural Lightmap）旨在摒弃传统 NeRF 昂贵的体积光线投射（Ray Marching），利用神经网络的强拟合能力，在离线阶段烘焙包含全局光照（GI）与高光反射的完整渲染积分，并在运行时通过极轻量级的 MLP（微型神经网络）在片元着色器中实时解码。

### 1.3 POC 第一阶段核心目标

* **数学连通性验证：** 在纯 PyTorch 环境下，跑通“可微光栅化 $\rightarrow$ 特征采样 $\rightarrow$ MLP 解码 $\rightarrow$ 误差反向传播”的完整闭环。
* **高光拟合验证：** 验证网络能够在 Linear HDR 空间下，通过合理的视角频域编码，学习并重现高频的高光信息。
* **工程去冗余：** 剥离不必要的编译器桥接（暂时延后 SlangPy），以极简的单 Mesh 场景（Hello World）跑通数据管线。

---

## 2. 系统架构设计 (Architecture Design)

本系统在宏观上分为**离线烘焙训练**与**实时渲染解码**两条主线。以下为全链路数据流与状态管理图：

```mermaid
graph TD
    %% 定义样式
    classDef offline fill:#e8f4f8,stroke:#5a9bd4,stroke-width:2px;
    classDef runtime fill:#fcf3e3,stroke:#e6a23c,stroke-width:2px;
    classDef data fill:#ffffff,stroke:#888888,stroke-dasharray: 5 5;

    subgraph Offline [离线烘焙与训练管线 (PyTorch)]
        A[场景几何 Mesh + UVs] --> C[nvdiffrast 可微光栅化]
        B[多视角 GT 图像 EXR] --> D[Loss 计算]
        C -->|G-Buffer: UV & Mask| E(采样特征图 Feature Map)
        E -->|特征向量| F{Tiny MLP 解码}
        F -->|预测 RGB| D
        D -->|MSE/L1 + TV Loss| E
        D -->|反向传播| F
    end

    subgraph Runtime [实时渲染管线 (Engine / Shader)]
        G[硬件光栅化 Rasterizer] --> H[UV 坐标 & 世界坐标]
        H --> I(采样已烘焙特征纹理)
        J[当前相机视角] --> K[计算归一化视线方向]
        I --> L[向量拼接 Concat]
        K -->|高频位置编码| L
        L --> M{已加载权重的 Tiny MLP}
        M -->|Softplus 激活| N[输出最终物理辐射度]
    end
    
    %% 资产导出路径
    E -. 导出量化纹理 .-> I
    F -. 导出网络权重 .-> M

    class Offline offline;
    class Runtime runtime;
    class A,B,G,J data;

```

### 2.1 系统层级划分

1. **数据生成层 (Data Generation)：** 使用 Blender 或 PBR 渲染器，生成包含相机位姿（MVP）、光源信息的线性高动态范围（Linear OpenEXR）多视角参考图像。
2. **几何光栅化层 (Rasterization)：** 基于 `nvdiffrast` 执行硬件级可微光栅化，提取屏幕空间逐像素的 UV 坐标、世界坐标与有效像素掩码（Mask）。
3. **神经特征场 (Neural Feature Field)：** 优化一张 $H \times W \times C$（如 12 维）的可学习张量纹理，隐式编码表面的反照率、法线、环境遮蔽及场景光照。
4. **动态解码层 (Dynamic Decoding)：** 将采样的特征向量与频域编码后的视线方向结合，输入至 `TinyMLP`，解码出最终的物理辐射度（RGB）。

---

## 3. 图形算法与逻辑推演 (Algorithm & Logic)

### 3.1 渲染方程降维表达

渲染方程定义的出射辐射度 $L_o$ 在本架构中被建模为：


$$L_o(p, \omega_o) = \mathcal{M}_{\theta}\Big( \mathcal{T}\big(\Phi(p)\big), \gamma(\omega_o) \Big)$$

* $\Phi(p) = (u, v)$：表面点的 UV 参数化坐标。
* $\mathcal{T}$：待优化的神经特征图张量（12 维）。
* $\gamma(\omega_o)$：视线方向的高频位置编码。
* $\mathcal{M}_{\theta}$：权重为 $\theta$ 的 Tiny MLP 网络。

### 3.2 视角频域位置编码 (Positional Encoding)

为了避免网络产生仅能输出漫反射的“谱偏置（Spectral Bias）”，必须对视角向量进行频域展开：


$$\gamma(\mathbf{d}) = \Big( \mathbf{d}, \sin(2^0 \pi \mathbf{d}), \cos(2^0 \pi \mathbf{d}), \dots, \sin(2^{L-1} \pi \mathbf{d}), \cos(2^{L-1} \pi \mathbf{d}) \Big)$$


**架构决策：** 为平衡 12 维特征图的“话语权”，避免网络过度依赖视角而忽略几何纹理，最高频阶数锁定为 **$L=2$**（总计 15 维）或 **$L=3$**（总计 21 维）。

### 3.3 复合损失函数 (Loss Function)

严禁单纯使用 MSE。采用 **L1 主损失** 保证高光锐利度，辅以 **总变差 (Total Variation) 正则化** 保证 UV 空间的连续性，防止特征图出现高频噪点与边缘渗色。


$$\mathcal{L}_{total} = \frac{1}{|M|}\sum_{i \in M} \Big\| I_{\text{pred}}^{(i)} - I_{\text{gt}}^{(i)} \Big\|_1 + \lambda_{tv} \mathcal{L}_{tv}(\mathcal{T})$$


*(注：$M$ 为前景像素掩码集合，背景像素严格不参与计算)*

---

## 4. 核心模块与代码实现规范 (Implementation Specs)

### 4.1 网络拓扑与 HDR 激活

为了支持物理世界中无上限的辐射度拟合，输出层必须摒弃 `Sigmoid`，改用 `Softplus` 以保证梯度在极暗区域的活性，并允许输出 $>1.0$ 的 HDR 高光。

```python
import torch
import torch.nn as nn

class TinyMLP(nn.Module):
    def __init__(self, in_dim=27, hidden_dim=32, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.Softplus() # 核心：允许物理辐射度 (HDR) 输出
        )

```

### 4.2 训练掩码与索引提取 (Indexing & Masking)

在执行网络前向传播时，**必须利用掩码展平数据**。这可节省最高 80% 的无效背景算力（FLOPs），避免显存溢出，并保护特征图对应背景 UV 区域不被错误梯度污染。

```python
# 光栅化提取 mask
mask = (rast[0, ..., 3] > 0).detach()

# 索引展平有效像素 (Indexing)
valid_features = features[mask]       # Shape: [N, 12]
valid_viewdir = view_dirs[mask]       # Shape: [N, 3]
valid_gt = gt_linear[mask]            # Shape: [N, 3]

# 拼接与前向推断
mlp_input = torch.cat([valid_features, valid_viewdir], dim=-1)
valid_pred = model.decoder(mlp_input)

# 计算纯净的有效像素 Loss
loss = F.l1_loss(valid_pred, valid_gt)

```

---

## 5. 数据源生成规范 (Data Generation Specs)

对于使用 Blender 导出的 GT 数据，必须严格遵循以下规范：

* **色彩空间：** 禁止导出带 Gamma 矫正或 Tone Mapping 的 sRGB PNG。必须将 `View Transform` 设为 `Raw`，导出为 `OpenEXR` 格式，确保网络在纯线性（Linear）空间下学习光照积分。
* **相机采样策略：** 采用上半球斐波那契螺旋分布（Fibonacci Sphere Lattice），辅以 $\pm 10\%$ 的距离扰动。
* **数据规模（第一阶段）：** 单一模型（如表面光滑的球体/雕塑）配合单一点光源，使用 **50 - 100** 张多视角图像即可满足基础架构的连通性与高光验证。

---

## 6. 核心排雷指南 (Fallbacks & Constraints)

1. **坐标系坍塌：** 确保渲染引擎（如 Blender）与 `nvdiffrast` 的相机坐标系方向完全一致（如区分左/右手系及 UP 轴）。视线向量 $\mathbf{d}$ 输入前必须进行归一化处理。
2. **学习率失衡：** 离散的特征图 $\mathcal{T}$ 只有在光栅化覆盖时才有局部梯度，其更新速度远慢于全局更新的 MLP。必须采用双时间尺度（TTUR），给特征图设置极大的学习率（如 `1e-1`），MLP 维持常规学习率（如 `1e-3`）。
3. **正则化抹杀高光：** 训练初期若高光丢失，需先将 TV Loss 权重 $\lambda_{tv}$ 降至 `1e-5` 或暂时关闭。确认网络容量与坐标系无误、高光出现后，再回调权重以抚平 UV 边缘接缝。