"""训练主循环 — Coarse-to-Fine 分辨率调度与 seam padding。"""
from __future__ import annotations

import random
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from src.config import Config, ResolutionStep
from src.dataset import GTDataset
from src.losses import CombinedLoss
from src.mesh import load_mesh
from src.renderer import DifferentiableRenderer
from src.seam_padding import dilate_texture
from src.sh import init_sh_texture


class Trainer:
    """可微烘焙训练器。

    支持 Coarse-to-Fine 分辨率调度：根据 epoch 从低分辨率逐渐提升
    纹理与渲染分辨率，并在训练过程中周期性地执行 seam padding。

    Args:
        config: 全局配置对象。
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ---- 1. 加载网格 ----
        mesh = load_mesh(config.data.mesh_path)
        self.vertices, self.faces, self.uvs, self.uv_idx = mesh.to_torch()

        # ---- 2. 创建数据集 ----
        self.dataset = GTDataset(
            gt_dir=config.data.gt_dir,
            camera_path=config.data.camera_path,
        )

        # ---- 3. 初始化 SH 纹理 ----
        self.sh_order = config.texture.sh_order
        _sh = init_sh_texture(
            config.texture.base_resolution,
            sh_order=self.sh_order,
            init_dc=config.texture.init_dc_value,
        )
        # .to(device) 会破坏 leaf 状态, 需要重建 nn.Parameter
        self.sh_texture = nn.Parameter(_sh.data.to(self.device))

        # ---- 4. 优化器与调度器 ----
        self.optimizer = Adam([self.sh_texture], lr=config.training.lr)
        self.scheduler = MultiStepLR(
            self.optimizer,
            milestones=config.training.lr_decay_epochs,
            gamma=config.training.lr_decay,
        )

        # ---- 5. 组合损失 ----
        self.criterion = CombinedLoss(
            lambda_l1=config.loss.lambda_l1,
            lambda_ssim=config.loss.lambda_ssim,
            lambda_tv=config.loss.lambda_tv,
        )

        # ---- 6. 解析分辨率调度 ----
        self.resolution_schedule: List[ResolutionStep] = config.training.resolution_schedule
        # 按_epoch 排序
        self.resolution_schedule.sort(key=lambda s: s.epoch)

        # ---- 7. 当前分辨率 & 渲染器 ----
        self.current_resolution = self._current_resolution(0)
        self.renderer = self._create_renderer(self.current_resolution)

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------
    def _current_resolution(self, epoch: int) -> int:
        """根据 epoch 查找当前应该使用的分辨率。"""
        res = self.resolution_schedule[0].resolution if self.resolution_schedule else self.config.texture.base_resolution
        for step in self.resolution_schedule:
            if epoch >= step.epoch:
                res = step.resolution
            else:
                break
        return res

    def _create_renderer(self, resolution: int) -> DifferentiableRenderer:
        """创建指定分辨率的渲染器。"""
        return DifferentiableRenderer(
            vertices=self.vertices,
            faces=self.faces,
            uvs=self.uvs,
            uv_idx=self.uv_idx,
            resolution=resolution,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Texture manipulation
    # ------------------------------------------------------------------
    def _resize_sh_texture(self, new_res: int) -> None:
        """双线性插值将 SH 纹理缩放到 new_res，并重建优化器。"""
        old_res = self.sh_texture.shape[1]
        if old_res == new_res:
            return

        # [1, H, W, C] → [1, C, H, W] 用于 interpolate
        tex = self.sh_texture.data.permute(0, 3, 1, 2)  # [1, C, H, W]
        tex = F.interpolate(tex, size=(new_res, new_res), mode="bilinear", align_corners=False)
        tex = tex.permute(0, 2, 3, 1)  # [1, H, W, C]

        self.sh_texture = nn.Parameter(tex.contiguous())

        # 重建优化器和调度器
        lr = self.config.training.lr
        self.optimizer = Adam([self.sh_texture], lr=lr)
        # 保持当前 lr_decay 逻辑
        self.scheduler = MultiStepLR(
            self.optimizer,
            milestones=self.config.training.lr_decay_epochs,
            gamma=self.config.training.lr_decay,
        )

    def _apply_seam_padding(self) -> None:
        """执行 seam padding：膨胀纹理中的空白区域。"""
        radius = self.config.seam_padding.dilation_radius
        H, W = self.sh_texture.shape[1], self.sh_texture.shape[2]
        # 简单有效掩码：所有非零像素（或全部视为有效）
        # 对已训练的纹理，我们使用全 1 掩码做 dilation（无损）
        valid_mask = torch.ones(1, H, W, 1, device=self.device)
        padded = dilate_texture(self.sh_texture.data, valid_mask, radius=radius)
        self.sh_texture = nn.Parameter(padded.contiguous())
        # 重建优化器以保持梯度跟踪
        self.optimizer = Adam([self.sh_texture], lr=self.optimizer.param_groups[0]["lr"])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        output_dir: str = "output",
        checkpoint_every: int = 200,
        resume_from: str | None = None,
    ) -> None:
        """主训练循环。

        Args:
            output_dir: checkpoint 保存目录。
            checkpoint_every: 每 N 个 epoch 保存一次 checkpoint。
            resume_from: 断点续训的 checkpoint 路径 (.pt)。
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        start_epoch = 0

        # ---- 断点续训 ----
        if resume_from is not None:
            ckpt = torch.load(resume_from, map_location=self.device)
            if isinstance(ckpt, dict) and "sh_texture" in ckpt:
                state = ckpt
                start_epoch = state.get("epoch", 0)
                tex = state["sh_texture"].to(self.device)
                self.sh_texture = nn.Parameter(tex)
                self.optimizer = Adam([self.sh_texture], lr=self.config.training.lr)
                self.scheduler = MultiStepLR(
                    self.optimizer,
                    milestones=self.config.training.lr_decay_epochs,
                    gamma=self.config.training.lr_decay,
                )
                # 跳过已完成的 scheduler 步数
                for _ in range(start_epoch):
                    self.scheduler.step()
                print(f"[Resume] 从 epoch {start_epoch} 继续, loss={state.get('loss', 'N/A')}")
            else:
                # 旧格式：仅纹理张量
                self.sh_texture = nn.Parameter(ckpt.to(self.device))
                print("[Resume] 加载纹理 (旧格式, epoch 未知)")

        num_epochs = self.config.training.num_epochs
        batch_size = self.config.training.batch_size
        seam_every = self.config.seam_padding.apply_every_n_epochs
        num_views = len(self.dataset)

        # 确保渲染器与当前纹理分辨率匹配
        tex_res = self.sh_texture.shape[1]
        self.current_resolution = tex_res
        self.renderer = self._create_renderer(tex_res)

        for epoch in range(start_epoch, num_epochs):
            # ---- 检查分辨率调度 ----
            target_res = self._current_resolution(epoch)
            if target_res != self.current_resolution:
                self._resize_sh_texture(target_res)
                self.renderer = self._create_renderer(target_res)
                self.current_resolution = target_res

            # ---- 随机采样 batch ----
            indices = random.sample(range(num_views), min(batch_size, num_views))

            epoch_loss = 0.0
            for idx in indices:
                self.optimizer.zero_grad()

                img_np, camera = self.dataset[idx]
                gt = torch.from_numpy(img_np).unsqueeze(0).to(self.device)  # [1, 3, H_gt, W_gt]

                # 渲染
                rendered, mask = self.renderer.render(self.sh_texture, camera)  # [1, H, W, 3], [1, H, W]

                # 将 GT resize 到渲染分辨率
                gt_hw = gt.permute(0, 1, 2, 3)  # [1, 3, H_gt, W_gt]
                H, W = rendered.shape[1], rendered.shape[2]
                gt_resized = F.interpolate(gt_hw, size=(H, W), mode="bilinear", align_corners=False)
                gt_resized = gt_resized.squeeze(0).permute(1, 2, 0).unsqueeze(0)  # [1, H, W, 3]

                # sRGB → linear: GT 图像是 sRGB 编码, 渲染输出是线性空间
                gt_linear = gt_resized.clamp(0, 1).pow(2.2)

                # 计算损失
                loss = self.criterion(rendered, gt_linear, mask, self.sh_texture)

                # 反向传播 & 更新
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            # 调度器步进
            self.scheduler.step()

            # 周期性 seam padding
            if seam_every > 0 and (epoch + 1) % seam_every == 0:
                self._apply_seam_padding()

            avg_loss = epoch_loss / len(indices) if indices else 0.0
            if (epoch + 1) % max(1, num_epochs // 10) == 0 or epoch == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}] loss={avg_loss:.6f} res={self.current_resolution}")

            # ---- 周期性 checkpoint ----
            if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
                ckpt_path = os.path.join(output_dir, f"sh_texture_epoch{epoch+1}.pt")
                torch.save({
                    "epoch": epoch + 1,
                    "sh_texture": self.get_sh_texture(),
                    "loss": avg_loss,
                    "resolution": self.current_resolution,
                }, ckpt_path)
                print(f"  [Checkpoint] {ckpt_path}")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_sh_texture(self) -> torch.Tensor:
        """返回 SH 纹理的 CPU 张量（detached）。"""
        return self.sh_texture.data.detach().cpu()
