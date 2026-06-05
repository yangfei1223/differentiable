"""训练主循环 — Coarse-to-Fine 分辨率调度与 seam padding。"""
from __future__ import annotations

import os
import random
from typing import List

import numpy as np

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
from src.sh import init_sh_texture, cat_sh_features
from src.utils import vis, vis_pair

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

        # ---- 3. 初始化 SH 纹理（3DGS 风格：DC 和高阶分开） ----
        self.sh_order = config.texture.sh_order
        _sh_dc, _sh_rest = init_sh_texture(
            config.texture.base_resolution,
            sh_order=self.sh_order,
            init_dc=config.texture.init_dc_value,
        )
        # .to(device) 会破坏 leaf 状态, 需要重建 nn.Parameter
        self.features_dc = nn.Parameter(_sh_dc.data.to(self.device))
        self.features_rest = nn.Parameter(_sh_rest.data.to(self.device))

        # ---- 4. 优化器与调度器（3DGS: 高阶 lr = DC lr / 20） ----
        base_lr = config.training.lr
        self.optimizer = Adam([
            {"params": [self.features_dc], "lr": base_lr, "name": "f_dc"},
            {"params": [self.features_rest], "lr": base_lr * self.config.training.rest_lr_ratio, "name": "f_rest"},
        ])
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

        # ---- 8. 训练历史记录 ----
        self.history: dict[str, list] = {"epoch": [], "loss": [], "psnr": []}

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
        old_res = self.features_dc.shape[1]
        if old_res == new_res:
            return

        # Resize DC
        tex = self.features_dc.data.permute(0, 3, 1, 2)  # [1, C, H, W]
        tex = F.interpolate(tex, size=(new_res, new_res), mode="bilinear", align_corners=False)
        tex = tex.permute(0, 2, 3, 1)  # [1, H, W, C]
        self.features_dc = nn.Parameter(tex.contiguous())

        # Resize Rest (可能为空通道，跳过)
        if self.features_rest.shape[-1] > 0:
            tex_r = self.features_rest.data.permute(0, 3, 1, 2)
            tex_r = F.interpolate(tex_r, size=(new_res, new_res), mode="bilinear", align_corners=False)
            tex_r = tex_r.permute(0, 2, 3, 1)
            self.features_rest = nn.Parameter(tex_r.contiguous())
        else:
            # 仅更新 spatial size，保持 0 通道
            self.features_rest = nn.Parameter(
                torch.zeros(1, new_res, new_res, 0, device=self.device)
            )

        # 重建优化器和调度器
        base_lr = self.config.training.lr
        self.optimizer = Adam([
            {"params": [self.features_dc], "lr": base_lr, "name": "f_dc"},
            {"params": [self.features_rest], "lr": base_lr * self.config.training.rest_lr_ratio, "name": "f_rest"},
        ])

    def _apply_seam_padding(self) -> None:
        """执行 seam padding：膨胀纹理中的空白区域。"""
        radius = self.config.seam_padding.dilation_radius
        H, W = self.features_dc.shape[1], self.features_dc.shape[2]
        valid_mask = torch.ones(1, H, W, 1, device=self.device)

        self.features_dc = nn.Parameter(
            dilate_texture(self.features_dc.data, valid_mask, radius=radius).contiguous()
        )
        # features_rest 可能为空（SH order 0 时通道数为 0），跳过
        if self.features_rest.shape[-1] > 0:
            self.features_rest = nn.Parameter(
                dilate_texture(self.features_rest.data, valid_mask, radius=radius).contiguous()
            )

        # 重建优化器以保持梯度跟踪
        base_lr = self.config.training.lr
        self.optimizer = Adam([
            {"params": [self.features_dc], "lr": base_lr, "name": "f_dc"},
            {"params": [self.features_rest], "lr": base_lr * self.config.training.rest_lr_ratio, "name": "f_rest"},
        ])

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
        os.makedirs(output_dir, exist_ok=True)

        start_epoch = 0

        # ---- 断点续训 ----
        if resume_from is not None:
            ckpt = torch.load(resume_from, map_location=self.device)
            if isinstance(ckpt, dict) and "features_dc" in ckpt:
                state = ckpt
                start_epoch = state.get("epoch", 0)
                self.features_dc = nn.Parameter(state["features_dc"].to(self.device))
                self.features_rest = nn.Parameter(state["features_rest"].to(self.device))
                base_lr = self.config.training.lr
                self.optimizer = Adam([
                    {"params": [self.features_dc], "lr": base_lr, "name": "f_dc"},
                    {"params": [self.features_rest], "lr": base_lr * self.config.training.rest_lr_ratio, "name": "f_rest"},
                ])
                self.scheduler = MultiStepLR(
                    self.optimizer,
                    milestones=self.config.training.lr_decay_epochs,
                    gamma=self.config.training.lr_decay,
                )
                for _ in range(start_epoch):
                    self.scheduler.step()
                print(f"[Resume] 从 epoch {start_epoch} 继续, loss={state.get('loss', 'N/A')}")
            elif isinstance(ckpt, dict) and "sh_texture" in ckpt:
                # 旧格式兼容：单张 sh_texture → 拆分为 DC + Rest
                state = ckpt
                start_epoch = state.get("epoch", 0)
                tex = state["sh_texture"].to(self.device)  # [1, H, W, C]
                self.features_dc = nn.Parameter(tex[..., :3].contiguous())
                self.features_rest = nn.Parameter(tex[..., 3:].contiguous())
                base_lr = self.config.training.lr
                self.optimizer = Adam([
                    {"params": [self.features_dc], "lr": base_lr, "name": "f_dc"},
                    {"params": [self.features_rest], "lr": base_lr * self.config.training.rest_lr_ratio, "name": "f_rest"},
                ])
                self.scheduler = MultiStepLR(
                    self.optimizer,
                    milestones=self.config.training.lr_decay_epochs,
                    gamma=self.config.training.lr_decay,
                )
                for _ in range(start_epoch):
                    self.scheduler.step()
                print(f"[Resume] 从旧格式恢复, epoch {start_epoch}")
            else:
                # 最旧格式：仅纹理张量
                tex = ckpt.to(self.device)
                self.features_dc = nn.Parameter(tex[..., :3].contiguous())
                self.features_rest = nn.Parameter(tex[..., 3:].contiguous())
                print("[Resume] 加载纹理 (最旧格式, epoch 未知)")

        num_epochs = self.config.training.num_epochs
        batch_size = self.config.training.batch_size
        seam_every = self.config.seam_padding.apply_every_n_epochs
        num_views = len(self.dataset)

        # 确保渲染器与当前纹理分辨率匹配
        tex_res = self.features_dc.shape[1]
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
                rendered, mask = self.renderer.render(
                    self.features_dc, self.features_rest, camera,
                )  # [1, H, W, 3], [1, H, W]

                # nvdiffrast 输出为 OpenGL 坐标 (原点左下)，垂直翻转到图像坐标 (原点左上)
                rendered = rendered.flip(1)
                mask = mask.flip(1)

                # 将 GT resize 到渲染分辨率
                gt_hw = gt.permute(0, 1, 2, 3)  # [1, 3, H_gt, W_gt]
                H, W = rendered.shape[1], rendered.shape[2]
                gt_resized = F.interpolate(gt_hw, size=(H, W), mode="bilinear", align_corners=False)
                gt_resized = gt_resized.squeeze(0).permute(1, 2, 0).unsqueeze(0)  # [1, H, W, 3]

                # sRGB → linear: GT 图像是 sRGB 编码, 渲染输出是线性空间
                gt_linear = gt_resized.clamp(0, 1).pow(2.2)

                # 计算损失（TV loss 需要拼接纹理）
                sh_tex_for_loss = cat_sh_features(self.features_dc, self.features_rest)
                loss = self.criterion(rendered, gt_linear, mask, sh_tex_for_loss)

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

            # ---- 计算该 epoch 的 PSNR (用第一个视角, no_grad) ----
            psnr_val = 0.0
            with torch.no_grad():
                _img, _cam = self.dataset[0]
                _gt = torch.from_numpy(_img).unsqueeze(0).to(self.device)
                _rendered, _mask = self.renderer.render(
                    self.features_dc, self.features_rest, _cam,
                )
                _rendered = _rendered.flip(1)
                _mask = _mask.flip(1)
                _gt_hw = _gt.permute(0, 1, 2, 3)
                H, W = _rendered.shape[1], _rendered.shape[2]
                _gt_r = F.interpolate(_gt_hw, size=(H, W), mode="bilinear", align_corners=False)
                _gt_r = _gt_r.squeeze(0).permute(1, 2, 0).unsqueeze(0)
                _gt_lin = _gt_r.clamp(0, 1).pow(2.2)
                _mask_f = _mask.unsqueeze(-1).float()
                n_valid = _mask.sum() * 3 + 1e-8
                mse = ((_rendered - _gt_lin) * _mask_f).pow(2).sum() / n_valid
                if mse > 0:
                    psnr_val = 10.0 * torch.log10(1.0 / mse).item()

            self.history["epoch"].append(epoch + 1)
            self.history["loss"].append(avg_loss)
            self.history["psnr"].append(psnr_val)

            if (epoch + 1) % max(1, num_epochs // 10) == 0 or epoch == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}] loss={avg_loss:.6f} psnr={psnr_val:.2f}dB res={self.current_resolution}")

            # ---- 周期性 checkpoint ----
            if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
                ep_tag = f"epoch{epoch+1}"
                ep_dir = os.path.join(output_dir, ep_tag)
                os.makedirs(ep_dir, exist_ok=True)

                ckpt_path = os.path.join(ep_dir, "sh_texture.pt")
                torch.save({
                    "epoch": epoch + 1,
                    "features_dc": self.features_dc.data.detach().cpu(),
                    "features_rest": self.features_rest.data.detach().cpu(),
                    "loss": avg_loss,
                    "resolution": self.current_resolution,
                }, ckpt_path)
                print(f"  [Checkpoint] {ckpt_path}")

                # ---- 调试输出: compare / diffuse / video ----
                try:
                    self._export_debug(ep_dir, epoch=epoch)
                except Exception as e:
                    print(f"  [Debug export warning] {e}")

    # ------------------------------------------------------------------
    # Debug exports
    # ------------------------------------------------------------------
    def _export_debug(self, output_dir: str, epoch: int) -> None:
        """在 checkpoint 时输出调试可视化。"""
        import cv2
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.exporter import export_diffuse_texture
        from src.video import render_video

        tex = self.get_sh_texture()

        # 0. Loss + PSNR 曲线
        epochs = self.history["epoch"]
        losses = self.history["loss"]
        psnrs = self.history["psnr"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs, losses, "b-", linewidth=1)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, psnrs, "r-", linewidth=1)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("PSNR (dB)")
        ax2.set_title("PSNR")
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f"Epoch {epoch+1}  |  Loss: {losses[-1]:.4f}  |  PSNR: {psnrs[-1]:.2f} dB", fontsize=12)
        fig.tight_layout()
        curve_path = os.path.join(output_dir, "curves.png")
        fig.savefig(curve_path, dpi=100)
        # 也保存一份到 output root，方便随时查看最新状态
        fig.savefig(os.path.join(os.path.dirname(output_dir), "curves.png"), dpi=100)
        plt.close(fig)

        # 1. Diffuse 贴图
        diffuse_path = os.path.join(output_dir, "diffuse.png")
        export_diffuse_texture(tex, diffuse_path, self.sh_order)

        # 2. GT vs Rendered 对比 (均匀采样 4 个方向: 前/右/后/左, 2x2 atlas)
        num_views = len(self.dataset)
        compare_count = min(4, num_views)
        compare_indices = [int(i * num_views / compare_count) for i in range(compare_count)]

        # 准备 DC-only 参数（高频置零）
        dc_only = self.features_dc.data
        rest_data = self.features_rest.data

        for ci, idx in enumerate(compare_indices):
            img_np, camera = self.dataset[idx]
            gt = torch.from_numpy(img_np).unsqueeze(0).to(self.device)

            # 渲染 2 个版本：Full 和 DC only
            with torch.no_grad():
                rgb_full, mask = self.renderer.render(self.features_dc, self.features_rest, camera)
                rgb_dc, _ = self.renderer.render(dc_only, rest_data * 0, camera)

            # 高频净贡献 = Full - DC（可能有负值，clamp 到 [0,1]）
            rgb_hf = (rgb_full - rgb_dc).clamp(0, 1)

            mask = mask.flip(1)
            mask_np = mask[0].cpu().numpy()

            def to_srgb_bgr(rgb_tensor):
                img = rgb_tensor[0].flip(0).clamp(0, 1).pow(1.0 / 2.2).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                bgr[mask_np < 0.5] = 0
                return bgr

            gt_bgr = cv2.cvtColor((img_np.transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

            panels = [
                (gt_bgr, "GT"),
                (to_srgb_bgr(rgb_full), "Full SH"),
                (to_srgb_bgr(rgb_dc), "DC Only"),
                (to_srgb_bgr(rgb_hf), "High Freq"),
            ]

            # 统一尺寸并加标签
            target_h = min(p[0].shape[0] for p in panels)
            resized = []
            for img, label in panels:
                h, w = img.shape[:2]
                r = cv2.resize(img, (w * target_h // h, target_h))
                cv2.putText(r, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                resized.append(r)

            # 2x2 atlas: top=GT|Full, bottom=DC|HF
            top = np.concatenate([resized[0], resized[1]], axis=1)
            bottom = np.concatenate([resized[2], resized[3]], axis=1)
            canvas = np.concatenate([top, bottom], axis=0)

            compare_path = os.path.join(output_dir, f"compare_{ci:04d}.png")
            cv2.imwrite(compare_path, canvas)

        # 3. 视频: Full SH + DC only + High-freq only
        from src.mesh import load_mesh
        mesh = load_mesh(self.config.data.mesh_path)
        cfg = self.config
        video_kwargs = dict(
            mesh=mesh,
            center=cfg.video.center,
            radius=cfg.video.radius,
            height=cfg.video.height,
            num_frames=cfg.video.num_frames,
            fov_deg=cfg.video.fov_deg,
            resolution=cfg.video.resolution,
            fps=cfg.video.fps,
        )

        # Full SH
        render_video(sh_texture=tex, output_path=os.path.join(output_dir, "orbit.mp4"), **video_kwargs)

        # DC only video
        dc_tex = torch.cat([self.features_dc.data.detach().cpu(),
                            torch.zeros_like(self.features_rest.data.detach().cpu())], dim=-1)
        render_video(sh_texture=dc_tex, output_path=os.path.join(output_dir, "orbit_dc.mp4"), **video_kwargs)

        # High-freq video: Full - DC 逐帧差分
        render_video(sh_texture=tex, output_path=os.path.join(output_dir, "orbit_hf.mp4"),
                     subtract_texture=dc_tex, **video_kwargs)

        print(f"  [Debug] diffuse + compare + video → {output_dir}")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_sh_texture(self) -> torch.Tensor:
        """返回拼接后的完整 SH 纹理的 CPU 张量（detached）。"""
        return cat_sh_features(self.features_dc, self.features_rest).detach().cpu()
