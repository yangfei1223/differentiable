"""训练主循环 — Coarse-to-Fine 分辨率调度与 seam padding。"""
from __future__ import annotations

import os
import random
from typing import List

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from src.config import Config, ResolutionStep
from src.dataset import GTDataset
from src.losses import CombinedLoss
from src.mesh import load_mesh
from src.renderer import DifferentiableRenderer
from src.seam_padding import dilate_texture
from src.utils import vis, vis_pair

class Trainer:
    """可微烘焙训练器。

    支持 Coarse-to-Fine 分辨率调度：根据 epoch 从低分辨率逐渐提升
    纹理与渲染分辨率，并在训练过程中周期性地执行 seam padding。

    Args:
        config: 全局配置对象。
    """

    def __init__(self, config: Config, shading_model=None) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ---- 1. 加载网格 ----
        mesh = load_mesh(config.data.mesh_path)
        self.vertices, self.faces, self.uvs, self.uv_idx, self.normals, self.normal_idx = mesh.to_torch()

        # ---- 2. 创建数据集 ----
        self.dataset = GTDataset(
            gt_dir=config.data.gt_dir,
            camera_path=config.data.camera_path,
        )

        # ---- 3. 着色模型 ----
        if shading_model is not None:
            self.model = shading_model
        else:
            from src.shading import create_shading_model
            self.model = create_shading_model(config.render_mode, config)
        self.model.init_textures(config.texture.base_resolution)

        # ---- 4. 优化器 ----
        self._rebuild_optimizer()
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
            normals=self.normals,
            normal_idx=self.normal_idx,
            resolution=resolution,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Texture manipulation
    # ------------------------------------------------------------------
    def _rebuild_optimizer(self) -> None:
        """根据 model.parameters() 重建优化器，保持特殊 lr 比例。"""
        base_lr = self.config.training.lr
        param_groups = []
        for i, p in enumerate(self.model.parameters()):
            if self.config.render_mode == "sh" and i == 1:
                param_groups.append({"params": [p], "lr": base_lr * self.config.training.rest_lr_ratio})
            elif self.config.render_mode == "pbr" and i == 1:
                param_groups.append({"params": [p], "lr": base_lr * self.config.pbr.env_lr_ratio})
            else:
                param_groups.append({"params": [p], "lr": base_lr})
        self.optimizer = Adam(param_groups)

    def _resize_textures(self, new_res: int) -> None:
        """双线性插值将材质纹理缩放到 new_res，并重建优化器。"""
        old_res = self.model.get_material_texture().shape[1]
        if old_res == new_res:
            return

        tex = self.model.get_material_texture().to(self.device)
        tex = tex.permute(0, 3, 1, 2)
        tex = F.interpolate(tex, size=(new_res, new_res), mode="bilinear", align_corners=False)
        tex = tex.permute(0, 2, 3, 1)
        self.model.set_material_texture(tex.contiguous())
        self._rebuild_optimizer()

    def _apply_seam_padding(self) -> None:
        """执行 seam padding：膨胀纹理中的空白区域。"""
        radius = self.config.seam_padding.dilation_radius
        tex = self.model.get_material_texture().to(self.device)
        H, W = tex.shape[1], tex.shape[2]
        valid_mask = torch.ones(1, H, W, 1, device=self.device)
        tex = dilate_texture(tex, valid_mask, radius=radius).contiguous()
        self.model.set_material_texture(tex)
        self._rebuild_optimizer()

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
            if isinstance(ckpt, dict):
                if "render_mode" in ckpt:
                    # New format: ShadingModel state_dict
                    self.model.load_state_dict(ckpt)
                    start_epoch = ckpt.get("epoch", 0)
                elif "features_dc" in ckpt:
                    # Old SH format
                    state = {
                        "render_mode": "sh",
                        "features_dc": ckpt["features_dc"],
                        "features_rest": ckpt["features_rest"],
                    }
                    self.model.load_state_dict(state)
                    start_epoch = ckpt.get("epoch", 0)
                elif "sh_texture" in ckpt:
                    # Oldest SH format
                    tex = ckpt["sh_texture"]
                    state = {
                        "render_mode": "sh",
                        "features_dc": tex[..., :3],
                        "features_rest": tex[..., 3:],
                    }
                    self.model.load_state_dict(state)
                    start_epoch = ckpt.get("epoch", 0)
                else:
                    start_epoch = ckpt.get("epoch", 0)
            else:
                # 最旧格式：仅纹理张量
                tex = ckpt.to(self.device)
                state = {
                    "render_mode": "sh",
                    "features_dc": tex[..., :3],
                    "features_rest": tex[..., 3:],
                }
                self.model.load_state_dict(state)
                print("[Resume] 加载纹理 (最旧格式, epoch 未知)")

            self._rebuild_optimizer()
            self.scheduler = MultiStepLR(
                self.optimizer,
                milestones=self.config.training.lr_decay_epochs,
                gamma=self.config.training.lr_decay,
            )
            for _ in range(start_epoch):
                self.scheduler.step()
            if start_epoch > 0:
                print(f"[Resume] 从 epoch {start_epoch} 继续")

        num_epochs = self.config.training.num_epochs
        batch_size = self.config.training.batch_size
        seam_every = self.config.seam_padding.apply_every_n_epochs
        num_views = len(self.dataset)

        # 确保渲染器与当前纹理分辨率匹配
        tex_res = self.model.get_material_texture().shape[1]
        self.current_resolution = tex_res
        self.renderer = self._create_renderer(tex_res)

        for epoch in range(start_epoch, num_epochs):
            # ---- 检查分辨率调度 ----
            target_res = self._current_resolution(epoch)
            if target_res != self.current_resolution:
                self._resize_textures(target_res)
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
                if self.config.render_mode == "sh":
                    # SH uses the old render path for backward compat
                    rendered, mask, _ = self.renderer.render(
                        self.model.features_dc, self.model.features_rest, camera,
                    )
                else:
                    rast, texc, wpos, interp_normals, vdirs = self.renderer.rasterize_and_interpolate(camera)
                    rendered, mask = self.model.shade(rast, texc, wpos, interp_normals, vdirs, camera, self.current_resolution)

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

                # 计算损失（TV loss 需要纹理）
                tex_for_loss = self.model.get_material_texture().to(self.device)
                loss = self.criterion(rendered, gt_linear, mask, tex_for_loss)

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
                if self.config.render_mode == "sh":
                    _rendered, _mask, _ = self.renderer.render(
                        self.model.features_dc, self.model.features_rest, _cam,
                    )
                else:
                    _rast, _texc, _wpos, _inorm, _vdir = self.renderer.rasterize_and_interpolate(_cam)
                    _rendered, _mask = self.model.shade(_rast, _texc, _wpos, _inorm, _vdir, _cam, self.current_resolution)
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
                ckpt = self.model.state_dict()
                ckpt["epoch"] = epoch + 1
                ckpt["loss"] = avg_loss
                ckpt["resolution"] = self.current_resolution
                torch.save(ckpt, ckpt_path)
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

        tex = self.model.get_material_texture()

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
        export_diffuse_texture(tex, diffuse_path, self.config.texture.sh_order)

        # 2. GT vs Rendered 对比 (均匀采样 4 个方向: 前/右/后/左, 2x2 atlas)
        num_views = len(self.dataset)
        compare_count = min(4, num_views)
        compare_indices = [int(i * num_views / compare_count) for i in range(compare_count)]

        # 准备 DC-only 参数（高频置零）
        if hasattr(self.model, 'features_dc'):
            dc_only = self.model.features_dc.data
            rest_data = self.model.features_rest.data
        else:
            dc_only = None
            rest_data = None

        for ci, idx in enumerate(compare_indices):
            img_np, camera = self.dataset[idx]
            gt = torch.from_numpy(img_np).unsqueeze(0).to(self.device)

            # 渲染 2 个版本：Full 和 DC only
            with torch.no_grad():
                if self.config.render_mode == "sh":
                    rgb_full, mask, _ = self.renderer.render(self.model.features_dc, self.model.features_rest, camera)
                    rgb_dc, _, _ = self.renderer.render(dc_only, rest_data * 0, camera)
                else:
                    rast, texc, wpos, inorm, vdir = self.renderer.rasterize_and_interpolate(camera)
                    rgb_full, mask = self.model.shade(rast, texc, wpos, inorm, vdir, camera, self.current_resolution)
                    rgb_dc = rgb_full  # PBR has no DC/rest split
                    mask = mask

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

        # Full SH / PBR
        render_video(sh_texture=tex, output_path=os.path.join(output_dir, "orbit.mp4"), **video_kwargs)

        # DC only video (SH only)
        if hasattr(self.model, 'features_dc'):
            dc_tex = torch.cat([self.model.features_dc.data.detach().cpu(),
                                torch.zeros_like(self.model.features_rest.data.detach().cpu())], dim=-1)
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
        return self.model.get_material_texture()
