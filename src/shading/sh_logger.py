"""SH 着色模型调试日志。"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from src.config import Config
from src.shading.logger import ShadingLogger


class SHLogger(ShadingLogger):
    """SH 着色模型日志: Full/DC/HF compare, diffuse 纹理, 3 视频。"""

    def save_checkpoint(
        self, model, output_dir: str, epoch: int, loss: float, resolution: int,
    ) -> str:
        ckpt = model.state_dict()
        ckpt["epoch"] = epoch
        ckpt["loss"] = loss
        ckpt["resolution"] = resolution
        path = os.path.join(output_dir, "sh_texture.pt")
        torch.save(ckpt, path)
        return path

    def export_debug(
        self, model, renderer, dataset, output_dir: str, epoch: int,
        history: dict, device: str, current_resolution: int,
    ) -> None:
        import cv2
        from src.exporter import export_diffuse_texture
        from src.mesh import load_mesh
        from src.video import render_video

        tex = model.get_material_texture()

        # 1. Diffuse 贴图
        export_diffuse_texture(tex, os.path.join(output_dir, "diffuse.png"), self.config.texture.sh_order)

        # 2. Compare: GT | Full SH / DC Only | High Freq
        self._export_compare(model, renderer, dataset, output_dir, device, current_resolution)

        # 3. 视频: Full / DC / HF
        mesh = load_mesh(self.config.data.mesh_path)
        cfg = self.config
        vk = dict(
            mesh=mesh, center=cfg.video.center, radius=cfg.video.radius,
            height=cfg.video.height, num_frames=cfg.video.num_frames,
            fov_deg=cfg.video.fov_deg, resolution=cfg.video.resolution, fps=cfg.video.fps,
        )

        render_video(sh_texture=tex, output_path=os.path.join(output_dir, "orbit.mp4"), **vk)

        dc_tex = torch.cat([model.features_dc.data.detach().cpu(),
                            torch.zeros_like(model.features_rest.data.detach().cpu())], dim=-1)
        render_video(sh_texture=dc_tex, output_path=os.path.join(output_dir, "orbit_dc.mp4"), **vk)
        render_video(sh_texture=tex, output_path=os.path.join(output_dir, "orbit_hf.mp4"),
                     subtract_texture=dc_tex, **vk)

        print(f"  [Debug] diffuse + compare + video → {output_dir}")

    def _export_compare(self, model, renderer, dataset, output_dir, device, resolution):
        import cv2

        num_views = len(dataset)
        indices = [int(i * num_views / min(4, num_views)) for i in range(min(4, num_views))]
        dc = model.features_dc.data
        rest = model.features_rest.data

        for ci, idx in enumerate(indices):
            img_np, camera = dataset[idx]
            with torch.no_grad():
                rgb_full, mask, _ = renderer.render(model.features_dc, model.features_rest, camera)
                rgb_dc, _, _ = renderer.render(dc, rest * 0, camera)
            rgb_hf = (rgb_full - rgb_dc).clamp(0, 1)

            mask = mask.flip(1)
            mask_np = mask[0].cpu().numpy()

            def to_bgr(t):
                img = t[0].flip(0).clamp(0, 1).pow(1 / 2.2).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                bgr[mask_np < 0.5] = 0
                return bgr

            gt = cv2.cvtColor((img_np.transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            panels = [(gt, "GT"), (to_bgr(rgb_full), "Full SH"), (to_bgr(rgb_dc), "DC Only"), (to_bgr(rgb_hf), "High Freq")]

            th = min(p[0].shape[0] for p in panels)
            rs = []
            for img, label in panels:
                h, w = img.shape[:2]
                r = cv2.resize(img, (w * th // h, th))
                cv2.putText(r, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                rs.append(r)

            canvas = np.concatenate([
                np.concatenate([rs[0], rs[1]], axis=1),
                np.concatenate([rs[2], rs[3]], axis=1),
            ], axis=0)
            cv2.imwrite(os.path.join(output_dir, f"compare_{ci:04d}.png"), canvas)
