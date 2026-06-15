"""Neural Lightmap 调试日志 — compare atlas + checkpoint + 视频。"""
from __future__ import annotations

import os

import numpy as np
import torch

from src.config import Config
from src.shading.logger import ShadingLogger


class NLMLogger(ShadingLogger):
    """NLM 着色模型日志。"""

    def save_checkpoint(
        self, model, output_dir: str, epoch: int, loss: float, resolution: int,
    ) -> str:
        ckpt = model.state_dict()
        ckpt["epoch"] = epoch
        ckpt["loss"] = loss
        ckpt["resolution"] = resolution
        path = os.path.join(output_dir, "nlm_checkpoint.pt")
        torch.save(ckpt, path)
        return path

    def export_debug(
        self, model, renderer, dataset, output_dir: str, epoch: int,
        history: dict, device: str, current_resolution: int,
        **kwargs,
    ) -> None:
        import cv2
        from src.mesh import load_mesh
        from src.video import render_video, render_video_multi

        # 1. Export feature maps + MLP weights
        model.export(output_dir)
        print(f"  [Debug] NLM feature maps + MLP weights → {output_dir}")

        is_multi = kwargs.get("is_multi", False)
        renderers = kwargs.get("renderers", None)
        submesh_names = kwargs.get("submesh_names", None)

        # 2. Compare images
        if is_multi and renderers is not None:
            self._export_compare_multi(
                model, renderers, submesh_names, dataset, output_dir, device, current_resolution
            )
        else:
            self._export_compare(model, renderer, dataset, output_dir, device, current_resolution)

        # 3. Orbit video
        mesh = load_mesh(self.config.data.mesh_path)
        cfg = self.config
        vk = dict(
            center=cfg.video.center, radius=cfg.video.radius,
            height=cfg.video.height, num_frames=cfg.video.num_frames,
            fov_deg=cfg.video.fov_deg, resolution=cfg.video.resolution, fps=cfg.video.fps,
        )
        if is_multi:
            render_video_multi(
                mesh=mesh, shading_model=model, submesh_names=submesh_names,
                output_path=os.path.join(output_dir, "orbit.mp4"), **vk,
            )
        else:
            render_video(
                mesh=mesh, shading_model=model,
                output_path=os.path.join(output_dir, "orbit.mp4"), **vk,
            )

        print(f"  [Debug] compare + video → {output_dir}")

    def _export_compare(self, model, renderer, dataset, output_dir, device, resolution):
        self._export_compare_impl(model, renderer, dataset, output_dir, device, resolution, is_multi=False)

    def _export_compare_multi(self, model, renderers, submesh_names, dataset, output_dir, device, resolution):
        self._export_compare_impl(model, renderers, dataset, output_dir, device, resolution,
                                  is_multi=True, submesh_names=submesh_names)

    def _export_compare_impl(
        self, model, renderer_or_renderers, dataset, output_dir, device, resolution,
        is_multi=False, submesh_names=None,
    ):
        import cv2

        num_views = len(dataset)
        indices = [int(i * num_views / min(4, num_views)) for i in range(min(4, num_views))]

        for ci, idx in enumerate(indices):
            img_np, camera = dataset[idx]

            with torch.no_grad():
                if is_multi:
                    rendered = torch.zeros(1, resolution, resolution, 3, device=device)
                    depth_buf = torch.full((1, resolution, resolution), float("inf"), device=device)
                    mask = torch.zeros(1, resolution, resolution, device=device)
                    for sub_name in submesh_names:
                        sub_renderer = renderer_or_renderers[sub_name]
                        rast, texc, wpos, inorm, vdir, tang, btang = sub_renderer.rasterize_and_interpolate(camera)
                        rgb_sub, mask_sub = model.shade_submesh(
                            sub_name, rast, texc, wpos, inorm, vdir, camera, resolution, tang, btang
                        )
                        sub_depth = rast[..., 2]
                        write = (mask_sub > 0.5) & (sub_depth < depth_buf)
                        rendered = torch.where(write.unsqueeze(-1), rgb_sub, rendered)
                        depth_buf = torch.where(write, sub_depth, depth_buf)
                        mask = torch.max(mask, mask_sub)
                else:
                    rast, texc, wpos, inorm, vdir, tang, btang = renderer_or_renderers.rasterize_and_interpolate(camera)
                    rendered, mask = model.shade(rast, texc, wpos, inorm, vdir, camera, resolution)

            # Feature visualization (first 3 channels of __default__ or first submesh)
            debug = model.get_debug_info()
            feature = debug.get("feature", rendered * 0)
            feat_vis = feature[..., :3].clamp(-1, 1)
            feat_vis = (feat_vis + 1) * 0.5  # to [0,1]

            mask = mask.flip(1)
            mask_np = mask[0].cpu().numpy()

            def to_bgr(t, gamma=True):
                img = t[0].flip(0).clamp(0, 1).detach().cpu().numpy()
                if gamma:
                    img = img ** (1 / 2.2)
                img = (img * 255).astype(np.uint8)
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                bgr[mask_np < 0.5] = 0
                return bgr

            gt = cv2.cvtColor((img_np.transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            panels = [
                (gt, "GT"),
                (to_bgr(rendered), "NLM"),
                (to_bgr(feat_vis, gamma=False), "Feature[0:3]"),
                (to_bgr(rendered * 0 + 0.5, gamma=False), "Residual"),
            ]

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
