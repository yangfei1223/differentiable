"""PBR 着色模型调试日志。"""
from __future__ import annotations

import os

import numpy as np
import torch

from src.config import Config
from src.shading.logger import ShadingLogger


class PBRLogger(ShadingLogger):
    """PBR 着色模型日志: GT/Rendered/Diffuse/Specular compare, 4 贴图 + BRDF LUT, 3 视频。"""

    def save_checkpoint(
        self, model, output_dir: str, epoch: int, loss: float, resolution: int,
    ) -> str:
        ckpt = model.state_dict()
        ckpt["epoch"] = epoch
        ckpt["loss"] = loss
        ckpt["resolution"] = resolution
        ckpt["brdf_lut"] = model.brdf_lut  # 保存 BRDF LUT，避免重新生成
        path = os.path.join(output_dir, "pbr_checkpoint.pt")
        torch.save(ckpt, path)
        return path

    def export_debug(
        self, model, renderer, dataset, output_dir: str, epoch: int,
        history: dict, device: str, current_resolution: int,
    ) -> None:
        import cv2
        from src.mesh import load_mesh
        from src.video import render_video

        # 1. 导出 PBR 材质贴图: base_color, roughness, metallic, env_map
        model.export(output_dir)
        # 额外保存 BRDF LUT
        brdf_path = os.path.join(output_dir, "brdf_lut.pt")
        torch.save(model.brdf_lut, brdf_path)
        print(f"  [Debug] PBR textures + BRDF LUT → {output_dir}")

        # 2. Compare: GT | Rendered / Diffuse | Specular
        self._export_compare(model, renderer, dataset, output_dir, device, current_resolution)

        # 3. 视频: Full / Diffuse / Specular
        mesh = load_mesh(self.config.data.mesh_path)
        cfg = self.config
        vk = dict(
            center=cfg.video.center, radius=cfg.video.radius,
            height=cfg.video.height, num_frames=cfg.video.num_frames,
            fov_deg=cfg.video.fov_deg, resolution=cfg.video.resolution, fps=cfg.video.fps,
        )

        # Full PBR
        render_video(mesh=mesh, shading_model=model,
                     output_path=os.path.join(output_dir, "orbit.mp4"), **vk)

        # Diffuse only: 置零 specular (metallic=0, roughness=1)
        self.render_component_video(model, mesh, output_dir, "orbit_diffuse.mp4",
                                    mode="diffuse", **vk)

        # Specular only: 置零 diffuse (metallic=1)
        self.render_component_video(model, mesh, output_dir, "orbit_specular.mp4",
                                    mode="specular", **vk)

        print(f"  [Debug] compare + textures + video → {output_dir}")

    def _export_compare(self, model, renderer, dataset, output_dir, device, resolution):
        import cv2

        num_views = len(dataset)
        indices = [int(i * num_views / min(4, num_views)) for i in range(min(4, num_views))]

        for ci, idx in enumerate(indices):
            img_np, camera = dataset[idx]

            with torch.no_grad():
                rast, texc, wpos, inorm, vdir = renderer.rasterize_and_interpolate(camera)
                rgb_full, mask = model.shade(rast, texc, wpos, inorm, vdir, camera, resolution)

            debug = model.get_debug_info()
            diffuse = debug.get("diffuse", rgb_full * 0)
            specular = debug.get("specular", rgb_full * 0)

            mask = mask.flip(1)
            mask_np = mask[0].cpu().numpy()

            def to_bgr(t):
                img = t[0].flip(0).clamp(0, 1).pow(1 / 2.2).detach().cpu().numpy()
                img = (img * 255).astype(np.uint8)
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                bgr[mask_np < 0.5] = 0
                return bgr

            gt = cv2.cvtColor((img_np.transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            panels = [
                (gt, "GT"),
                (to_bgr(rgb_full), "Rendered"),
                (to_bgr(diffuse), "Diffuse"),
                (to_bgr(specular), "Specular"),
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

    def render_component_video(self, model, mesh, output_dir, filename, mode, **vk):
        """渲染 diffuse/specular 单分量视频。

        临时修改模型材质，渲染后恢复。
        """
        import torch.nn as nn

        # 保存原始材质
        orig_mat = model.mat_texture.data.clone()

        # 构造覆盖材质: roughness=1(纯diffuse) 或 metallic=1(纯specular)
        if mode == "diffuse":
            # metallic=0, roughness=1 → 只有 diffuse
            override = orig_mat.clone()
            # sigmoid inverse of 1.0 ≈ large positive, sigmoid inverse of 0.0 ≈ -5.0
            override[..., 3] = 10.0   # roughness → sigmoid → ~1.0
            override[..., 4] = -5.0   # metallic → sigmoid → ~0.007
        elif mode == "specular":
            # metallic=1 → F0=base_color, kd=0
            override = orig_mat.clone()
            override[..., 4] = 10.0   # metallic → sigmoid → ~1.0

        model.mat_texture = nn.Parameter(override)
        try:
            from src.video import render_video
            render_video(mesh=mesh, shading_model=model, output_path=os.path.join(output_dir, filename), **vk)
        finally:
            model.mat_texture = nn.Parameter(orig_mat)
