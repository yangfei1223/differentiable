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
        **kwargs,
    ) -> None:
        import cv2
        from src.mesh import load_mesh
        from src.video import render_video

        # 1. 导出 PBR 材质贴图: base_color, roughness, metallic, env_map
        model.export(output_dir)
        # 保存 BRDF LUT (.pt + .png)
        brdf_path = os.path.join(output_dir, "brdf_lut.pt")
        torch.save(model.brdf_lut, brdf_path)
        self._export_brdf_lut_image(model.brdf_lut, output_dir)
        print(f"  [Debug] PBR textures + BRDF LUT → {output_dir}")

        is_multi = kwargs.get("is_multi", False)
        renderers = kwargs.get("renderers", None)
        submesh_names = kwargs.get("submesh_names", None)

        # 2. Compare images
        if is_multi and renderers is not None:
            self._export_compare_multi(model, renderers, submesh_names, dataset, output_dir, device, current_resolution)
        else:
            self._export_compare(model, renderer, dataset, output_dir, device, current_resolution)

        # 3. Videos
        mesh = load_mesh(self.config.data.mesh_path)
        cfg = self.config
        vk = dict(
            center=cfg.video.center, radius=cfg.video.radius,
            height=cfg.video.height, num_frames=cfg.video.num_frames,
            fov_deg=cfg.video.fov_deg, resolution=cfg.video.resolution, fps=cfg.video.fps,
        )

        if is_multi:
            from src.video import render_video_multi
            render_video_multi(mesh=mesh, shading_model=model,
                              submesh_names=submesh_names,
                              output_path=os.path.join(output_dir, "orbit.mp4"), **vk)
        else:
            render_video(mesh=mesh, shading_model=model,
                         output_path=os.path.join(output_dir, "orbit.mp4"), **vk)
            self.render_component_video(model, mesh, output_dir, "orbit_diffuse.mp4",
                                        mode="diffuse", **vk)
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
                rast, texc, wpos, inorm, vdir, tang, btan = renderer.rasterize_and_interpolate(camera)
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

    def _export_compare_multi(self, model, renderers, submesh_names, dataset, output_dir, device, resolution):
        """Multi-mesh compare: render all submeshes, composite, compare with GT."""
        import cv2

        num_views = len(dataset)
        indices = [int(i * num_views / min(4, num_views)) for i in range(min(4, num_views))]

        for ci, idx in enumerate(indices):
            img_np, camera = dataset[idx]

            with torch.no_grad():
                rendered = torch.zeros(1, resolution, resolution, 3, device=device)
                depth_buf = torch.full((1, resolution, resolution), float("inf"), device=device)
                mask = torch.zeros(1, resolution, resolution, device=device)

                for sub_name in submesh_names:
                    sub_renderer = renderers[sub_name]
                    rast, texc, wpos, inorm, vdir, tang, btang = sub_renderer.rasterize_and_interpolate(camera)
                    rgb_sub, mask_sub = model.shade_submesh(
                        sub_name, rast, texc, wpos, inorm, vdir, camera, resolution, tang, btang)
                    sub_depth = rast[..., 2]
                    write = (mask_sub > 0.5) & (sub_depth < depth_buf)
                    rendered = torch.where(write.unsqueeze(-1), rgb_sub, rendered)
                    depth_buf = torch.where(write, sub_depth, depth_buf)
                    mask = torch.max(mask, mask_sub)

            debug = model.get_debug_info()
            diffuse = debug.get("diffuse", rendered * 0)
            specular = debug.get("specular", rendered * 0)

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
                (to_bgr(rendered), "Rendered"),
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

    def _export_brdf_lut_image(self, brdf_lut: torch.Tensor, output_dir: str) -> None:
        """导出 BRDF LUT 为 PNG（左 scale / 右 bias 并排）。"""
        import cv2
        # brdf_lut: [size, size, 2] — scale 和 bias 各一个通道
        size = brdf_lut.shape[0]
        scale = brdf_lut[:, :, 0].numpy()  # [size, size]
        bias = brdf_lut[:, :, 1].numpy()

        # 归一化到 [0, 255]
        scale_img = (scale * 255).clip(0, 255).astype(np.uint8)
        bias_img = (bias * 255).clip(0, 255).astype(np.uint8)

        # 用 colormap 增强可读性
        scale_color = cv2.applyColorMap(scale_img, cv2.COLORMAP_VIRIDIS)
        bias_color = cv2.applyColorMap(bias_img, cv2.COLORMAP_VIRIDIS)

        # 拼文字标签
        scale_color = cv2.putText(scale_color, "Scale (F0 mult)", (4, 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        bias_color = cv2.putText(bias_color, "Bias (constant)", (4, 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 并排拼接
        canvas = np.concatenate([scale_color, bias_color], axis=1)
        cv2.imwrite(os.path.join(output_dir, "brdf_lut.png"), canvas)

    def render_component_video(self, model, mesh, output_dir, filename, mode, **vk):
        """渲染 diffuse/specular 单分量视频。

        直接渲染完整 PBR 帧，从 debug_info 提取对应分量，不修改材质。
        """
        from src.video import render_video

        # 创建 wrapper 模型，shade 时提取指定分量
        import torch.nn as nn

        class ComponentModel:
            """包装 PBRShadingModel，渲染指定分量（diffuse/specular）。"""
            def __init__(self, base_model, component):
                self.base = base_model
                self.component = component  # "diffuse" or "specular"
                self.mat_texture = base_model.mat_texture
                self.env_map = base_model.env_map

            def shade(self, rast_out, texc, world_pos, normals, view_dirs, camera,
                      resolution, tangents=None, bitangents=None):
                rgb, mask = self.base.shade(
                    rast_out, texc, world_pos, normals, view_dirs, camera,
                    resolution, tangents, bitangents,
                )
                debug = self.base.get_debug_info()
                component = debug.get(self.component, torch.zeros_like(rgb))
                return component, mask

        comp_model = ComponentModel(model, mode)
        comp_model.mat_texture = model.mat_texture
        comp_model.env_map = model.env_map

        render_video(
            mesh=mesh, shading_model=comp_model,
            output_path=os.path.join(output_dir, filename), **vk,
        )
