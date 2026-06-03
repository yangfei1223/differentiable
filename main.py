"""CLI 入口点。"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from src.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="可微烘焙管线")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="配置文件路径")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "export", "video"],
                        help="运行模式: train, export 或 video")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="导出/视频模式下 SH 纹理检查点路径 (.pt)")
    parser.add_argument("--resume", type=str, default=None,
                        help="断点续训的 checkpoint 路径 (.pt)")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("differentiable-bake")

    args = parse_args()
    cfg = load_config(args.config)

    if args.mode == "train":
        logger.info("=" * 60)
        logger.info("可微烘焙管线 — 训练模式")
        logger.info("=" * 60)
        logger.info(f"  网格: {cfg.data.mesh_path}")
        logger.info(f"  GT 目录: {cfg.data.gt_dir}")
        logger.info(f"  SH 阶数: {cfg.texture.sh_order}")
        logger.info(f"  目标分辨率: {cfg.texture.target_resolution}")
        logger.info(f"  训练轮数: {cfg.training.num_epochs}")

        from src.trainer import Trainer
        trainer = Trainer(cfg)
        output_dir = Path(cfg.export.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        trainer.train(
            output_dir=str(output_dir),
            checkpoint_every=200,
            resume_from=args.resume,
        )
        ckpt_path = output_dir / "sh_texture.pt"
        torch.save(trainer.get_sh_texture(), str(ckpt_path))
        logger.info(f"最终检查点已保存: {ckpt_path}")

    elif args.mode == "export":
        if args.checkpoint is None:
            logger.error("导出模式需要 --checkpoint 参数")
            sys.exit(1)

        logger.info("可微烘焙管线 — 导出模式")
        sh_texture = torch.load(args.checkpoint, map_location="cpu")

        output_dir = Path(cfg.export.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if cfg.export.format == "gltf":
            from src.exporter import export_diffuse_texture
            tex_path = str(output_dir / "diffuse.png")
            export_diffuse_texture(sh_texture, tex_path, cfg.texture.sh_order)
            logger.info(f"漫反射贴图已导出: {tex_path}")

        elif cfg.export.format == "custom":
            from src.exporter import export_sh_channels
            paths = export_sh_channels(
                sh_texture, str(output_dir / "sh_channels"), cfg.texture.sh_order
            )
            logger.info(f"SH 通道已导出 ({len(paths)} 张)")

    elif args.mode == "video":
        if args.checkpoint is None:
            logger.error("视频模式需要 --checkpoint 参数")
            sys.exit(1)

        logger.info("可微烘焙管线 — 视频导出")
        sh_texture = torch.load(args.checkpoint, map_location="cpu")

        from src.mesh import load_mesh
        from src.video import render_video

        mesh = load_mesh(cfg.data.mesh_path)
        video_cfg = cfg.video
        video_path = str(Path(cfg.export.output_dir) / "orbit.mp4")

        logger.info(f"  分辨率: {video_cfg.resolution}, 帧数: {video_cfg.num_frames}, FPS: {video_cfg.fps}")
        render_video(
            sh_texture=sh_texture,
            mesh=mesh,
            output_path=video_path,
            center=video_cfg.center,
            radius=video_cfg.radius,
            height=video_cfg.height,
            num_frames=video_cfg.num_frames,
            fov_deg=video_cfg.fov_deg,
            resolution=video_cfg.resolution,
            fps=video_cfg.fps,
        )
        logger.info(f"视频已导出: {video_path}")

    logger.info("完成。")


if __name__ == "__main__":
    main()
