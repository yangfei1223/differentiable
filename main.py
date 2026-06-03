"""CLI 入口点。"""
from __future__ import annotations

import argparse
import sys

from src.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="可微烘焙管线")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="配置文件路径")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "export"],
                        help="运行模式: train 或 export")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="导出模式下 SH 纹理检查点路径 (.pt)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.mode == "train":
        print(f"[Train] 配置已加载: {args.config}")
        print(f"  网格: {cfg.data.mesh_path}")
        print(f"  SH 阶数: {cfg.texture.sh_order}")
        print(f"  目标分辨率: {cfg.texture.target_resolution}")
    elif args.mode == "export":
        if args.checkpoint is None:
            print("错误: 导出模式需要 --checkpoint 参数")
            sys.exit(1)
        print(f"[Export] 检查点: {args.checkpoint}")


if __name__ == "__main__":
    main()
