"""Extract GT panels (Rendered/Diffuse/Specular) from compare_NNNN.png for report.

Usage:
    python scripts/extract_gt_panels.py --scene piano --out-dir app/resource/piano_no_normal_ab
    python scripts/extract_gt_panels.py --scene helmet --out-dir app/resource/helmet_no_normal_ab

Each compare_NNNN.png is a 2x2 grid (2048x2048 typically):
    [GT, Rendered, Diffuse, Specular] per pbr_logger.py layout.
Crops to panels and saves per-camera panels:
    gt_rendered_cam{idx}.png, gt_diffuse_cam{idx}.png, gt_specular_cam{idx}.png,
    gt_compare_{NNNN}_full.png (full original).
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# camera index -> compare file index mapping
# cam0 -> compare_0000, cam50 -> compare_0001, cam100 -> compare_0002, cam150 -> compare_0003
CAM_TO_FILE = {0: 0, 50: 1, 100: 2, 150: 3}


def extract(scene: str, out_dir: Path) -> None:
    src_dir = Path(f'output/{scene}_no_normal/epoch2000')
    out_dir.mkdir(parents=True, exist_ok=True)

    for cam_idx, file_idx in CAM_TO_FILE.items():
        src = src_dir / f'compare_{file_idx:04d}.png'
        if not src.exists():
            print(f'WARN: {src} not found, skipping cam{cam_idx}')
            continue

        img = np.array(Image.open(src).convert('RGB'))
        h, w = img.shape[:2]
        # 2x2 grid: top-left=GT, top-right=Rendered, bottom-left=Diffuse, bottom-right=Specular
        rendered = img[: h // 2, w // 2:]
        diffuse = img[h // 2:, : w // 2]
        specular = img[h // 2:, w // 2:]

        Image.fromarray(rendered).save(out_dir / f'gt_rendered_cam{cam_idx}.png')
        Image.fromarray(diffuse).save(out_dir / f'gt_diffuse_cam{cam_idx}.png')
        Image.fromarray(specular).save(out_dir / f'gt_specular_cam{cam_idx}.png')
        shutil.copy(src, out_dir / f'gt_compare_{file_idx:04d}_full.png')
        print(f'cam{cam_idx}: {src.name} -> 3 panels + full saved to {out_dir}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True, choices=['piano', 'helmet'])
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    extract(args.scene, Path(args.out_dir))


if __name__ == '__main__':
    main()
