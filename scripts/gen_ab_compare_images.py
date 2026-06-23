"""Generate AB comparison images and PSNR stats for a scene report.

Inputs (already produced):
    app/resource/{scene}_no_normal_ab/
        gt_rendered_cam{0,50,100,150}.png      (1024x1024, GT Rendered panel)
        gt_diffuse_cam{...}.png                 (1024x1024, GT Diffuse panel)
        gt_specular_cam{...}.png                (1024x1024, GT Specular panel)
        web_cam{...}.png                        (Web viewport screenshot, final)
        diffuse_web_cam{...}.png                (Web diffuse debug channel)
        specular_web_cam{...}.png               (Web specular debug channel)

Outputs:
    compare_final_cam{...}.png     (left=GT Rendered, right=Web final, side by side)
    compare_diffuse_cam{...}.png
    compare_specular_cam{...}.png
    psnr.txt                       (per-cam final PSNR + summary)

Note: Web screenshots are viewport-sized (not 1024x1024). We center-crop
the largest square from the viewport, then resize to GT panel size for
visual side-by-side and PSNR. GT panels are gamma-encoded (pow(1/2.2))
while Web channels are linear values, so direct PSNR is approximate;
visual side-by-side is the primary signal.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

CAMS = [0, 50, 100, 150]
CHANNELS = [
    # (channel_name, gt_panel_filename_prefix, web_filename_prefix, apply_gamma_to_web)
    # GT panels are pow(1/2.2) encoded. Web final (rgb) is also sRGB-encoded by Three.js,
    # so no extra gamma. Web diffuse/specular debug outputs are linear; we apply pow(1/2.2)
    # for visual side-by-side only (NOT for PSNR — PSNR uses raw web values).
    ('final',    'gt_rendered',  'web',         False),
    ('diffuse',  'gt_diffuse',   'diffuse_web', True),
    ('specular', 'gt_specular',  'specular_web', True),
]


def center_crop_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s]


def resize(img: np.ndarray, size: int) -> np.ndarray:
    resample = getattr(Image, 'Resampling', getattr(Image, 'LANCZOS', 1))
    resample = getattr(resample, 'LANCZOS', resample)
    return np.array(Image.fromarray(img).resize((size, size), resample))


def apply_gamma(img: np.ndarray) -> np.ndarray:
    """Apply pow(1/2.2) for visual matching to GT panels."""
    f = img.astype(np.float32) / 255.0
    f = np.clip(f, 0, 1) ** (1.0 / 2.2)
    return (f * 255).astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = (diff ** 2).mean()
    return float('inf') if mse == 0 else float(10 * np.log10(255 ** 2 / mse))


def process(scene: str, out_dir: Path) -> None:
    psnr_lines = []
    for cam in CAMS:
        for ch_name, gt_prefix, web_prefix, gamma in CHANNELS:
            gt_path = out_dir / f'{gt_prefix}_cam{cam}.png'
            web_path = out_dir / f'{web_prefix}_cam{cam}.png'
            if not gt_path.exists() or not web_path.exists():
                print(f'  skip cam{cam} {ch_name}: missing {gt_path.name} or {web_path.name}')
                continue
            gt = np.array(Image.open(gt_path).convert('RGB'))
            web_raw = np.array(Image.open(web_path).convert('RGB'))
            # Center-crop square + resize to GT size
            web_crop = center_crop_square(web_raw)
            web = resize(web_crop, gt.shape[0])
            # Visual gamma adjust for side-by-side
            web_vis = apply_gamma(web) if gamma else web
            # Stack side by side
            side = np.concatenate([gt, web_vis], axis=1)
            Image.fromarray(side).save(out_dir / f'compare_{ch_name}_cam{cam}.png')
            # PSNR on overlap (non-black in both)
            if ch_name == 'final':
                web_fg = web.max(axis=2) > 30
                gt_fg = gt.max(axis=2) > 30
                both = web_fg & gt_fg
                if both.sum() > 0:
                    p = psnr(web[both], gt[both])
                    psnr_lines.append(f'cam{cam}: PSNR(overlap)={p:.2f} dB, '
                                       f'overlap={both.mean()*100:.1f}%')
                    print(f'  cam{cam} final: PSNR={p:.2f} dB overlap={both.mean()*100:.1f}%')
    if psnr_lines:
        (out_dir / 'psnr.txt').write_text('\n'.join(psnr_lines) + '\n', encoding='utf-8')
        print(f'wrote {out_dir / "psnr.txt"}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True, choices=['piano', 'helmet'])
    args = ap.parse_args()
    out_dir = Path(f'app/resource/{args.scene}_no_normal_ab')
    print(f'=== {args.scene} ===')
    process(args.scene, out_dir)


if __name__ == '__main__':
    main()
