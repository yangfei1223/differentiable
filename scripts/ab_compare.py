"""AB pixel comparison: Web render at camera[50] vs Python training GT.

Uses no_normal output as GT (matches uNormalMapEnabled=false in shader).

Usage:
    python scripts/ab_compare.py helmet
    python scripts/ab_compare.py piano

Reads:
    app/debug/ab_web_{scene}_cam50.png — Web viewer 1024x1024 render
    output/{scene}_no_normal/epoch2000/compare_0001.png — GT (4-panel)

GT panel layout (from pbr_logger.py):
    [GT, Rendered, Diffuse, Specular] in 2x2 grid.
    Top-right (rows 0..h/2, cols w/2..w) = Rendered (model output, pow(1/2.2) encoded).
"""
import sys
import numpy as np
from PIL import Image


def compare(scene):
    web_path = f'app/debug/ab_web_{scene}_cam50.png'
    gt_full_path = f'output/{scene}_no_normal/epoch2000/compare_0001.png'

    web = np.array(Image.open(web_path).convert('RGB'))
    gt_full = np.array(Image.open(gt_full_path).convert('RGB'))

    # GT is 2048x2048 2x2 grid; top-right is 'Rendered' panel
    gh, gw = gt_full.shape[:2]
    gt = gt_full[:gh // 2, gw // 2:]  # 1024x1024

    print(f'=== {scene} ===')
    print(f'Web render: {web.shape}')
    print(f'GT rendered panel: {gt.shape}')

    # Both should be 1024x1024 — no resize needed if Web is offscreen
    if web.shape != gt.shape:
        try:
            resample = Image.Resampling.LANCZOS
        except (AttributeError, ImportError):
            resample = getattr(Image, 'LANCZOS', 1)
        gt_resized = np.array(Image.fromarray(gt).resize(
            (web.shape[1], web.shape[0]), resample))
        print(f'GT resized to: {gt_resized.shape}')
    else:
        gt_resized = gt

    def stats(img, name):
        fg_mask = img.max(axis=2) > 30
        fg = img[fg_mask]
        print(f'{name}: fg%={fg_mask.mean() * 100:.1f}%, '
              f'all-mean={img.mean(axis=(0, 1)).astype(int).tolist()}, '
              f'fg-mean={fg.mean(axis=0).astype(int).tolist()}, '
              f'fg-max={fg.max(axis=0).tolist()}')

    stats(web, 'Web ')
    stats(gt_resized, 'GT  ')

    # PSNR over overlap region (non-black in both)
    web_fg = web.max(axis=2) > 30
    gt_fg = gt_resized.max(axis=2) > 30
    both_fg = web_fg & gt_fg

    print(f'\nOverlap fg pixels: {both_fg.sum()} ({both_fg.mean() * 100:.1f}%)')
    if both_fg.sum() > 0:
        diff = web[both_fg].astype(np.float32) - gt_resized[both_fg].astype(np.float32)
        mse = (diff ** 2).mean()
        rmse = np.sqrt(mse)
        psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float('inf')
        print(f'PSNR (overlap): {psnr:.2f} dB')
        print(f'RMSE: {rmse:.2f}')
        print(f'Mean diff (web-gt): {diff.mean(axis=0).astype(int).tolist()}')
        print(f'Abs diff mean: {np.abs(diff).mean(axis=0).astype(int).tolist()}')

    # Diff visualization
    diff_img = np.abs(web.astype(np.int16) - gt_resized.astype(np.int16)).astype(np.uint8)
    diff_scaled = np.clip(diff_img * 3, 0, 255).astype(np.uint8)
    Image.fromarray(diff_scaled).save(f'app/debug/ab_diff_{scene}.png')
    side = np.concatenate([gt_resized, web, diff_scaled], axis=1)
    Image.fromarray(side).save(f'app/debug/ab_side_by_side_{scene}.png')
    print(f'Saved: app/debug/ab_diff_{scene}.png, ab_side_by_side_{scene}.png')


if __name__ == '__main__':
    scene = sys.argv[1] if len(sys.argv) > 1 else 'helmet'
    compare(scene)
