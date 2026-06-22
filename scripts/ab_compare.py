"""AB pixel comparison: Web render at camera[50] vs Python training GT."""
import numpy as np
from PIL import Image

web = np.array(Image.open('app/debug/ab_web_cam50.png').convert('RGB'))
gt_full = np.array(Image.open('output/helmet_260604_pbr/epoch2000/compare_0001.png').convert('RGB'))

# GT is 2048x2048 2x2 grid; top-right is 'Rendered' panel
gh, gw = gt_full.shape[:2]
gt = gt_full[:gh//2, gw//2:]  # 1024x1024

print(f'Web screenshot: {web.shape}')
print(f'GT rendered panel: {gt.shape}')

# Resize GT to match web size for comparison
gt_resized = np.array(Image.fromarray(gt).resize((web.shape[1], web.shape[0]), Image.LANCZOS))
print(f'GT resized: {gt_resized.shape}')

# Crop UI out of web (assume UI is in corners; rough crop to viewport area)
# For now just compare whole-frame statistics
def stats(img, name):
    fg_mask = img.max(axis=2) > 30
    fg = img[fg_mask]
    print(f'{name}: fg%={fg_mask.mean()*100:.1f}%, all-pixel mean={img.mean(axis=(0,1)).astype(int).tolist()}, fg mean={fg.mean(axis=0).astype(int).tolist()}, fg max={fg.max(axis=0).tolist()}')

stats(web, 'Web ')
stats(gt_resized, 'GT  ')

# PSNR (only over non-black pixels in both)
web_fg = web.max(axis=2) > 30
gt_fg = gt_resized.max(axis=2) > 30
both_fg = web_fg & gt_fg
print(f'\nBoth-fg pixel count: {both_fg.sum()} ({both_fg.mean()*100:.1f}%)')
if both_fg.sum() > 0:
    diff = web[both_fg].astype(np.float32) - gt_resized[both_fg].astype(np.float32)
    mse = (diff ** 2).mean()
    rmse = np.sqrt(mse)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    print(f'PSNR (overlap region): {psnr:.2f} dB')
    print(f'RMSE: {rmse:.2f}')
    print(f'Mean diff (web-gt): {diff.mean(axis=0).astype(int).tolist()}')
    print(f'Abs diff mean: {np.abs(diff).mean(axis=0).astype(int).tolist()}')

# Diff visualization
h, w = web.shape[:2]
diff_img = np.abs(web.astype(np.int16) - gt_resized.astype(np.int16)).astype(np.uint8)
# Scale up 3x for visibility
diff_img_scaled = np.clip(diff_img * 3, 0, 255).astype(np.uint8)
Image.fromarray(diff_img_scaled).save('app/debug/ab_diff_web_vs_gt.png')
print('Saved diff: app/debug/ab_diff_web_vs_gt.png')

# Save side-by-side
side = np.concatenate([gt_resized, web, diff_img_scaled], axis=1)
Image.fromarray(side).save('app/debug/ab_side_by_side.png')
print('Saved side-by-side: app/debug/ab_side_by_side.png')
