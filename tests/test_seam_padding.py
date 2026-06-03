"""UV seam padding 测试 — 边界膨胀算子。"""
from __future__ import annotations

import torch
import pytest

from src.seam_padding import dilate_texture


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def _ones_block(H: int, W: int, cy: int, cx: int, size: int, channels: int = 3):
    """返回 [1,H,W,C] 的零张量，中心 (cy,cx) 处有 size×size 的 1.0 块。"""
    tex = torch.zeros(1, H, W, channels)
    half = size // 2
    tex[0, cy - half : cy + half + 1, cx - half : cx + half + 1, :] = 1.0
    return tex


def _mask_from_texture(tex: torch.Tensor) -> torch.Tensor:
    """从纹理生成 valid_mask: 任意通道 > 0 → 1。"""
    return (tex.sum(dim=-1, keepdim=True) > 0).float()


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------
class TestDilateTexture:
    def test_dilation_fills_neighbors(self):
        """2×2 block of value 1.0 in center of 8×8, dilate radius=1 → 邻居应被填充。"""
        H, W, C = 8, 8, 3
        # 中心 4×4 区域偏左上放一个 2×2 的块 (行 3:5, 列 3:5)
        tex = _ones_block(H, W, cy=4, cx=4, size=2, channels=C)
        mask = _mask_from_texture(tex)

        result = dilate_texture(tex, mask, radius=1)

        # 原始 2×2 块是 (3,3)~(4,4)
        # radius=1 意味着至少填入周围的 1 像素
        # 检查 (2,3) — 原始块上方一行, 应该有非零值
        assert result[0, 2, 3, 0].item() > 0.0, "Dilation should fill pixel above block"
        # 检查 (5,4) — 原始块下方一行
        assert result[0, 5, 4, 0].item() > 0.0, "Dilation should fill pixel below block"
        # 检查 (3,2) — 原始块左侧一列
        assert result[0, 3, 2, 0].item() > 0.0, "Dilation should fill pixel left of block"
        # 检查 (4,5) — 原始块右侧一列
        assert result[0, 4, 5, 0].item() > 0.0, "Dilation should fill pixel right of block"

    def test_dilation_preserves_original(self):
        """原始有效像素在膨胀后应保持精确值不变。"""
        H, W, C = 8, 8, 3
        tex = _ones_block(H, W, cy=4, cx=4, size=2, channels=C)
        mask = _mask_from_texture(tex)

        result = dilate_texture(tex, mask, radius=1)

        # mask > 0 的位置应和原始值完全一致
        # mask is [1,H,W,1], tex/result are [1,H,W,C] — squeeze to [H,W] for boolean indexing
        mask_bool = mask.squeeze(-1).squeeze(0) > 0  # [H, W]
        original_pixels = tex[0][mask_bool]   # [N_valid, C]
        result_pixels = result[0][mask_bool]  # [N_valid, C]
        assert torch.allclose(original_pixels, result_pixels, atol=1e-6), \
            "Original valid pixels must be preserved exactly"

    def test_larger_radius_fills_more(self):
        """radius=3 应比 radius=1 填充更大的面积。"""
        H, W, C = 16, 16, 3
        # 放一个 1×1 的点
        tex = torch.zeros(1, H, W, C)
        tex[0, 8, 8, :] = 1.0
        mask = _mask_from_texture(tex)

        result_r1 = dilate_texture(tex, mask, radius=1)
        result_r3 = dilate_texture(tex, mask, radius=3)

        filled_r1 = (result_r1 > 0).sum().item()
        filled_r3 = (result_r3 > 0).sum().item()

        assert filled_r3 > filled_r1, \
            f"radius=3 ({filled_r3} pixels) should fill more area than radius=1 ({filled_r1} pixels)"
