"""UV 坐标参数化 — sigmoid 编码，保证输出在 (0, 1)。"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


def _inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """sigmoid 的逆函数: logit(x) = log(x / (1-x))。"""
    x = x.clamp(1e-6, 1 - 1e-6)
    return torch.log(x / (1 - x))


class UVParameterizer(nn.Module):
    """可微 UV 坐标参数化。

    用 sigmoid 解码保证 UV 坐标严格在 (0, 1) 内。
    内部存储 sigmoid 的逆（logit）作为优化参数。
    """

    def __init__(self, initial_uvs: np.ndarray, uv_idx: np.ndarray):
        super().__init__()
        uvs_tensor = torch.from_numpy(initial_uvs.astype(np.float32))
        self.raw = nn.Parameter(_inverse_sigmoid(uvs_tensor))
        self.register_buffer(
            "_uv_idx",
            torch.from_numpy(uv_idx.astype(np.int32)),
        )

    def get_uvs(self) -> torch.Tensor:
        """返回解码后的 UV 坐标 [V, 2]，值在 (0, 1)。"""
        return torch.sigmoid(self.raw)

    def get_uv_idx(self) -> torch.Tensor:
        """返回 UV 索引 [F, 3]，int32。"""
        return self._uv_idx
