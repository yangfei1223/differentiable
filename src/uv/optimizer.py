"""UV 优化器 — Adam 包装。"""
from __future__ import annotations

import torch

from src.uv.param import UVParameterizer


class UVOptimizer:
    """UV 坐标的 Adam 优化器。

    用 Adam 替代 L-BFGS：354K 维度的稀疏梯度空间，
    L-BFGS 步长极小（max 0.0018px），Adam 的逐参数自适应
    学习率更适合。

    Args:
        uv_param: UVParameterizer 实例。
        lr: 学习率。
    """

    def __init__(self, uv_param: UVParameterizer, lr: float = 0.001, max_iter: int = 1):
        self.uv_param = uv_param
        self._optimizer = torch.optim.Adam(uv_param.parameters(), lr=lr)

    def zero_grad(self) -> None:
        """清零梯度。"""
        self._optimizer.zero_grad()

    def step(self, loss: torch.Tensor) -> torch.Tensor:
        """执行一步 Adam 优化。

        Args:
            loss: 已计算 backward 的 loss（或需要先 backward）。
        """
        loss.backward()
        self._optimizer.step()
        return loss
