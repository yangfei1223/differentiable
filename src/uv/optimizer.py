"""UV 优化器 — L-BFGS 包装。"""
from __future__ import annotations

from typing import Callable

import torch

from src.uv.param import UVParameterizer


class UVOptimizer:
    """UV 坐标的 L-BFGS 优化器。

    包装 torch.optim.LBFGS，提供 zero_grad / step 接口。

    Args:
        uv_param: UVParameterizer 实例。
        lr: 学习率。
        max_iter: L-BFGS 每步最大迭代次数。
    """

    def __init__(self, uv_param: UVParameterizer, lr: float = 0.001, max_iter: int = 20):
        self.uv_param = uv_param
        self._optimizer = torch.optim.LBFGS(
            uv_param.parameters(),
            lr=lr,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

    def zero_grad(self) -> None:
        """清零梯度。"""
        self._optimizer.zero_grad()
        if self.uv_param.raw.grad is not None:
            self.uv_param.raw.grad.zero_()

    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """执行一步 L-BFGS 优化。"""
        return self._optimizer.step(closure)
