"""接口契约（core.interfaces）。

所有符号回归后端统一实现 SRBackend，管道只依赖该 Protocol，确保可插拔与
可独立验证。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .types import SRResult


@runtime_checkable
class SRBackend(Protocol):
    name: str

    def available(self) -> bool:
        """后端是否可用（重型依赖缺失时返回 False，触发离线兜底）。"""
        ...

    def fit(self, X: np.ndarray, y: np.ndarray, n_vars: int) -> SRResult:
        """在 (X, y) 上拟合，返回 SRResult。"""
        ...
