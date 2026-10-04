"""全局确定性入口（core.seed）。

所有随机源（标准库 random、numpy）统一在此设种子，保证同 seed 两次运行
核心指标逐位一致。禁止在模块内另起随机源。
"""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["get_seed", "set_all"]


_SEED: int | None = None


def set_all(seed: int) -> int:
    """一次性设齐所有随机源并返回 seed。

    - 标准库 random
    - numpy（含默认 RNG 与 Legacy RandomState）
    - 环境变量 PYTHONHASHSEED（影响 set/dict 哈希，间接影响无种子遍历顺序）
    """
    global _SEED
    seed = int(seed)
    _SEED = seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed % (2**32))
    return seed


def get_seed() -> int | None:
    return _SEED
