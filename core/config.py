"""配置与 schema 校验（core.config）。

所有可调超参经环境变量覆盖，统一在此收敛并做范围校验。
前缀：ENV_FEYNMANFORGE_* 。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .errors import ConfigError

PREFIX = "FEYNMANFORGE_"


@dataclass
class Config:
    # ---- GP 引擎 ----
    pop_size: int = 240
    generations: int = 70
    max_depth_init: int = 4
    max_depth: int = 7
    tournament: int = 4
    crossover_rate: float = 0.85
    mutation_rate: float = 0.12
    hoist_rate: float = 0.03
    parsimony: float = 0.001
    # 常量候选集（含“漂亮”有理数与无理数，使真实常量可被精确命中）
    constants: tuple = (
        -3.0,
        -2.0,
        -1.0,
        -0.5,
        -0.25,
        0.25,
        0.5,
        1.0,
        2.0,
        3.0,
        3.141592653589793,
        2.718281828459045,
    )
    # 连续随机常量范围（异类搜索）
    const_range: tuple = (-2.0, 2.0)
    # 恢复阈值：测试集 R2 >= 该值视为“精确恢复”
    recovery_r2: float = 0.9999
    # 是否对选定个体做仿射精修 a*t(x)+b
    affine_refine: bool = True
    # 仿射精修迭代（岭回归）
    affine_iters: int = 5

    # ---- 数据 ----
    n_train: int = 200
    n_test: int = 400
    noise_std: float = 0.0

    # ---- 评测 ----
    seeds: tuple = (7, 19, 42)

    # ---- 随机种子 ----
    seed: int = 7

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.pop_size < 10:
            raise ConfigError("pop_size 必须 >= 10")
        if self.generations < 1:
            raise ConfigError("generations 必须 >= 1")
        if not (0.0 <= self.crossover_rate <= 1.0):
            raise ConfigError("crossover_rate 必须在 [0,1]")
        if self.max_depth < 2:
            raise ConfigError("max_depth 必须 >= 2")
        if not (0.0 <= self.recovery_r2 <= 1.0):
            raise ConfigError("recovery_r2 必须在 [0,1]")
        if self.n_train < 10 or self.n_test < 10:
            raise ConfigError("样本量过小")


def from_env(overrides: dict | None = None) -> Config:
    """从环境变量读取覆盖，返回校验后的 Config。"""
    cfg = Config()
    mapping = {
        "POP_SIZE": "pop_size",
        "GENERATIONS": "generations",
        "MAX_DEPTH": "max_depth",
        "MAX_DEPTH_INIT": "max_depth_init",
        "TOURNAMENT": "tournament",
        "CROSSOVER_RATE": "crossover_rate",
        "MUTATION_RATE": "mutation_rate",
        "RECOVERY_R2": "recovery_r2",
        "N_TRAIN": "n_train",
        "N_TEST": "n_test",
        "NOISE_STD": "noise_std",
        "SEED": "seed",
        "AFFINE_REFINE": "affine_refine",
    }
    for env_key, attr in mapping.items():
        val = os.environ.get(PREFIX + env_key)
        if val is None:
            continue
        cur = getattr(cfg, attr)
        try:
            if isinstance(cur, bool):
                setattr(cfg, attr, val.lower() in ("1", "true", "yes", "on"))
            elif isinstance(cur, int):
                setattr(cfg, attr, int(val))
            elif isinstance(cur, float):
                setattr(cfg, attr, float(val))
            else:
                setattr(cfg, attr, type(cur)(val))
        except ValueError as e:  # pragma: no cover
            raise ConfigError(f"环境变量 {PREFIX + env_key}={val!r} 无法解析: {e}")
    if overrides:
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    cfg._validate()
    return cfg
