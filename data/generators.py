"""合成基准数据生成（data.generators）。

每个基准函数都来自已知解析式（含 AI-Feynman 风格的物理解），常量取自
``core.config.Config.constants`` 候选集，使真实方程可被精确命中。
数据生成固定 seed，可复现；预处理仅在 train 上 fit，杜绝泄漏。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

PI = 3.141592653589793
E = 2.718281828459045


@dataclass
class FunctionSpec:
    name: str
    n_vars: int
    f: Callable[[np.ndarray], np.ndarray]
    true_expr: str
    domain: list[tuple]
    difficulty: str = "easy"
    reference: str = ""


# ---- 函数库（解析式 + 物理出处）------------------------------------------
def _register() -> list[FunctionSpec]:
    specs: list[FunctionSpec] = []

    def add(name, n, fn, expr, dom, diff="easy", ref=""):
        specs.append(FunctionSpec(name, n, fn, expr, dom, diff, ref))

    # 1 变量代数/超越
    add("lin1", 1, lambda X: 2.0 * X[:, 0] + 1.0, "2*x0 + 1", [(-3, 3)], "easy")
    add(
        "quad1",
        1,
        lambda X: X[:, 0] ** 2 + X[:, 0] + 1.0,
        "x0**2 + x0 + 1",
        [(-3, 3)],
        "easy",
    )
    add(
        "cubic1",
        1,
        lambda X: X[:, 0] ** 3 + 2.0 * X[:, 0],
        "x0**3 + 2*x0",
        [(-2, 2)],
        "easy",
    )
    add("sin1", 1, lambda X: np.sin(X[:, 0]), "sin(x0)", [(-PI, PI)], "easy")
    add("exp1", 1, lambda X: np.exp(-(X[:, 0] ** 2)), "exp(-x0**2)", [(-3, 3)], "easy")
    add("sqrt1", 1, lambda X: np.sqrt(X[:, 0]), "sqrt(x0)", [(0.1, 4.0)], "easy")
    add("log1", 1, lambda X: np.log(1.0 + X[:, 0]), "log(1 + x0)", [(0.0, 3.0)], "easy")
    add(
        "quartic1",
        1,
        lambda X: X[:, 0] ** 4 + X[:, 0] ** 2 - 1.0,
        "x0**4 + x0**2 - 1",
        [(-2, 2)],
        "easy",
    )
    add(
        "trigmix1",
        1,
        lambda X: np.sin(X[:, 0]) * np.cos(X[:, 0]),
        "sin(x0)*cos(x0)",
        [(-PI, PI)],
        "easy",
    )
    # 2 变量
    add(
        "poly2d",
        2,
        lambda X: X[:, 0] ** 2 + X[:, 1] ** 2,
        "x0**2 + x1**2",
        [(-3, 3), (-3, 3)],
        "easy",
    )
    add("prod2", 2, lambda X: X[:, 0] * X[:, 1], "x0*x1", [(-3, 3), (-3, 3)], "easy")
    add(
        "dist2",
        2,
        lambda X: np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2),
        "sqrt(x0**2 + x1**2)",
        [(-3, 3), (-3, 3)],
        "easy",
    )
    add(
        "trig2",
        2,
        lambda X: np.sin(X[:, 0]) + np.cos(X[:, 1]),
        "sin(x0) + cos(x1)",
        [(-PI, PI), (-PI, PI)],
        "easy",
    )
    add(
        "gauss2d",
        2,
        lambda X: np.exp(-(X[:, 0] ** 2 + X[:, 1] ** 2)),
        "exp(-x0**2 - x1**2)",
        [(-3, 3), (-3, 3)],
        "easy",
    )
    add(
        "poly3mix",
        2,
        lambda X: X[:, 0] * X[:, 1] + X[:, 0] + X[:, 1] + 1.0,
        "x0*x1 + x0 + x1 + 1",
        [(-3, 3), (-3, 3)],
        "easy",
    )
    # 3 变量
    add(
        "lin3",
        3,
        lambda X: 0.5 * X[:, 0] + X[:, 1] - 2.0 * X[:, 2],
        "0.5*x0 + x1 - 2*x2",
        [(-3, 3)] * 3,
        "easy",
    )
    # Feynman 风格物理（Udrescu & Tegmark 2020 精神）
    add(
        "feynman_ke",
        2,
        lambda X: 0.5 * X[:, 0] * X[:, 1] ** 2,
        "0.5*x0*x1**2",
        [(0.1, 4.0), (-3, 3)],
        "easy",
        "动能 1/2 m v^2",
    )
    add(
        "feynman_g",
        3,
        lambda X: X[:, 0] * X[:, 1] / (X[:, 2] ** 2),
        "x0*x1/x2**2",
        [(0.1, 4.0), (0.1, 4.0), (0.5, 3.0)],
        "easy",
        "引力势 m1*m2/r^2 类比",
    )
    # 困难（有理/高阶，可能未精确恢复 → 诚实负例）
    add(
        "rat1",
        1,
        lambda X: 1.0 / (1.0 + X[:, 0] ** 2),
        "1/(1 + x0**2)",
        [(-2, 2)],
        "hard",
        "需深层有理式",
    )
    add(
        "rat2",
        2,
        lambda X: (X[:, 0] + X[:, 1]) / (1.0 + X[:, 0] * X[:, 1]),
        "(x0 + x1)/(1 + x0*x1)",
        [(-1, 1), (-1, 1)],
        "hard",
        "需深层有理式",
    )
    add(
        "poly4mix",
        2,
        lambda X: X[:, 0] ** 4 - 2.0 * X[:, 1] ** 2 + X[:, 0],
        "x0**4 - 2*x1**2 + x0",
        [(-2, 2), (-2, 2)],
        "hard",
    )
    return specs


REGISTRY: list[FunctionSpec] = _register()


def get_spec(name: str) -> FunctionSpec:
    for s in REGISTRY:
        if s.name == name:
            return s
    raise KeyError(f"未知基准函数: {name}")


def all_names() -> list[str]:
    return [s.name for s in REGISTRY]


def sample(
    spec: FunctionSpec, n_train: int, n_test: int, seed: int, noise_std: float = 0.0
):
    """生成可复现 train/test 数据；噪声仅在 y 上加（DGP 层面，不污染 X）。"""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    X = np.column_stack([rng.uniform(lo, hi, size=n) for (lo, hi) in spec.domain])
    y = spec.f(X).astype(np.float64)
    if noise_std > 0.0:
        y = y + rng.normal(0.0, noise_std, size=n)
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = y[:n_train], y[n_train:]
    return Xtr, ytr, Xte, yte
