"""旗舰与后端注册表（sr.feynmanforge）。

FeynmanForge = Age-Fitness Pareto 遗传编程（Schmidt & Lipson 2011），SOTA 级
符号回归引擎；纯 numpy，离线可跑，确定性可复现。
"""

from __future__ import annotations

import time

import numpy as np
import sympy as sp

from core.config import Config
from core.types import SRResult
from sr.gp import GPRegressor
from sr.tree import Node, affine_refine, optimize_constants


def _safe_sympy(tree: Node, n_vars: int) -> str:
    """把表达式树转成可读 sympy 串（仅展示用）。

    刻意不调用 sp.simplify：对深层嵌套表达式（如高斯 2D 的 exp/cos 组合），
    sympy 化简开销指数级，会拖垮整轮拟合。恢复判定以 R2 为准，符号串只用于
    人类阅读，故直接 str(构造式) 即可，既快又确定。
    """
    try:
        syms = sp.symbols(f"x0:{n_vars}")
        expr = tree.to_sympy(syms)
        return str(expr)
    except Exception:  # noqa: BLE001
        return tree.to_string()


def _r2(pred, y):
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def _build_result(
    method: str,
    tree: Node,
    Xtr,
    ytr,
    Xte,
    yte,
    n_vars: int,
    cfg: Config,
    elapsed: float,
) -> SRResult:
    tree_out = tree
    # 常量精修：把“结构正确、常量略偏”的近似推到 R2≈1.0
    if cfg.affine_refine:
        tree_out, _, _ = affine_refine(tree, Xtr, ytr)
        tree_out, _ = optimize_constants(tree_out, Xtr, ytr, max_nfev=200)
    else:
        tree_out, _ = optimize_constants(tree_out, Xtr, ytr, max_nfev=200)
    pred_tr = tree_out.eval(Xtr)
    r2_tr = _r2(pred_tr, ytr)
    if Xte is not None and yte is not None:
        pred_te = tree_out.eval(Xte)
        r2_te = _r2(pred_te, yte)
        rmse_te = float(np.sqrt(np.mean((yte - pred_te) ** 2)))
    else:
        r2_te, rmse_te = r2_tr, float(np.sqrt(np.mean((ytr - pred_tr) ** 2)))
    recovered = bool(r2_te >= cfg.recovery_r2)
    return SRResult(
        method=method,
        expr_str=tree_out.to_string(),
        complexity=tree_out.size(),
        r2_train=r2_tr,
        r2_test=r2_te,
        rmse_test=rmse_te,
        recovered=recovered,
        n_vars=n_vars,
        elapsed_sec=elapsed,
        sympy_str=_safe_sympy(tree_out, n_vars),
        notes="affine_refine=True" if cfg.affine_refine else "",
    )


class FeynmanForgeBackend:
    name = "FeynmanForge"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def available(self) -> bool:
        return True

    def fit(
        self, X: np.ndarray, y: np.ndarray, n_vars: int, Xte=None, yte=None
    ) -> SRResult:
        t0 = time.perf_counter()
        gp = GPRegressor(self.cfg, age_fitness=True)
        tree, _ = gp.fit(X, y, n_vars)
        return _build_result(
            self.name, tree, X, y, Xte, yte, n_vars, self.cfg, time.perf_counter() - t0
        )


class EPSNetBackend:
    name = "EPSNet"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def available(self) -> bool:
        return True

    def fit(
        self, X: np.ndarray, y: np.ndarray, n_vars: int, Xte=None, yte=None
    ) -> SRResult:
        from .eps_net import fit_eps_net

        te = Xte if Xte is not None else X
        yt = yte if yte is not None else y
        res = fit_eps_net(X, y, te, yt, n_vars, self.cfg.recovery_r2)
        return res


def build_backends(cfg: Config):
    """返回参与基准的 backend 列表（GP 旗舰 + 三个基线 + EPS 兜底）。"""
    return [
        FeynmanForgeBackend(cfg),
        LinearBaselineShim(cfg),
        RandomForestShim(cfg),
        NaiveGPShim(cfg),
        EPSNetBackend(cfg),
    ]


# ---- Shim：让基线类兼容 SRBackend 契约（携带 cfg）------------------------
from .baselines import (
    LinearBaseline,
    NaiveGPBaseline,
    RandomForestBaseline,
)


class LinearBaselineShim:
    name = "Linear"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._b = LinearBaseline()

    def available(self):
        return self._b.available()

    def fit(self, X, y, n_vars, Xte=None, yte=None):
        return self._b.fit(X, y, n_vars, self.cfg, Xte, yte)


class RandomForestShim:
    name = "RandomForest"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._b = RandomForestBaseline()

    def available(self):
        return self._b.available()

    def fit(self, X, y, n_vars, Xte=None, yte=None):
        res = self._b.fit(X, y, n_vars, self.cfg)
        if Xte is not None and yte is not None and self._b.available():
            from sklearn.ensemble import RandomForestRegressor

            m = RandomForestRegressor(
                n_estimators=120, random_state=int(self.cfg.seed), n_jobs=1
            )
            m.fit(X, y)
            pred = m.predict(Xte)
            res.r2_test = _r2(pred, yte)
            res.rmse_test = float(np.sqrt(np.mean((yte - pred) ** 2)))
            res.recovered = False
        return res


class NaiveGPShim:
    name = "NaiveGP"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._b = NaiveGPBaseline()

    def available(self):
        return self._b.available()

    def fit(self, X, y, n_vars, Xte=None, yte=None):
        return self._b.fit(X, y, n_vars, self.cfg, Xte, yte)
