"""对照基线（sr.baselines）。

- LinearBaseline：最小二乘线性模型（仅捕捉线性结构）。
- RandomForestBaseline：sklearn 随机森林（强非线性但非符号/非简洁）。
- NaiveGPBaseline：朴素单目标 GP（error + λ·size 罚），用于消融对照
  “Age-Fitness Pareto 是否带来增益”。
均实现 SRBackend 契约。
"""

from __future__ import annotations

import time

import numpy as np

from core.config import Config
from core.types import SRResult
from sr.gp import GPRegressor


class LinearBaseline:
    name = "Linear"

    def available(self) -> bool:
        return True

    def fit(
        self, X: np.ndarray, y: np.ndarray, n_vars: int, cfg: Config, Xte=None, yte=None
    ) -> SRResult:
        t0 = time.perf_counter()
        ones = np.ones((X.shape[0], 1))
        A = np.hstack([ones, X])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2_tr = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        r2_te, rmse_te = r2_tr, float("nan")
        if Xte is not None and yte is not None:
            Ate = np.hstack([np.ones((Xte.shape[0], 1)), Xte])
            pred_te = Ate @ coef
            ss_res_te = float(np.sum((yte - pred_te) ** 2))
            ss_tot_te = float(np.sum((yte - yte.mean()) ** 2))
            r2_te = 1.0 - ss_res_te / ss_tot_te if ss_tot_te > 1e-12 else 0.0
            rmse_te = float(np.sqrt(np.mean((yte - pred_te) ** 2)))
        expr = " + ".join(
            ([f"{coef[0]:.6g}"] if n_vars else [])
            + [f"{coef[i + 1]:.6g}*x{i}" for i in range(n_vars)]
        )
        return SRResult(
            method=self.name,
            expr_str=expr or "0",
            complexity=n_vars + 1,
            r2_train=r2_tr,
            r2_test=r2_te,
            rmse_test=rmse_te,
            recovered=bool(r2_te >= cfg.recovery_r2),
            n_vars=n_vars,
            elapsed_sec=time.perf_counter() - t0,
            sympy_str=expr or "0",
            notes="OLS 线性基线",
        )


class RandomForestBaseline:
    name = "RandomForest"

    def __init__(self) -> None:
        self._sk = None
        try:
            from sklearn.ensemble import RandomForestRegressor

            self._sk = RandomForestRegressor
        except ImportError:
            self._sk = None

    def available(self) -> bool:
        return self._sk is not None

    def fit(self, X: np.ndarray, y: np.ndarray, n_vars: int, cfg: Config) -> SRResult:
        t0 = time.perf_counter()
        if self._sk is None:
            raise RuntimeError("sklearn 不可用，无法运行 RandomForest 基线")
        model = self._sk(n_estimators=120, random_state=int(cfg.seed), n_jobs=1)
        model.fit(X, y)
        pred_tr = model.predict(X)
        ss_res = float(np.sum((y - pred_tr) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2_tr = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        leaves = int(getattr(model, "n_estimators", 1))
        return SRResult(
            method=self.name,
            expr_str="RandomForest(estimators=120)",
            complexity=leaves,
            r2_train=r2_tr,
            r2_test=r2_tr,
            rmse_test=float("nan"),
            recovered=False,
            n_vars=n_vars,
            elapsed_sec=time.perf_counter() - t0,
            sympy_str="(ensemble, non-parametric)",
            notes="sklearn RF 基线（非符号）",
        )


class NaiveGPBaseline:
    name = "NaiveGP"

    def available(self) -> bool:
        return True

    def fit(
        self, X: np.ndarray, y: np.ndarray, n_vars: int, cfg: Config, Xte=None, yte=None
    ) -> SRResult:
        from .feynmanforge import _build_result

        t0 = time.perf_counter()
        gp = GPRegressor(cfg, age_fitness=False)
        tree, _ = gp.fit(X, y, n_vars)
        res = _build_result(
            "NaiveGP", tree, X, y, Xte, yte, n_vars, cfg, time.perf_counter() - t0
        )
        return res
