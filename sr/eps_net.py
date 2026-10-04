"""Tier-1 离线兜底：基函数展开 + 岭回归 + 迭代剪枝（sr.eps_net）。

纯 numpy，零重型依赖、零下载、确定性可复现。作为 SOTA(GP) 不可用时的兜底，
也可作为对照基线。对多项式类函数可精确恢复；对超越函数提供合理近似。
"""

from __future__ import annotations

import numpy as np

from core.types import SRResult


def _basis(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    n = X.shape[1]
    cols: list[np.ndarray] = [np.ones(X.shape[0])]
    names: list[str] = ["1"]
    for i in range(n):
        v = X[:, i]
        cols += [
            v,
            v**2,
            v**3,
            np.sin(v),
            np.cos(v),
            np.exp(-(v**2)),
            np.log(np.abs(v) + 0.1),
            np.sqrt(np.abs(v)),
        ]
        names += [
            f"x{i}",
            f"x{i}**2",
            f"x{i}**3",
            f"sin(x{i})",
            f"cos(x{i})",
            f"exp(-x{i}**2)",
            f"log(|x{i}|+0.1)",
            f"sqrt(|x{i}|)",
        ]
    for i in range(n):
        for j in range(i + 1, n):
            cols.append(X[:, i] * X[:, j])
            names.append(f"x{i}*x{j}")
    return np.column_stack(cols), names


def _fit(X, y):
    Phi, names = _basis(X)
    # 岭回归
    A = Phi.T @ Phi + 1e-8 * np.eye(Phi.shape[1])
    b = Phi.T @ y
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    full_r2 = _r2(Phi @ coef, y)
    # 迭代剪枝：按 |coef| 升序逐步剔除，直到 R2 跌出 (full_r2 - 1e-4)
    order = np.argsort(np.abs(coef))
    keep = list(range(len(coef)))
    for idx in order:
        if len(keep) <= 1:
            break
        trial = [k for k in keep if k != idx]
        r2_t = _r2(Phi[:, trial] @ coef[trial], y)
        if r2_t >= full_r2 - 1e-4:
            keep = trial
        else:
            break
    keep.sort()
    expr = " + ".join(f"{coef[k]:.6g}*{names[k]}" for k in keep if abs(coef[k]) > 1e-9)
    if not expr:
        expr = "0"
    return coef, keep, names, expr, full_r2


def _r2(pred, y):
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def fit_eps_net(Xtr, ytr, Xte, yte, n_vars: int, recovery_r2: float) -> SRResult:
    coef, keep, _, expr, _ = _fit(Xtr, ytr)
    Phi_te, _ = _basis(Xte)
    pred_te = Phi_te[:, keep] @ coef[keep]
    r2_te = _r2(pred_te, yte)
    Phi_tr, _ = _basis(Xtr)
    pred_tr = Phi_tr[:, keep] @ coef[keep]
    r2_tr = _r2(pred_tr, ytr)
    rmse_te = float(np.sqrt(np.mean((yte - pred_te) ** 2)))
    recovered = bool(r2_te >= recovery_r2)
    return SRResult(
        method="EPSNet",
        expr_str=expr,
        complexity=len(keep),
        r2_train=r2_tr,
        r2_test=r2_te,
        rmse_test=rmse_te,
        recovered=recovered,
        n_vars=n_vars,
        elapsed_sec=0.0,
        sympy_str=expr,
        notes="Tier-1 离线兜底 / 基线",
    )
