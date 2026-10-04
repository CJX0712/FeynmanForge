"""评测指标与聚合（eval.metrics）。"""

from __future__ import annotations

import numpy as np

from core.types import BenchmarkRow, MethodAgg


def aggregate(rows: list[BenchmarkRow], method: str) -> MethodAgg:
    sub = [r for r in rows if r.method == method]
    if not sub:
        return MethodAgg(method=method, n_runs=0)
    r2 = np.array([r.r2_test for r in sub], dtype=float)
    rec = np.array([1.0 if r.recovered else 0.0 for r in sub], dtype=float)
    # per-function best（每个函数取最优 seed 的恢复标志）
    funcs = sorted({r.function for r in sub})
    best_rec = []
    for fn in funcs:
        fr = [1.0 if r.recovered else 0.0 for r in sub if r.function == fn]
        best_rec.append(max(fr))
    return MethodAgg(
        method=method,
        n_runs=len(sub),
        median_r2=float(np.median(r2)),
        mean_r2=float(np.mean(r2)),
        std_r2=float(np.std(r2)),
        recovery_rate=float(np.mean(rec)),
        mean_complexity=float(np.mean([r.complexity for r in sub])),
        mean_elapsed=float(np.mean([r.elapsed_sec for r in sub])),
        best_recovery_rate=float(np.mean(best_rec)),
    )


def gate_summary(
    summary_rows: list[BenchmarkRow], flagship: str, baselines: list[str]
) -> dict:
    aggs = {
        a.method: a
        for a in [aggregate(summary_rows, m) for m in ([flagship] + baselines)]
    }
    gate = {
        "flagship": flagship,
        "median_r2": {},
        "recovery_rate": {},
        "best_recovery_rate": {},
        "verdict": {},
    }
    fa = aggs.get(flagship)
    if not fa or fa.n_runs == 0:
        gate["verdict"]["overall"] = "NO_DATA"
        return gate
    for metric in ("median_r2", "recovery_rate", "best_recovery_rate"):
        d = {}
        for m in [flagship] + baselines:
            a = aggs.get(m)
            d[m] = round(getattr(a, metric), 6) if a else None
        gate[metric] = d
    # 判定：旗舰恢复率高于所有基线，且 median_r2 不低于任一基线
    fa_rec = fa.recovery_rate
    fa_r2 = fa.median_r2
    beats_rec = all(
        (aggs[b].recovery_rate < fa_rec + 1e-9) for b in baselines if aggs.get(b)
    )
    not_worse_r2 = all(
        (aggs[b].median_r2 <= fa_r2 + 1e-9) for b in baselines if aggs.get(b)
    )
    gate["verdict"]["beats_baselines_recovery"] = bool(beats_rec)
    gate["verdict"]["not_worse_r2"] = bool(not_worse_r2)
    gate["verdict"]["overall"] = "PASS" if (beats_rec and not_worse_r2) else "REVIEW"
    return gate
