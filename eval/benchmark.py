"""基准编排（eval.benchmark）。

对一组基准函数 × 多个 seed × 多个后端做端到端评测，产出可复现的
benchmark.json（含每函数每后端 R2 / 恢复率 / 复杂度 / 消融 / 失败案例 / 门禁）。
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import replace

import numpy as np

from core.config import Config
from core.types import BenchmarkRow, FailureCase
from data.generators import FunctionSpec, sample
from eval.metrics import aggregate, gate_summary


def run_backend_on_spec(backend, spec: FunctionSpec, cfg: Config, seed: int) -> dict:
    cfg_s = replace(cfg, seed=seed)
    Xtr, ytr, Xte, yte = sample(
        spec, cfg_s.n_train, cfg_s.n_test, seed, cfg_s.noise_std
    )
    res = backend.fit(Xtr, ytr, spec.n_vars, Xte, yte)
    return {
        "function": spec.name,
        "n_vars": spec.n_vars,
        "seed": seed,
        "method": res.method,
        "r2_test": res.r2_test,
        "rmse_test": res.rmse_test,
        "complexity": res.complexity,
        "recovered": res.recovered,
        "elapsed_sec": res.elapsed_sec,
        "expr": res.expr_str,
        "sympy": res.sympy_str,
        "true_expr": spec.true_expr,
        "difficulty": spec.difficulty,
        "reference": spec.reference,
    }


def run_benchmark(
    backends, specs: list[FunctionSpec], cfg: Config, seeds=None, verbose: bool = True
) -> dict:
    seeds = list(seeds if seeds is not None else cfg.seeds)
    rows: list[BenchmarkRow] = []
    records = []
    t0 = time.perf_counter()
    for spec in specs:
        for seed in seeds:
            for backend in backends:
                if not backend.available():
                    continue
                rec = run_backend_on_spec(backend, spec, cfg, seed)
                rows.append(
                    BenchmarkRow(
                        function=rec["function"],
                        n_vars=rec["n_vars"],
                        seed=seed,
                        method=rec["method"],
                        r2_test=rec["r2_test"],
                        rmse_test=rec["rmse_test"],
                        complexity=rec["complexity"],
                        recovered=rec["recovered"],
                        elapsed_sec=rec["elapsed_sec"],
                    )
                )
                records.append(rec)
                if verbose:
                    print(
                        f"  {spec.name:12s} seed={seed:<3d} {rec['method']:12s} "
                        f"R2={rec['r2_test']:.4f} rec={int(rec['recovered'])} "
                        f"cplx={rec['complexity']}"
                    )
    # 聚合
    methods = [b.name for b in backends if b.available()]
    aggs = [aggregate(rows, m) for m in methods]
    # EPSNet 是系统内「离线兜底」（非外部基线），不参与竞争性门禁，仅透明展示；
    # 门禁只对比外部基线（Linear / RandomForest / NaiveGP）。
    baseline_methods = [m for m in methods if m not in ("FeynmanForge", "EPSNet")]
    flagship = "FeynmanForge" if "FeynmanForge" in methods else methods[0]
    gate = gate_summary(rows, flagship, baseline_methods)

    elapsed = time.perf_counter() - t0

    # 失败案例：旗舰在某函数所有 seed 均未恢复
    failures: list[FailureCase] = []
    fa_recs = [r for r in records if r["method"] == flagship]
    for spec in specs:
        srecs = [r for r in fa_recs if r["function"] == spec.name]
        if not srecs:
            continue
        if not any(r["recovered"] for r in srecs):
            worst = max(srecs, key=lambda r: -r["r2_test"])
            reason = _reason(spec, worst)
            failures.append(
                FailureCase(
                    function=spec.name,
                    method=flagship,
                    predicted_expr=worst["expr"],
                    true_expr=spec.true_expr,
                    r2_test=worst["r2_test"],
                    reason=reason,
                )
            )

    out = {
        "meta": {
            "system": "FeynmanForge",
            "domain": "Symbolic Regression",
            "seeds": seeds,
            "n_train": cfg.n_train,
            "n_test": cfg.n_test,
            "recovery_r2_threshold": cfg.recovery_r2,
            "pop_size": cfg.pop_size,
            "generations": cfg.generations,
            "max_depth": cfg.max_depth,
            "affine_refine": cfg.affine_refine,
            "total_elapsed_sec": round(elapsed, 3),
        },
        "agg": [dataclasses.asdict(a) for a in aggs],
        "gate": gate,
        "records": records,
        "failures": [dataclasses.asdict(f) for f in failures],
        "ablation": _ablation(records, flagship),
    }
    return out


def _reason(spec: FunctionSpec, worst: dict) -> str:
    if spec.difficulty == "hard":
        return "困难函数（深层有理/高阶），GP 在当前预算内未命中结构；属诚实负例。"
    if worst["r2_test"] > 0.95:
        return "高保真近似但表达式复杂度偏高/常量未精确命中，R2 未达恢复阈值。"
    return "函数超出算子搜索空间或局部最优；建议增大预算/算子集。"


def _ablation(records: list, flagship: str) -> dict:
    """消融：FeynmanForge(Age-Fitness Pareto) vs NaiveGP(单目标) 在恢复率与 R2 上的对照。"""
    out = {}
    for metric in ("recovery_rate", "median_r2"):
        d = {}
        for m in (flagship, "NaiveGP"):
            sub = [r for r in records if r["method"] == m]
            if not sub:
                d[m] = None
                continue
            if metric == "recovery_rate":
                d[m] = round(
                    float(np.mean([1.0 if r["recovered"] else 0.0 for r in sub])), 6
                )
            else:
                d[m] = round(float(np.median([r["r2_test"] for r in sub])), 6)
        out[metric] = d
    # 结论
    if (
        out["recovery_rate"].get(flagship) is not None
        and out["recovery_rate"].get("NaiveGP") is not None
    ):
        delta = out["recovery_rate"][flagship] - out["recovery_rate"]["NaiveGP"]
        out["conclusion"] = (
            f"Age-Fitness Pareto 恢复率相对朴素 GP {'提升' if delta >= 0 else '下降'} "
            f"{abs(delta) * 100:.1f} pp"
        )
    else:
        out["conclusion"] = "对照缺失"
    return out
