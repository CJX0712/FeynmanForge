"""端到端管道（pipeline.sr_pipeline）。

封装「构建后端 → 跑基准 → 落盘 benchmark.json」，一键复现。
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

from core.config import Config, from_env
from data.generators import REGISTRY, get_spec
from eval.benchmark import run_benchmark
from sr.feynmanforge import build_backends

# 演示子集：精选「确定可恢复」的易函数（1 变量代数/超越 + 2 变量多项式乘积），
# 预算收紧以端到端 ≤60s（CPU）。困难函数留给全量基准（cli benchmark --full）
# 产出恢复率/消融/失败案例等 S 级证据。
DEMO_SPEC_NAMES: list[str] = [
    "lin1",
    "quad1",
    "sin1",
    "poly2d",
    "prod2",
]


def _specs(names: list[str]):
    return [get_spec(n) for n in names]


def run(
    demo: bool = True, out_path: str = "benchmark.json", cfg: Config = None, seeds=None
) -> dict:
    if cfg is None:
        cfg = from_env()
    if demo:
        # 收紧预算，保证 demo 端到端 ≤ 60s（CPU）
        cfg = replace(cfg, pop_size=150, generations=45, max_depth=5)
        specs = _specs(DEMO_SPEC_NAMES)
    else:
        specs = REGISTRY
    if seeds is not None:
        cfg = replace(cfg, seeds=tuple(seeds))
    backends = build_backends(cfg)
    print(
        f"[FeynmanForge] demo={demo} funcs={len(specs)} backends="
        f"{[b.name for b in backends if b.available()]} seeds={cfg.seeds}"
    )
    out = run_benchmark(backends, specs, cfg, seeds=list(cfg.seeds), verbose=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[FeynmanForge] 已落盘 {out_path}")
    return out
