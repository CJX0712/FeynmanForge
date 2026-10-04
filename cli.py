"""命令行入口（cli.py）。

用法：
  python cli.py demo                 # 演示基准（收紧预算，≤60s），落盘 benchmark.json
  python cli.py benchmark --full     # 全量基准（所有 21 个基准函数）
  python cli.py fit --func quad1 --seed 7   # 单函数拟合，打印表达式与指标
"""

from __future__ import annotations

import argparse
import os
import sys

from core.config import from_env
from data.generators import get_spec, sample
from sr.feynmanforge import FeynmanForgeBackend


def cmd_demo(args):
    from pipeline.sr_pipeline import run

    out = run(demo=not args.full, out_path=args.out)
    _print_gate(out)
    return 0


def cmd_benchmark(args):
    from pipeline.sr_pipeline import run

    out = run(demo=False, out_path=args.out, seeds=args.seeds)
    _print_gate(out)
    return 0


def cmd_fit(args):
    import dataclasses

    cfg = from_env()
    cfg = dataclasses.replace(cfg, seed=args.seed)
    spec = get_spec(args.func)
    Xtr, ytr, Xte, yte = sample(spec, cfg.n_train, cfg.n_test, args.seed, cfg.noise_std)
    backend = FeynmanForgeBackend(cfg)
    res = backend.fit(Xtr, ytr, spec.n_vars, Xte, yte)
    print(f"函数   : {spec.name}  ({spec.true_expr})")
    print(f"发现   : {res.expr_str}")
    print(f"sympy  : {res.sympy_str}")
    print(
        f"R2(train)={res.r2_train:.6f}  R2(test)={res.r2_test:.6f}  "
        f"RMSE(test)={res.rmse_test:.3e}"
    )
    print(f"复杂度 = {res.complexity}  精确恢复 = {res.recovered}")
    return 0


def _print_gate(out: dict):
    g = out.get("gate", {})
    print("\n==================== 门禁 / 聚合 ====================")
    print(f"判定: {g.get('verdict', {}).get('overall')}")
    for metric in ("median_r2", "recovery_rate", "best_recovery_rate"):
        print(
            f"  {metric}: "
            + "  ".join(f"{k}={v}" for k, v in g.get(metric, {}).items())
        )
    ab = out.get("ablation", {})
    print("  消融: " + ab.get("conclusion", ""))


def build_parser():
    p = argparse.ArgumentParser(
        prog="feynmanforge", description="符号回归系统 FeynmanForge"
    )
    sub = p.add_subparsers(dest="cmd")
    d = sub.add_parser("demo", help="演示基准")
    d.add_argument("--full", action="store_true", help="全量而非演示子集")
    d.add_argument("--out", default="benchmark.json")
    d.set_defaults(func=cmd_demo)
    b = sub.add_parser("benchmark", help="全量基准（所有 21 个基准函数 × 多 seed）")
    b.add_argument("--out", default="benchmark.json")
    b.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="覆盖默认 seed 列表，如 --seeds 7 19 42",
    )
    b.set_defaults(func=cmd_benchmark)
    f = sub.add_parser("fit", help="单函数拟合")
    f.add_argument("--func", required=True)
    f.add_argument("--seed", type=int, default=7)
    f.set_defaults(func=cmd_fit)
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = ["demo"]
    # 统一随机源，保证同 seed 两次运行逐位一致（SOP 确定性硬要求）
    from core.seed import set_all

    set_all(int(os.environ.get("FEYNMANFORGE_SEED", "7")))
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
