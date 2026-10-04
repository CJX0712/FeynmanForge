"""端到端演示（examples.run_demo）。

落盘 benchmark.json 供复现与发布；打印门禁结论。
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.sr_pipeline import run


def main():
    out_path = os.path.join(ROOT, "benchmark.json")
    out = run(demo=True, out_path=out_path)
    print("\n=== 关键结论 ===")
    fa = next((a for a in out["agg"] if a["method"] == "FeynmanForge"), None)
    if fa:
        print(
            f"FeynmanForge  median R2={fa['median_r2']:.4f}  "
            f"恢复率={fa['recovery_rate'] * 100:.1f}%  "
            f"per-function 最优恢复率={fa['best_recovery_rate'] * 100:.1f}%"
        )
    print("门禁:", out["gate"]["verdict"].get("overall"))
    print("消融:", out["ablation"].get("conclusion"))


if __name__ == "__main__":
    main()
