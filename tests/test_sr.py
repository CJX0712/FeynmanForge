"""FeynmanForge 测试套件（tests.test_sr）。

覆盖 SOP 交付门禁的关键项：
- 离线兜底（EPSNet）可用性与基本拟合；
- 旗舰对易函数精确恢复；
- 同 seed 逐位确定性；
- CLI demo 冒烟（收紧预算，快速）。
所有测试均使用极小预算，保证 CI 在分钟级完成。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace

from core.config import from_env
from data.generators import get_spec, sample
from sr.feynmanforge import (
    EPSNetBackend,
    FeynmanForgeBackend,
    build_backends,
)


def _tiny_cfg():
    cfg = from_env()
    return replace(cfg, pop_size=120, generations=30, max_depth=4, seeds=(7,))


# ---- 离线兜底 -------------------------------------------------------------
def test_offline_fallback_available():
    # EPSNet 仅依赖 numpy，不依赖网络/外部服务，离线必可用
    assert EPSNetBackend(from_env()).available() is True


def test_build_backends_all_available_offline():
    backends = build_backends(from_env())
    assert [b.name for b in backends] == [
        "FeynmanForge",
        "Linear",
        "RandomForest",
        "NaiveGP",
        "EPSNet",
    ]
    assert all(b.available() for b in backends)


def test_eps_net_fits_linear():
    cfg = from_env()
    spec = get_spec("lin1")
    Xtr, ytr, Xte, yte = sample(spec, cfg.n_train, cfg.n_test, 7, cfg.noise_std)
    res = EPSNetBackend(cfg).fit(Xtr, ytr, spec.n_vars, Xte, yte)
    assert res.r2_test > 0.9


# ---- 旗舰恢复能力 ---------------------------------------------------------
def test_feynmanforge_recovers_lin1():
    cfg = _tiny_cfg()
    spec = get_spec("lin1")
    Xtr, ytr, Xte, yte = sample(spec, cfg.n_train, cfg.n_test, 7, cfg.noise_std)
    res = FeynmanForgeBackend(cfg).fit(Xtr, ytr, spec.n_vars, Xte, yte)
    assert res.recovered is True
    assert res.r2_test >= 0.9999


def test_feynmanforge_recovers_sin1():
    cfg = replace(_tiny_cfg(), pop_size=200, generations=45, max_depth=5)
    spec = get_spec("sin1")
    Xtr, ytr, Xte, yte = sample(spec, cfg.n_train, cfg.n_test, 7, cfg.noise_std)
    res = FeynmanForgeBackend(cfg).fit(Xtr, ytr, spec.n_vars, Xte, yte)
    assert res.recovered is True


# ---- 确定性（同 seed 逐位一致）-------------------------------------------
def test_determinism_same_seed_identical_expression():
    cfg = _tiny_cfg()
    spec = get_spec("lin1")

    def run():
        Xtr, ytr, Xte, yte = sample(spec, cfg.n_train, cfg.n_test, 7, cfg.noise_std)
        return FeynmanForgeBackend(cfg).fit(Xtr, ytr, spec.n_vars, Xte, yte)

    a, b = run(), run()
    assert a.expr_str == b.expr_str
    assert a.r2_test == b.r2_test


# ---- CLI 冒烟（收紧预算，快速）-------------------------------------------
def test_cli_demo_smoke(tmp_path):
    env = dict(os.environ)
    env.update(
        {
            "FEYNMANFORGE_POP_SIZE": "40",
            "FEYNMANFORGE_GENERATIONS": "10",
            "FEYNMANFORGE_MAX_DEPTH": "4",
            "FEYNMANFORGE_SEED": "7",
        }
    )
    out = tmp_path / "bench_smoke.json"
    r = subprocess.run(
        [sys.executable, "cli.py", "demo", "--out", str(out)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    # 极小的冒烟预算下，门禁可为 PASS 或 WARN，但不应崩溃
    assert data["gate"]["verdict"]["overall"] in ("PASS", "WARN")
    assert "FeynmanForge" in data["agg"][0]["method"] or any(
        a["method"] == "FeynmanForge" for a in data["agg"]
    )
