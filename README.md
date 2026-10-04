# FeynmanForge

> 世界级的符号回归（Symbolic Regression）系统 —— Age-Fitness Pareto 遗传编程，纯 numpy 实现，离线可跑，确定性可复现。

- **作者 / Author**：晨星（GitHub: [CJX0712](https://github.com/CJX0712)）
- **领域 / Domain**：AI · 自动科学发现 / 方程挖掘（Symbolic Regression）
- **核心算法**：Age-Fitness Pareto Genetic Programming（Schmidt & Lipson, 2011），NSGA-II 快速非支配排序 + 拥挤度保多样性
- **依赖**：`numpy` `scipy` `scikit-learn` `sympy`（均已锁定版本，跨平台 wheel，无需本地编译）
- **许可**：MIT

---

## 一句话定位

给定一组 `(X, y)` 样本，FeynmanForge 自动找出**人类可读、可解释**的解析表达式（如 `exp(-x0**2 - x1**2)`、`0.5*x0*x1**2`），而非黑箱模型。它在**恢复率**上显著优于朴素单目标 GP 与经典基线（线性/随机森林），并且**完全离线、逐位可复现**。

---

## 特性

| 特性 | 说明 |
|---|---|
| 🏆 SOTA 级引擎 | Age-Fitness Pareto GP，双目标（误差↓ + 年龄↓）抑制膨胀，促使简洁可恢复表达式涌现 |
| 🔌 离线兜底 | EPSNet（纯 numpy 基函数扩展 + 岭回归 + 剪枝）在任意环境下可用，保证系统永不崩溃 |
| 🎯 精确恢复 | affine 精修 + scipy `least_squares` 常量联合优化，将“结构正确、常量略偏”推到 R²≈1.0 |
| 🔁 逐位确定性 | 统一随机源（标准库 random / numpy / PYTHONHASHSEED），同 seed 两次运行核心指标一致 |
| ⚡ 零依赖推理 | 推理仅依赖 numpy；无需 GPU、无需联网 |
| 🧪 严格门禁 | 恢复率 / 中位 R² / 消融（vs 朴素 GP）/ 诚实失败案例，全部量化 |

---

## 安装

```bash
# 推荐：使用本仓库锁定的虚拟环境（Python 3.13）
python -m venv .venv && source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.lock.txt
```

> 依赖均为预编译 wheel（cp312 / cp313 的 win / linux / macOS），无需 C/C++ 编译工具链。

---

## 快速开始

```bash
# 1) 演示基准（收紧预算，端到端 ≤ 60s）：旗舰 vs 4 个基线 + 兜底
python cli.py demo --out benchmark.json

# 2) 全量基准（21 个基准函数 × 3 seed × 5 后端），产出 S 级证据
python cli.py benchmark --out benchmark.json

# 3) 单函数拟合，打印表达式与指标
python cli.py fit --func quad1 --seed 7
```

---

## 演示结果（demo，CPU ≤ 60s）

| 后端 | median R² | 恢复率 |
|---|---|---|
| **FeynmanForge（旗舰）** | **1.0** | **1.0**（5/5 函数 × 3 seed 全部精确恢复） |
| EPSNet（离线兜底） | 1.0 | 1.0 |
| NaiveGP（消融对照） | 1.0 | 0.6 |
| RandomForest | 0.9995 | 0.0 |
| Linear | 0.298 | 0.2 |

**消融结论**：Age-Fitness Pareto 恢复率相对朴素单目标 GP **提升 40.0 个百分点**。

（全量 21 函数基准的逐函数结果、消融与失败案例分析见 `benchmark.json` 与 `docs/model_card.md`。）

---

## 架构

```
FeynmanForge/
├── core/            # 配置 / 类型 / 错误码 / 全局确定性入口
│   ├── config.py    # Config + 环境变量覆盖（FEYNMANFORGE_*）
│   ├── seed.py      # set_all(seed)：统一随机源，保证可复现
│   ├── types.py     # SRResult / BenchmarkRow / FailureCase ...
│   └── errors.py    # FeynmanForgeError + 错误码 E100..E500
├── data/            # 21 个已知解析式基准（含 AI-Feynman 风格物理量）
│   └── generators.py# sample(spec, n_train, n_test, seed) — 可复现数据
├── sr/              # 符号回归引擎
│   ├── grammar.py   # 数值保护算子（_pdiv/_pexp/_plog/_psqrt）
│   ├── tree.py      # 表达式树：eval / mutate / crossover / hoist / 常量精修
│   ├── gp.py        # NSGA-II + Age-Fitness Pareto 主循环
│   ├── eps_net.py   # Tier-1 离线兜底（纯 numpy）
│   ├── baselines.py # Linear / RandomForest / NaiveGP 基线
│   └── feynmanforge.py # 旗舰后端 + 后端注册表 + 兜底编排
├── eval/            # 评测与聚合
│   ├── metrics.py   # aggregate / gate_summary
│   └── benchmark.py # run_benchmark：门禁 / 消融 / 失败案例
├── pipeline/        # 端到端管道（演示 / 全量）
│   └── sr_pipeline.py
├── tests/           # pytest：离线兜底 / 恢复 / 确定性 / CLI 冒烟
├── docs/            # architecture.md / model_card.md
├── cli.py           # demo / benchmark / fit 三个子命令
└── requirements.lock.txt
```

详细设计见 [`docs/architecture.md`](docs/architecture.md)；模型能力与局限见 [`docs/model_card.md`](docs/model_card.md)。

---

## 随机种子与确定性

```python
from core.seed import set_all

set_all(7)  # 一次性设齐 random / numpy / PYTHONHASHSEED
```

同一 `(函数, seed)` 配置下，两次运行得到的表达式字符串与 R² 逐位一致。CI 中 `pytest` 包含确定性回归测试。

---

## 引用与方法谱系

- Schmidt, M. & Lipson, H. (2011). *Age-Fitness Pareto Optimization*. Genetic Programming Theory and Practice.
- Udrescu, S. & Tegmark, M. (2020). *AI Feynman: A Physics-Inspired Method for Symbolic Regression*.
- La Cava, W. et al. (2021). *SRBench: A Living Benchmark for Symbolic Regression*.
- Koza, J. (1992). *Genetic Programming*.

---

© 2026 晨星 (CJX0712). MIT License.
