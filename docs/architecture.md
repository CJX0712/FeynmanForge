# 架构设计（Architecture）

> FeynmanForge = Age-Fitness Pareto 遗传编程符号回归引擎。本文档描述模块职责、数据流与关键设计决策。

## 1. 整体数据流

```
样本 (X, y)
   │  data/generators.sample(spec, seed)   # 可复现，无泄漏
   ▼
FeynmanForgeBackend.fit(Xtr, ytr, n_vars, Xte, yte)
   │  1. GPRegressor.fit  →  最优表达式树
   │  2. affine_refine    →  a·t(x) + b 仿射精修
   │  3. optimize_constants → scipy least_squares 常量联合精修
   ▼
SRResult(expr_str, sympy_str, r2_test, rmse_test, recovered, complexity)
   │  eval/benchmark.run_benchmark  # 多函数 × 多 seed × 多后端
   ▼
benchmark.json (agg / gate / ablation / failures)
```

## 2. 核心模块

### core/
- **config.py**：`Config` 数据类 + `from_env()` 环境变量覆盖（`FEYNMANFORGE_*`）。集中所有超参并做范围校验。
- **seed.py**：`set_all(seed)` 统一标准库 `random`、numpy 默认 RNG 与 `PYTHONHASHSEED`，保证逐位可复现。
- **types.py**：`SRResult` / `BenchmarkRow` / `MethodAgg` / `BenchmarkSummary` / `FailureCase`。
- **errors.py**：`FeynmanForgeError` + 错误码（E100 配置 / E200 数据 / E300 引擎 / E400 评测 / E500 系统）。

### data/
- **generators.py**：21 个已知解析式基准（代数 / 超越 / AI-Feynman 风格物理量 / 困难有理高阶）。`sample()` 用独立 `np.random.default_rng(seed)`，与调用顺序无关。

### sr/（引擎）
- **grammar.py**：数值保护算子 `_pdiv` / `_pexp`(clip 30) / `_plog` / `_psqrt`(EPS=1e-9)，避免除零 / 溢出导致整轮崩溃。
- **tree.py**：
  - `Node.eval` 向量化求值（整批 X 一次计算）；
  - `make_random_tree` / `ramped_init`（half-and-half，深度多样性）；
  - `mutate` / `crossover` / `hoist`（抗膨胀）；
  - `affine_refine`：对树结果做岭回归 `a·t(x)+b`，提升常量命中；
  - `optimize_constants`：联合精修全部常量节点（`least_squares`，**有界 ±30**，防发散）。
- **gp.py**：`GPRegressor`
  - NSGA-II 快速非支配排序 `_fast_nds` + 拥挤度 `_crowding`；
  - 双目标 `(error, age)`：age 抑制膨胀个体长期霸占前沿 → 简洁解涌现；
  - `age_fitness=False` 时退化为朴素单目标 GP（消融对照）；
  - `_eval_tree` 含有限性检查与超大值裁剪，防止溢出雪崩。
- **eps_net.py**：Tier-1 离线兜底。基函数扩展（多项式 / 三角 / 指数 / 对数 / 根号 / 交叉项）+ 岭回归 + 迭代剪枝。纯 numpy，永远可用。
- **baselines.py**：`LinearBaseline`（OLS）、`RandomForestBaseline`（sklearn）、`NaiveGPBaseline`（单目标 GP）。
- **feynmanforge.py**：`FeynmanForgeBackend`（旗舰）、`EPSNetBackend`（兜底）、`build_backends`（注册 5 后端）、Shim 适配层（统一 `SRBackend` 契约，携带 `Config`）。

### eval/
- **metrics.py**：`aggregate(rows, method)` → 均值/中位/恢复率；`gate_summary(rows, flagship, baselines)` → 门禁判定。
- **benchmark.py**：`run_benchmark` 编排多函数×多 seed×多后端，产出聚合、门禁、消融（旗舰 vs NaiveGP）、失败案例。

## 3. 关键设计决策

| 决策 | 理由 |
|---|---|
| 双目标 Age-Fitness | 单目标 GP 易膨胀、陷入局部最优；age 目标促使简洁可恢复解 |
| 常量有界 `least_squares` | 无界优化会把系数漂到天文数字 → 溢出且发散（曾致 gauss2d 单轮 150s） |
| `_safe_sympy` 不做 `simplify` | 深层嵌套表达式 `sp.simplify` 开销指数级（gauss2d 150s → 4s） |
| 数值保护算子 | 保证任意数据下 eval 有限，避免整轮崩溃 |
| EPSNet 兜底 | 在任何环境（含 numpy-only）下系统都能给出结果，永不致命失败 |
| 显式 `set_all` | 满足 SOP “同 seed 逐位一致” 的硬交付要求 |

## 4. 复现

```bash
python cli.py demo --out benchmark.json          # ≤60s 演示
python cli.py benchmark --out benchmark.json      # 全量证据
pytest -q                                        # 含确定性回归
```
