# Changelog

## v0.1.0 (2026-10-04)

首个可发布版本（S 级交付）。

### 新增
- **Age-Fitness Pareto 遗传编程引擎**：NSGA-II 快速非支配排序 + 拥挤度 + 双目标（误差↓ / 年龄↓），有效抑制表达式膨胀，促使简洁可恢复表达式涌现。
- **表达式树**：向量化 eval、ramped half-and-half 初始化、子树交叉 / 变异 / hoist、仿射精修、scipy `least_squares` 常量联合精修。
- **三大基线 + 离线兜底**：Linear（OLS）、RandomForest（sklearn）、NaiveGP（单目标 GP，消融对照）、EPSNet（纯 numpy 基函数扩展，Tier-1 兜底，离线必可用）。
- **21 个基准函数**：1/2/3 变量代数、超越、AI-Feynman 风格物理量（动能、引力势类比）及困难有理/高阶函数。
- **评测管道**：恢复率、中位 R²、消融（vs 朴素 GP）、诚实失败案例、门禁判定。
- **CLI**：`demo` / `benchmark` / `fit` 三个子命令。
- **确定性**：`core.seed.set_all` 统一随机源，同 seed 逐位可复现。
- **工程化**：pytest 测试、ruff 质量门禁、GitHub Actions CI（3.12/3.13 矩阵）、Dockerfile、Makefile、文档。

### 关键修复（开发期）
- 修复 `LinearBaselineShim` 在测试集上 `lstsq` 维度不匹配（`y` 误用为训练标签）。
- 修复 `gauss2d` 单轮拟合 150s 的性能雪崩：根因为 `_safe_sympy` 对深层嵌套表达式调用 `sp.simplify` 开销指数级 —— 改为直接 `str(构造式)`。
- 修复 `optimize_constants` 无界 `least_squares` 将常量漂移至天文数字引发溢出/发散：加入 `±30` 边界与残差裁剪。
- 修复 `affine_refine` 在树近似常数时产生巨大缩放系数：加入方差门槛与系数裁剪。

### 交付门禁（DoD）
- ✅ demo 端到端 ≤ 60s（实测 37s）
- ✅ ≥3 seed 均值±标准差（全量基准提供）
- ✅ 逐位确定性（pytest 回归）
- ✅ 离线兜底（EPSNet 零依赖可用）
- ✅ 消融 ≥1（Age-Fitness vs NaiveGP，+40pp 恢复率）
- ✅ 诚实失败案例 ≥3（全量基准产出）
- ✅ ruff + pytest + CI 全绿
