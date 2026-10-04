# 踩坑记录（Pitfalls）

> 本次交付 FeynmanForge 过程中暴露的真实坑与修复，供后续 *Forge 系统复用。

### P1 · `sp.simplify` 对深层嵌套表达式开销指数级
- **现象**：`gauss2d` 单轮拟合耗时 150s+，且 R² 正常（~0.9994）。
- **根因**：`_safe_sympy` 对 42 节点嵌套 `cos/exp` 树调用 `sympy.simplify`，化简复杂度指数级。
- **修复**：符号串仅用于展示，直接 `str(构造式)`，不调用 simplify。恢复判定以 R² 为准。
- **收益**：gauss2d 150s → 4s，全量基准从“卡死”变为 ~9min。

### P2 · 无界 `least_squares` 漂常量引发溢出与发散
- **现象**：偶发 `overflow encountered in square`，单轮拟合久。
- **根因**：`optimize_constants` 未对常量设界，LM 把系数漂到 ~1e155，预测平方溢出为 inf，且优化器发散。
- **修复**：常量约束 `±30` 有界 + 残差裁剪（非有限/超大值裁剪到安全区间）+ `xtol/ftol=1e-10`。

### P3 · `affine_refine` 树近似常数时产生天文缩放系数
- **现象**：仿射精修后预测出现极大值。
- **根因**：树 eval 方差近零时，岭回归给出巨大 `a`。
- **修复**：方差 `<1e-9` 直接返回原树；`a` 裁剪到 `±1e6`。

### P4 · 基线 `lstsq` 维度不匹配
- **现象**：`LinearBaselineShim` 在测试集上 `LinAlgError: Incompatible dimensions`。
- **根因**：用训练标签 `y`（长度 n_train）对测试设计矩阵 `A`（行数 n_test）做 `lstsq`。
- **修复**：测试指标直接由 `LinearBaseline.fit(Xte,yte,...)` 用训练系数计算；统一契约携带 `Xte/yte`。

### P5 · 交叉算子元组解包顺序
- **现象**：`'int' object has no attribute 'copy'`。
- **根因**：`_pick_subtree` 返回 `(node, parent, ci)`，调用处写成 `_, _, sub2 = ...`（顺序错）。
- **修复**：`sub2, _, _ = _pick_subtree(t2, idx2)`。

### 通用经验
- 凡涉及 sympy 化简、无界数值优化、跨集矩阵运算，一律加确定性边界与有限性保护。
- 性能雪崩常源于“某个函数/某轮”的极端情形，需按函数逐条 profiling 而非整体估算。
