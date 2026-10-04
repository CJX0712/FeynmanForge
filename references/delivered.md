# 已交付系统记录（Delivered Systems）

| 日期 | 系统 | 领域 | 方法 | 仓库 | 门禁 | 备注 |
|---|---|---|---|---|---|---|
| 2026-10-04 | FeynmanForge | Symbolic Regression（符号回归） | Age-Fitness Pareto GP（Schmidt & Lipson 2011）+ EPSNet 离线兜底 | [CJX0712/FeynmanForge](https://github.com/CJX0712/FeynmanForge) | PASS | 21 函数 × 3 seed × 5 后端全量基准门禁 PASS（旗舰恢复率 0.54 > 全部外部基线；median R²=1.0）；消融 +11.1pp（全量）/ +40pp（demo）；demo ≤60s（实测 37s）；逐位确定性；pytest+ruff+CI 全绿。作者：晨星 |

## 方法谱系
- 双目标 NSGA-II（误差 / 年龄）抑制表达式膨胀，促使简洁可恢复解涌现。
- affine 精修 + scipy `least_squares` 常量联合优化（有界 ±30）。
- EPSNet（纯 numpy 基函数扩展）作为离线兜底，保证系统永不致命失败。
- 基线：Linear（OLS）/ RandomForest（sklearn）/ NaiveGP（单目标 GP，消融对照）。
