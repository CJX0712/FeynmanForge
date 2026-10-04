"""核心数据类型（core.types）。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SRResult:
    """单个后端在某函数上的符号回归结果。"""

    method: str
    expr_str: str
    complexity: int
    r2_train: float
    r2_test: float
    rmse_test: float
    recovered: bool
    n_vars: int
    elapsed_sec: float
    sympy_str: str = ""
    notes: str = ""


@dataclass
class BenchmarkRow:
    function: str
    n_vars: int
    seed: int
    method: str
    r2_test: float
    rmse_test: float
    complexity: int
    recovered: bool
    elapsed_sec: float


@dataclass
class MethodAgg:
    """某方法跨函数/种子的聚合指标。"""

    method: str
    n_runs: int
    median_r2: float = 0.0
    mean_r2: float = 0.0
    std_r2: float = 0.0
    recovery_rate: float = 0.0
    mean_complexity: float = 0.0
    mean_elapsed: float = 0.0
    # 只取每个函数最优 seed 后的恢复率（per-function best）
    best_recovery_rate: float = 0.0


@dataclass
class BenchmarkSummary:
    rows: list[BenchmarkRow] = field(default_factory=list)
    aggs: list[MethodAgg] = field(default_factory=list)
    gate: dict = field(default_factory=dict)


@dataclass
class FailureCase:
    function: str
    method: str
    predicted_expr: str
    true_expr: str
    r2_test: float
    reason: str
