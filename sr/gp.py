"""遗传编程引擎（sr.gp）。

旗舰采用 Age-Fitness Pareto（Schmidt & Lipson 2011）：
  - 双目标最小化 (error, age)，NSGA-II 快速非支配排序 + 拥挤度保多样性；
  - age 抑制“膨胀个体”在帕累托前沿长期霸屏，促进简洁可恢复表达式涌现。
  - 关掉 age 目标即为“朴素单目标 GP”（消融对照：error + λ·size 罚）。

所有随机经 numpy Generator，保证同 seed 逐位确定。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.config import Config
from sr.tree import Node, crossover, hoist, mutate, ramped_init

MAX_SIZE_HARD = 50  # 单棵树节点硬上限，超出直接判死，控评测成本


@dataclass
class _Ind:
    tree: Node
    error: float
    age: int
    size: int
    r2: float = 0.0


def _eval_tree(tree: Node, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    size = tree.size()
    if size > MAX_SIZE_HARD:
        return float("inf"), 0.0
    pred = tree.eval(X)
    if not np.all(np.isfinite(pred)):
        return float("inf"), 0.0
    # 防溢出：超大有限值（如 exp 链）裁剪到安全区间，避免平方时溢出为 inf，
    # 同时让病态个体保持“大误差”而被自然淘汰。
    pred = np.clip(pred, -1e9, 1e9)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    error = max(0.0, 1.0 - r2)
    return error, r2


def _fast_nds(objs: list[tuple[float, float]]) -> list[list[int]]:
    """NSGA-II 快速非支配排序，返回 fronts（索引列表）。"""
    n = len(objs)
    S = [[] for _ in range(n)]
    nd = [0] * n
    fronts: list[list[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            a, b = objs[p]
            c, d = objs[q]
            # p 支配 q
            if (a <= c and b <= d) and (a < c or b < d):
                S[p].append(q)
            elif (c <= a and d <= b) and (c < a or d < b):
                nd[p] += 1
        if nd[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                nd[q] -= 1
                if nd[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def _crowding(objs: list[tuple[float, float]], front: list[int]) -> list[float]:
    k = len(front)
    cd = [0.0] * k
    if k <= 2:
        return [float("inf")] * k
    for m in range(2):
        vals = sorted(range(k), key=lambda i: objs[front[i]][m])
        cd[vals[0]] = cd[vals[-1]] = float("inf")
        lo, hi = objs[front[vals[0]]][m], objs[front[vals[-1]]][m]
        rng = hi - lo
        for j in range(1, k - 1):
            prev, nxt = objs[front[vals[j - 1]]][m], objs[front[vals[j + 1]]][m]
            cd[vals[j]] += (nxt - prev) / rng if rng > 1e-12 else 0.0
    return cd


def _tournament(fronts, crowd, ranks, rng, k):
    pool = [int(rng.integers(0, len(ranks))) for _ in range(k)]
    best = pool[0]
    for c in pool[1:]:
        if ranks[c] < ranks[best] or (
            ranks[c] == ranks[best] and crowd[c] > crowd[best]
        ):
            best = c
    return best


class GPRegressor:
    def __init__(self, cfg: Config, age_fitness: bool = True):
        self.cfg = cfg
        self.age_fitness = age_fitness

    def fit(self, Xtr: np.ndarray, ytr: np.ndarray, n_vars: int) -> tuple[Node, float]:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        pop_size = cfg.pop_size
        pop: list[_Ind] = []
        for t in ramped_init(
            pop_size, cfg.max_depth_init, n_vars, rng, cfg.constants, cfg.const_range
        ):
            e, r2 = _eval_tree(t, Xtr, ytr)
            pop.append(_Ind(t, e, 0, t.size(), r2))

        for gen in range(cfg.generations):
            # 对当前种群算前沿 / rank / 拥挤度（供锦标赛使用）
            if self.age_fitness:
                objs = [(ind.error, float(ind.age)) for ind in pop]
                fronts = _fast_nds(objs)
                ranks = [0] * len(pop)
                crowd = [0.0] * len(pop)
                for fi, fr in enumerate(fronts):
                    cd = _crowding(objs, fr)
                    for rank_i, idx in enumerate(fr):
                        ranks[idx] = fi
                        crowd[idx] = cd[rank_i]
            # 生成子代
            offspring: list[_Ind] = []
            attempts = 0
            while len(offspring) < pop_size and attempts < pop_size * 4:
                attempts += 1
                if self.age_fitness:
                    i1 = _tournament(fronts, crowd, ranks, rng, cfg.tournament)
                    i2 = _tournament(fronts, crowd, ranks, rng, cfg.tournament)
                    p1, p2 = pop[i1], pop[i2]
                else:
                    p1 = min(pop, key=lambda x: x.error + cfg.parsimony * x.size)
                    p2 = min(pop, key=lambda x: x.error + cfg.parsimony * x.size)
                child_tree = p1.tree.copy()
                if rng.random() < cfg.crossover_rate:
                    child_tree = crossover(p1.tree, p2.tree, rng)
                if rng.random() < cfg.mutation_rate:
                    child_tree = mutate(
                        child_tree,
                        rng,
                        cfg.max_depth,
                        n_vars,
                        cfg.constants,
                        cfg.const_range,
                        p_leaf=0.3,
                    )
                if rng.random() < cfg.hoist_rate:
                    child_tree = hoist(child_tree, rng)
                e, r2 = _eval_tree(child_tree, Xtr, ytr)
                offspring.append(_Ind(child_tree, e, 0, child_tree.size(), r2))

            # 合并 + 选择
            combined = pop + offspring
            if self.age_fitness:
                objs = [(ind.error, float(ind.age)) for ind in combined]
                fronts = _fast_nds(objs)
                # 给每个个体赋 rank（front 序号）
                ranks = [0] * len(combined)
                crowd = [0.0] * len(combined)
                for fi, fr in enumerate(fronts):
                    cd = _crowding(objs, fr)
                    for rank_i, idx in enumerate(fr):
                        ranks[idx] = fi
                        crowd[idx] = cd[rank_i]
                # 选 N：按 rank 升序，同 rank 按拥挤度降序
                order = sorted(
                    range(len(combined)), key=lambda i: (ranks[i], -crowd[i])
                )
                survivors = [combined[i] for i in order[:pop_size]]
            else:
                # 朴素：单目标 error + 罚，按该值选 + 精英保最佳
                scored = sorted(
                    combined, key=lambda x: x.error + cfg.parsimony * x.size
                )
                survivors = scored[:pop_size]

            for s in survivors:
                s.age += 1
            pop = survivors

        # 终选：最小化 (error, size)
        best = min(pop, key=lambda x: (x.error, x.size))
        return best.tree, best.r2
