"""表达式树（sr.tree）。

向量化求值（整批 X 一次计算）；复杂度 = 节点数；支持 sympy 转换用于符号级
校验。所有随机操作接收 numpy Generator，保证全局确定性。
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from .grammar import BINARY, UNARY, apply_op, str_fmt


class Node:
    __slots__ = ("children", "op", "value", "var_idx")

    def __init__(
        self,
        op: str,
        children: list[Node] | None = None,
        value: float = 0.0,
        var_idx: int = 0,
    ):
        self.op = op  # 'var' | 'const' | 算子名
        self.children = children if children is not None else []
        self.value = value
        self.var_idx = var_idx

    # ---- 构造辅助 -------------------------------------------------------
    @staticmethod
    def var(idx: int) -> Node:
        return Node("var", [], 0.0, idx)

    @staticmethod
    def const(v: float) -> Node:
        return Node("const", [], float(v), 0)

    @staticmethod
    def func(op: str, *kids: Node) -> Node:
        return Node(op, list(kids))

    # ---- 求值 -----------------------------------------------------------
    def eval(self, X: np.ndarray) -> np.ndarray:
        op = self.op
        if op == "var":
            return X[:, self.var_idx].astype(np.float64)
        if op == "const":
            return np.full(X.shape[0], self.value, dtype=np.float64)
        if op in UNARY:
            return apply_op(op, self.children[0].eval(X))
        return apply_op(op, self.children[0].eval(X), self.children[1].eval(X))

    # ---- 复杂度 ---------------------------------------------------------
    def size(self) -> int:
        s = 1
        for c in self.children:
            s += c.size()
        return s

    # ---- 复制 -----------------------------------------------------------
    def copy(self) -> Node:
        return Node(
            self.op, [c.copy() for c in self.children], self.value, self.var_idx
        )

    # ---- 字符串 ---------------------------------------------------------
    def to_string(self) -> str:
        op = self.op
        if op == "var":
            return f"x{self.var_idx}"
        if op == "const":
            v = self.value
            if v == int(v) and abs(v) < 1e6:
                return str(int(v))
            return repr(float(v))
        args = [c.to_string() for c in self.children]
        return str_fmt(op).format(*args)

    # ---- sympy ----------------------------------------------------------
    def to_sympy(self, var_syms, sp_module=sp) -> sp.Expr:
        op = self.op
        if op == "var":
            return var_syms[self.var_idx]
        if op == "const":
            return sp_module.Float(self.value)
        kids = [c.to_sympy(var_syms, sp_module) for c in self.children]
        if op == "sin":
            return sp_module.sin(kids[0])
        if op == "cos":
            return sp_module.cos(kids[0])
        if op == "exp":
            return sp_module.exp(kids[0])
        if op == "log":
            return sp_module.log(kids[0])
        if op == "sqrt":
            return sp_module.sqrt(kids[0])
        if op == "square":
            return sp_module.Pow(kids[0], 2)
        if op == "neg":
            return -kids[0]
        if op == "add":
            return kids[0] + kids[1]
        if op == "sub":
            return kids[0] - kids[1]
        if op == "mul":
            return kids[0] * kids[1]
        if op == "div":
            return kids[0] / kids[1]
        raise ValueError(f"未知算子 {op}")

    # ---- 收集节点（含父指针）-------------------------------------------
    def iter_nodes(self, parent: Node | None = None, ci: int = 0):
        yield (self, parent, ci)
        for i, c in enumerate(self.children):
            yield from c.iter_nodes(self, i)


def make_random_tree(
    max_depth: int,
    n_vars: int,
    rng: np.random.Generator,
    p_leaf: float,
    const_set: tuple,
    const_range: tuple,
    full: bool = False,
) -> Node:
    """随机生长一棵表达式树。

    full=True 时强制长到 max_depth（保证深度多样性，ramped half-and-half）。
    """
    if (
        (full and max_depth <= 1)
        or (max_depth <= 1)
        or (not full and rng.random() < p_leaf)
    ):
        if rng.random() < 0.5:
            return Node.var(int(rng.integers(0, n_vars)))
        # 常量：一半取自“漂亮”候选集，一半连续随机
        if rng.random() < 0.5:
            return Node.const(float(rng.choice(const_set)))
        return Node.const(float(rng.uniform(*const_range)))
    if rng.random() < 0.5:  # 二元算子
        op = str(rng.choice(list(BINARY.keys())))
        return Node.func(
            op,
            make_random_tree(
                max_depth - 1, n_vars, rng, p_leaf, const_set, const_range, full
            ),
            make_random_tree(
                max_depth - 1, n_vars, rng, p_leaf, const_set, const_range, full
            ),
        )
    op = str(rng.choice(list(UNARY.keys())))
    return Node.func(
        op,
        make_random_tree(
            max_depth - 1, n_vars, rng, p_leaf, const_set, const_range, full
        ),
    )


def ramped_init(
    pop_size: int,
    max_depth_init: int,
    n_vars: int,
    rng: np.random.Generator,
    const_set: tuple,
    const_range: tuple,
    p_leaf: float = 0.3,
) -> list[Node]:
    pop: list[Node] = []
    depths = list(range(2, max_depth_init + 1))
    i = 0
    while len(pop) < pop_size:
        d = depths[i % len(depths)]
        full = (i // len(depths)) % 2 == 0
        pop.append(
            make_random_tree(d, n_vars, rng, p_leaf, const_set, const_range, full)
        )
        i += 1
    return pop


def _pick_subtree(root: Node, index: int):
    for node, parent, ci in root.iter_nodes():
        if index == 0:
            return node, parent, ci
        index -= 1
    raise IndexError("index 越界")


def replace_subtree(root: Node, index: int, new: Node) -> Node:
    root = root.copy()
    if index == 0:
        return new.copy()
    for node, parent, ci in root.iter_nodes():
        if index == 0:
            parent.children[ci] = new.copy()
            return root
        index -= 1
    raise IndexError("index 越界")


def mutate(
    root: Node,
    rng: np.random.Generator,
    max_depth_mut: int,
    n_vars: int,
    const_set: tuple,
    const_range: tuple,
    p_leaf: float,
) -> Node:
    """变异：随机挑一棵子树，整体替换为新的随机子树；常量节点独立扰动。"""
    size = root.size()
    idx = int(rng.integers(0, size))
    node, _, _ = _pick_subtree(root, idx)
    if node.op == "const" and rng.random() < 0.5:
        # 常量扰动（小幅）
        new_val = node.value + float(rng.normal(0, 0.2))
        new_root = root.copy()
        for nn, pp, cci in new_root.iter_nodes():
            if nn is node:
                pp.children[cci] = Node.const(new_val)
                break
        return new_root
    new_sub = make_random_tree(
        max_depth_mut,
        n_vars,
        rng,
        p_leaf,
        const_set,
        const_range,
        full=(rng.random() < 0.5),
    )
    return replace_subtree(root, idx, new_sub)


def crossover(t1: Node, t2: Node, rng: np.random.Generator) -> Node:
    """子树交叉：在 t1 中挑一棵子树，替换为 t2 中的一棵随机子树。"""
    idx1 = int(rng.integers(0, t1.size()))
    # t2 中随机子树
    idx2 = int(rng.integers(0, t2.size()))
    sub2, _, _ = _pick_subtree(t2, idx2)
    return replace_subtree(t1, idx1, sub2)


# ---- 仿射精修：对选定树做 y ≈ a*t(x) + b 的最小二乘拟合 -------------------
def affine_refine(
    root: Node, X: np.ndarray, y: np.ndarray
) -> tuple[Node, float, float]:
    """对评估结果做全局仿射缩放 a*t(x)+b（岭回归），提升常量命中精度。"""
    t = root.eval(X)
    # 方差近零 → 树近似常数，仿射缩放无意义，直接返回
    if float(np.std(t)) < 1e-9:
        return root, 1.0, 0.0
    # 设计矩阵 [t, 1]，岭回归
    A = np.vstack([t, np.ones_like(t)]).T
    lam = 1e-8
    AtA = A.T @ A + lam * np.eye(2)
    Aty = A.T @ y
    coef, *_ = np.linalg.lstsq(AtA, Aty, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    # 裁剪缩放系数，避免病态树产生天文数字系数
    a = max(-1e6, min(1e6, a))
    if abs(a) < 1e-12:
        return root, 1.0, float(b)
    # 把 a,b 吸收进树：构造 (a)*(原树) + b 的新树
    new_root = Node.func(
        "add", Node.func("mul", Node.const(a), root.copy()), Node.const(b)
    )
    return new_root, float(a), float(b)


# ---- 常量精修：对树中全部常量节点做联合最小二乘优化（gplearn 风格）--------
def collect_const_nodes(root: Node) -> list[Node]:
    return [n for n, _, _ in root.iter_nodes() if n.op == "const"]


def optimize_constants(
    root: Node, X: np.ndarray, y: np.ndarray, max_nfev: int = 1000
) -> tuple[Node, float]:
    """把树中常量节点联合精修到最小二乘最优，将“结构正确但常量略偏”的近似
    推到 R2≈1.0（确定性：LM 无随机）。

    约束常量在有限区间内，避免无界 least_squares 把系数漂移到天文数字
    （既引发溢出，又导致其发散拖慢整轮拟合）。affine_refine 已做全局缩放，
    单常量通常很小，±30 的余量足够覆盖修正项。
    """
    consts = collect_const_nodes(root)
    if not consts:
        return root, 0.0
    orig = [c.value for c in consts]
    B = 30.0
    bounds = ([-B] * len(consts), [B] * len(consts))

    def resid(p):
        for c, v in zip(consts, p):
            c.value = float(v)
        pred = root.eval(X)
        # 防溢出：非有限或超大值裁剪到安全区间，避免 least_squares 发散
        if not np.all(np.isfinite(pred)):
            pred = np.where(np.isfinite(pred), pred, 1e9)
        pred = np.clip(pred, -1e9, 1e9)
        return (pred - y).astype(np.float64)

    try:
        from scipy.optimize import least_squares

        sol = least_squares(
            resid,
            np.array(orig, dtype=np.float64),
            bounds=bounds,
            max_nfev=max_nfev,
            xtol=1e-10,
            ftol=1e-10,
        )
        for c, v in zip(consts, sol.x):
            c.value = float(v)
        return root, float(sol.cost)
    except Exception:  # noqa: BLE001
        for c, v in zip(consts, orig):
            c.value = v
        return root, 0.0


# ---- Hoist 变异：抗膨胀（Schmidt & Lipson 2011 同款）----------------------
def hoist(root: Node, rng: np.random.Generator) -> Node:
    """随机挑一棵带子节点的子树 A，用 A 内部一棵随机后代 B 替换 A，缩短表达式。"""
    # 收集带子节点的节点
    internal = [(n, p, ci) for n, p, ci in root.iter_nodes() if len(n.children) > 0]
    if not internal:
        return root.copy()
    target, _, _ = internal[int(rng.integers(0, len(internal)))]
    # target 的随机后代（非自身）
    desc = [(n, p, ci2) for n, p, ci2 in target.iter_nodes() if n is not target]
    if not desc:
        return root.copy()
    b = desc[int(rng.integers(0, len(desc)))][0]
    new_root = root.copy()
    for nn, pp, cci in new_root.iter_nodes():
        if nn is target:
            pp.children[cci] = b.copy()
            return new_root
    return new_root
