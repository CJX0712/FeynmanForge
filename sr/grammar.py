"""函数集与保护算子（sr.grammar）。

所有算子均向量化（对整批 X 一次计算）且做数值保护，避免除零 / 溢出导致
NaN 污染评测。与 paperswithcode/SRBench 主流算子集一致（+/-/*/、sin/cos/exp/
log/sqrt/square/neg）。
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9
_EXP_CLIP = 30.0


def _pdiv(a, b):
    denom = np.where(np.abs(b) > _EPS, b, _EPS)
    return a / denom


def _pexp(x):
    return np.exp(np.clip(x, -_EXP_CLIP, _EXP_CLIP))


def _plog(x):
    return np.log(np.abs(x) + _EPS)


def _psqrt(x):
    return np.sqrt(np.abs(x))


# 算子表：symbol -> (arity, numpy_func, sympy_name, str_fmt)
UNARY = {
    "sin": (np.sin, "sin", "sin({})"),
    "cos": (np.cos, "cos", "cos({})"),
    "exp": (_pexp, "exp", "exp({})"),
    "log": (_plog, "log", "log({})"),
    "sqrt": (_psqrt, "sqrt", "sqrt({})"),
    "square": (lambda x: x * x, "square", "({})**2"),
    "neg": (lambda x: -x, "neg", "(-{})"),
}

BINARY = {
    "add": (lambda a, b: a + b, "add", "({} + {})"),
    "sub": (lambda a, b: a - b, "sub", "({} - {})"),
    "mul": (lambda a, b: a * b, "mul", "({} * {})"),
    "div": (_pdiv, "div", "({} / {})"),
}

ALL_OPS = {**UNARY, **BINARY}


def is_unary(op):
    return op in UNARY


def is_binary(op):
    return op in BINARY


def apply_op(op, *args):
    if op in UNARY:
        return UNARY[op][0](args[0])
    return BINARY[op][0](args[0], args[1])


def sympy_name(op):
    return UNARY[op][1] if op in UNARY else BINARY[op][1]


def str_fmt(op):
    return UNARY[op][2] if op in UNARY else BINARY[op][2]


# 二元算子（用于随机生长时优先，保证非线性表达力）
BINARY_OPS = list(BINARY.keys())
UNARY_OPS = list(UNARY.keys())
