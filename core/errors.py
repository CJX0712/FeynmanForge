"""错误码与异常（core.errors）。

错误码分段：
  E100 配置/输入
  E200 数据生成
  E300 GP 引擎
  E400 基线/后端
  E500 评测/管道
"""

from __future__ import annotations


class FeynmanForgeError(Exception):
    """基类。"""

    code = "E000"


class ConfigError(FeynmanForgeError):
    code = "E100"


class DataGenError(FeynmanForgeError):
    code = "E200"


class GPError(FeynmanForgeError):
    code = "E300"


class BackendError(FeynmanForgeError):
    code = "E400"


class EvalError(FeynmanForgeError):
    code = "E500"
