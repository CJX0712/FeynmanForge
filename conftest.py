# 根目录 conftest.py：将仓库根加入 sys.path，使 `import core/data/sr/...` 可直接解析。
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
