"""确定性验证：同配置两次运行，核心指标逐位一致（忽略墙钟耗时字段）。"""
import json
import subprocess

VENV = "C:/Users/Administrator/.workbuddy/binaries/python/envs/feynmanforge/Scripts/python.exe"


def run(out):
    subprocess.run([VENV, "cli.py", "demo", "--out", out], check=True)


run("/tmp/d1.json")
run("/tmp/d2.json")

with open("/tmp/d1.json", encoding="utf-8") as f:
    a = json.load(f)
with open("/tmp/d2.json", encoding="utf-8") as f:
    b = json.load(f)


def strip(d):
    return [
        {k: v for k, v in r.items() if k != "elapsed_sec"}
        for r in d["records"]
    ]


assert strip(a) == strip(b), "records 不一致（非确定性！）"
assert a["gate"] == b["gate"], "gate 不一致"


def strip_agg(aggs):
    # 忽略墙钟耗时字段（mean_elapsed 每轮不同属正常），其余指标须逐位一致
    return [
        {k: v for k, v in agg.items() if k != "mean_elapsed"}
        for agg in aggs
    ]


assert strip_agg(a["agg"]) == strip_agg(b["agg"]), "agg 不一致（非耗时字段）"
print("DETERMINISM OK：两次运行核心指标逐位一致（忽略耗时字段）")
