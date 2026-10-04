# FeynmanForge 运行时镜像（纯 numpy 推理 + 训练，无需 GPU）
FROM python:3.13-slim

WORKDIR /app

# 锁定依赖，确定性构建
COPY requirements.lock.txt /app/requirements.lock.txt
RUN pip install --no-cache-dir -r requirements.lock.txt

COPY . /app

# 默认运行演示基准
CMD ["python", "cli.py", "demo", "--out", "benchmark.json"]
