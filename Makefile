# FeynmanForge Makefile
PY := python
VENV := .venv

.PHONY: venv install lint format test demo benchmark fit clean

venv:
	$(PY) -m venv $(VENV)
	$(VENV)/Scripts/pip install -U pip

install:
	$(VENV)/Scripts/pip install -r requirements.lock.txt

lint:
	$(VENV)/Scripts/python -m ruff check .

format:
	$(VENV)/Scripts/python -m ruff format .

test:
	$(VENV)/Scripts/python -m pytest -q

demo:
	$(VENV)/Scripts/python cli.py demo --out benchmark.json

benchmark:
	$(VENV)/Scripts/python cli.py benchmark --out benchmark.json

fit:
	$(VENV)/Scripts/python cli.py fit --func quad1 --seed 7

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache $(VENV)
