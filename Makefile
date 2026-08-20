# Convenience targets. Everything here is a plain command you can also type by hand —
# nothing is hidden, and `make -n <target>` shows you exactly what would run.
.PHONY: help setup check fixture walkthrough todo blockers log correlation budget clean

PY ?= python

help:  ## show this help
	@echo "DSC@UIT 2026 — common commands"
	@echo ""
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "New here? Read START_HERE.md"

setup:  ## create the venv and install dependencies
	$(PY) -m venv .venv
	./.venv/bin/pip install -r requirements.txt
	@echo "now run: source .venv/bin/activate && make check"

check:  ## verify the install end-to-end (synthetic data, no GPU needed)
	$(PY) phases/0_harness/smoke_test.py

fixture:  ## generate synthetic practice data in data/fixture/
	$(PY) tools/make_fixture.py --out data/fixture

walkthrough:  ## print the guided tutorial
	@echo "open docs/walkthrough.md — it runs the whole pipeline on synthetic data"

todo:  ## list everything outstanding
	$(PY) tools/todo.py

blockers:  ## list only the things that must be resolved before trusting a number
	$(PY) tools/todo.py --blockers

log:  ## show the experiment log
	@column -s, -t work/experiments/runs.csv 2>/dev/null || cat work/experiments/runs.csv

correlation:  ## does dev predict the leaderboard? (the Phase 1 gate)
	@$(PY) -c "import sys;sys.path.insert(0,'.');from src.exp_log import correlation as c;print(c())"

budget:  ## parameter budget against the 4B ceiling
	$(PY) src/params.py

clean:  ## remove caches and generated runs (keeps runs.csv, configs, analysis)
	rm -rf data/fixture data/processed/.bm25_* data/processed/.emb_*
	rm -f work/experiments/runs/*.jsonl work/experiments/predictions/*.jsonl
	find . -name __pycache__ -type d -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
	@echo "cleaned (runs.csv, work/configs and work/analysis kept)"
