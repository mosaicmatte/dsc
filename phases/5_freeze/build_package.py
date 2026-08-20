#!/usr/bin/env python3
"""Task B2 — build the reproduction package.

WHAT BTC REQUIRES
-----------------
Repository (or zip) + README with exact step-by-step reproduction + pinned
requirements + fixed seeds + weights or documented download steps. Docker optional.

THE ONLY TEST THAT MATTERS
--------------------------
Clone into a fresh directory, follow the generated README literally, and confirm
you reproduce the submitted file byte for byte. Then have a teammate who did NOT
write it do the same. Everything your README leaves implicit, they will hit.

USAGE
  python phases/5_freeze/build_package.py --config work/configs/FINAL.yaml --out dist/
  python phases/5_freeze/build_package.py --config work/configs/FINAL.yaml --out dist/ --zip
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.abspath(  # repo root: phases/<n>_<name>/ -> ../..
    os.path.join(os.path.dirname(__file__), "..", "..")))

from src import config  # noqa: E402

INCLUDE = ["src", "phases", "tools", "docs", "paper",
           "work/configs", "work/analysis",
           "README.md", "START_HERE.md", "requirements.txt", "Makefile"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="work/configs/FINAL.yaml")
    ap.add_argument("--out", default="dist")
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()

    cfg = config.load(a.config)
    run_id = cfg.get("run_id", "final")
    root = os.path.join(a.out, f"dsc2026-{run_id}")
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root)

    for item in INCLUDE:
        if not os.path.exists(item):
            continue
        dst = os.path.join(root, item)
        if os.path.isdir(item):
            shutil.copytree(item, dst,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc",
                                                          ".*", "*.npy", "*.pkl"))
        else:
            shutil.copy2(item, dst)

    os.makedirs(os.path.join(root, "experiments"), exist_ok=True)
    if os.path.exists("work/experiments/runs.csv"):
        shutil.copy2("work/experiments/runs.csv", os.path.join(root, "work/experiments/runs.csv"))

    # pinned requirements — the loose requirements.txt does not reproduce anything
    frozen = os.path.join(root, "requirements-frozen.txt")
    try:
        with open(frozen, "w", encoding="utf-8") as f:
            f.write(subprocess.check_output([sys.executable, "-m", "pip", "freeze"],
                                            text=True))
        print(f"pinned {frozen}")
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not pip freeze ({e}) — write it by hand", file=sys.stderr)

    env = cfg.get("environment", {})
    models = cfg.get("pipeline", {}) or {}
    readme = f"""# DSC@UIT 2026 — reproduction package

Run: `{run_id}`
Git commit: `{env.get('git_commit', 'UNKNOWN — record this')}`
Python: {env.get('python', '?')}

## 0. Requirements

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-frozen.txt
```

Package versions used for the submitted run:

| package | version |
|---|---|
""" + "\n".join(f"| {k} | {v} |" for k, v in (env.get("packages") or {}).items()
                if v) + f"""

## 1. Data

Place the BTC-provided files in `data/raw/`:

```
data/raw/<corpus file>
data/raw/<train file>
data/raw/<public test file>
data/raw/<private test file>
```

No other data is used. No augmentation. No external APIs.

## 2. Preprocess

```bash
python phases/0_harness/ingest.py --raw-corpus data/raw/<corpus> \\
    --raw-queries data/raw/<train> --raw-test data/raw/<test>
python phases/0_harness/ingest.py --validate
python phases/0_harness/build_dev_split.py --seed {cfg.get('log_row', {}).get('seed', 42)}
python phases/1_bm25/chunk_corpus.py --granularity article
```

## 3. Models

All models are downloaded from the Hugging Face Hub at the revisions below.
No training data other than BTC's is used.

| role | model | revision |
|---|---|---|
| | | |

<!-- TODO(BLOCKER/phase5-B2): fill in from phases/5_freeze/freeze_checklist.md §2.
     Model name AND revision SHA for each. "main" is not a revision. -->

## 4. Reproduce the submission

<!-- TODO(BLOCKER/phase5-B2): the exact commands, in order, from
     work/configs/FINAL.yaml. Every command copy-pasteable. Test them literally from a
     clean clone -- not from this working directory, which has caches the clean
     clone will not have. -->

```bash
# retrieval
# reranking
# submission
```

Output: `submissions/{run_id}.zip`

## 5. Verify

```bash
sha256sum submissions/{run_id}.zip
```

Expected: `<TODO(BLOCKER/phase5-B2): paste the sha256 after running §4 once>`

Running steps 2–4 twice from a clean state must produce byte-identical output.
All seeds are fixed in `work/configs/FINAL.yaml`.

## 6. Experiment log

`work/experiments/runs.csv` contains every run behind the paper's ablation table.
Each row's `config` column points at the frozen configuration in `work/configs/`.
"""
    with open(os.path.join(root, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    print(f"\npackage -> {root}")
    print("\nTHREE SECTIONS ARE DELIBERATELY LEFT BLANK. Fill them in now:")
    print("  §3 model revisions   (from freeze_checklist.md §2)")
    print("  §4 exact commands    (from work/configs/FINAL.yaml)")
    print("  §5 expected sha256   (after running §4 once)")

    if a.zip:
        archive = shutil.make_archive(root, "zip", a.out, os.path.basename(root))
        print(f"\nzipped -> {archive}")

    print("\nFINAL TEST: clone into a fresh directory, follow the README literally,\n"
          "and have a teammate who did not write it do the same.")


if __name__ == "__main__":
    main()
