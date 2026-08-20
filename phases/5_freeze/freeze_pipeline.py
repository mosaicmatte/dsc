#!/usr/bin/env python3
"""Task B1 — freeze the winning configuration.

WHAT IT DOES
------------
Reads `work/experiments/runs.csv`, shows you the leaderboard-ranked table, and writes
the chosen run's config to `work/configs/FINAL.yaml` together with an environment
fingerprint (package versions, python version, git commit) so the run can be
reconstructed.

WHY LEADERBOARD AND NOT DEV
---------------------------
Dev was the iteration loop and you have been fitting to it for four weeks — some
of its advantage over the leaderboard is overfitting. The leaderboard is a
held-out estimate of the Private Test. When they disagree about which run is
best, the leaderboard wins, and the disagreement itself is worth a paragraph in
the paper.

USAGE
  python phases/5_freeze/freeze_pipeline.py --list
  python phases/5_freeze/freeze_pipeline.py --run-id <id> --out work/configs/FINAL.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(  # repo root: phases/<n>_<name>/ -> ../..
    os.path.join(os.path.dirname(__file__), "..", "..")))

from src import config, exp_log  # noqa: E402


def env_fingerprint():
    def sh(cmd):
        try:
            return subprocess.check_output(cmd, shell=True, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:  # noqa: BLE001
            return None
    pkgs = {}
    for mod in ("torch", "transformers", "sentence_transformers", "numpy",
                "faiss", "peft", "pyvi", "underthesea"):
        try:
            m = __import__(mod)
            pkgs[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            pkgs[mod] = None
    return {"python": sys.version.split()[0], "packages": pkgs,
            "git_commit": sh("git rev-parse HEAD"),
            "git_dirty": bool(sh("git status --porcelain")),
            "platform": sys.platform}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default="work/configs/FINAL.yaml")
    a = ap.parse_args()

    rows = exp_log.read_log()
    if not rows:
        raise SystemExit("work/experiments/runs.csv is empty — nothing to freeze")

    def lb(r):
        try:
            return float(r.get("leaderboard", ""))
        except ValueError:
            return -1.0

    if a.list or not a.run_id:
        ranked = sorted(rows, key=lb, reverse=True)
        print(f"{'run_id':<34} {'lb':>8} {'dev_R':>8} {'dev_P':>8}  notes")
        print("-" * 100)
        for r in ranked[:25]:
            print(f"{r['run_id']:<34} {r.get('leaderboard',''):>8} "
                  f"{r.get('dev_R',''):>8} {r.get('dev_P',''):>8}  "
                  f"{r.get('notes','')[:38]}")
        scored = [r for r in ranked if lb(r) >= 0]
        if scored:
            print(f"\nleaderboard best: {scored[0]['run_id']} ({scored[0]['leaderboard']})")
        dev_best = max(rows, key=lambda r: float(r.get("dev_official") or -1))
        print(f"dev best        : {dev_best['run_id']} ({dev_best.get('dev_official')})")
        if scored and dev_best["run_id"] != scored[0]["run_id"]:
            print("\nNOTE: dev and leaderboard disagree about the winner. Freeze the\n"
                  "leaderboard's choice, and write the disagreement into the paper —\n"
                  "it says something real about your dev split.")
        if not a.run_id:
            return

    row = next((r for r in rows if r["run_id"] == a.run_id), None)
    if row is None:
        raise SystemExit(f"run_id {a.run_id!r} not in work/experiments/runs.csv")

    src_cfg = row.get("config") or f"work/configs/{a.run_id}.yaml"
    payload = {"run_id": a.run_id, "log_row": row, "environment": env_fingerprint()}
    if os.path.exists(src_cfg):
        payload["pipeline"] = config.load(src_cfg)
    else:
        payload["pipeline"] = None
        print(f"WARNING: no frozen config at {src_cfg}. Reconstruct it by hand and\n"
              f"         paste it under `pipeline:` — a FINAL.yaml without the\n"
              f"         pipeline section does not reproduce anything.", file=sys.stderr)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    try:
        import yaml  # type: ignore
        text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    except ImportError:
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"froze {a.run_id} -> {a.out}")
    if payload["environment"]["git_dirty"]:
        print("WARNING: git working tree is dirty. Commit and tag before freezing:\n"
              f"  git add -A && git commit -m 'freeze {a.run_id}' && "
              f"git tag final-{a.run_id}")
    else:
        print(f"next:\n  git tag final-{a.run_id}\n"
              f"  python phases/5_freeze/build_package.py --config {a.out} --out dist/")


if __name__ == "__main__":
    main()
