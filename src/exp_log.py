"""The experiment log. One CSV row per run, appended, never edited by hand.

The rule from the roadmap: *if you cannot regenerate a submission from its
run_id four weeks later, the run did not happen.* That means every row points at
a frozen config in ``work/configs/<run_id>.yaml`` and (once submitted) a git tag.

The paper's ablation table is a `pandas.read_csv` away, which is the entire
point of paying this small tax on every run.
"""
from __future__ import annotations

import csv
import datetime as _dt
import os
from typing import Any, Dict, List

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "work", "experiments", "runs.csv")

FIELDS = [
    "run_id", "date", "task", "phase",
    "chunking", "retriever", "negatives", "reranker", "cutoff_rule",
    "dev_P", "dev_R", "dev_official", "leaderboard",
    "n_params", "config", "git_tag", "seed", "notes",
]


def ensure_log(path: str = LOG_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    return path


def log_run(row: Dict[str, Any], path: str = LOG_PATH) -> Dict[str, Any]:
    """Append one run. Unknown keys are folded into ``notes`` rather than lost."""
    ensure_log(path)
    out = {k: "" for k in FIELDS}
    extra = []
    for k, v in row.items():
        if k in out:
            out[k] = v
        else:
            extra.append(f"{k}={v}")
    if extra:
        out["notes"] = "; ".join(filter(None, [str(out["notes"]), *extra]))
    if not out["date"]:
        out["date"] = _dt.date.today().isoformat()
    for k in ("dev_P", "dev_R", "dev_official"):
        if isinstance(out[k], float):
            out[k] = f"{out[k]:.4f}"
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(out)
    return out


def read_log(path: str = LOG_PATH) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def update_leaderboard(run_id: str, score: str | float,
                       path: str = LOG_PATH) -> bool:
    """Fill in the leaderboard column once Codabench reports back."""
    rows = read_log(path)
    hit = False
    for r in rows:
        if r["run_id"] == run_id:
            r["leaderboard"] = f"{score:.4f}" if isinstance(score, float) else str(score)
            hit = True
    if hit:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    return hit


def correlation(path: str = LOG_PATH) -> Dict[str, Any]:
    """Do dev and leaderboard move together?

    Phase 1 gate: if these diverge, STOP modelling and fix the harness. A dev
    split that does not predict the leaderboard is worse than no dev split,
    because it makes you confident while you are wrong.
    """
    rows = [r for r in read_log(path) if r.get("dev_official") and r.get("leaderboard")]
    pairs = []
    for r in rows:
        try:
            pairs.append((float(r["dev_official"]), float(r["leaderboard"]), r["run_id"]))
        except ValueError:
            continue
    n = len(pairs)
    if n < 3:
        return {"n": n, "pearson": None, "spearman": None,
                "verdict": "need >=3 submitted runs before this means anything"}

    def _pearson(xs, ys):
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
        return num / den if den else 0.0

    def _rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        for pos, i in enumerate(order):
            rk[i] = float(pos)
        return rk

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    pe = _pearson(xs, ys)
    sp = _pearson(_rank(xs), _rank(ys))
    verdict = ("healthy" if sp >= 0.8 else
               "SUSPECT - inspect the harness before trusting dev" if sp >= 0.5 else
               "BROKEN - stop modelling, fix the harness")
    return {"n": n, "pearson": pe, "spearman": sp, "verdict": verdict}
