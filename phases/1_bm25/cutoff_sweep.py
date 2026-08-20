#!/usr/bin/env python3
"""Task B4 — sweep the answer-set cutoff. THE figure for the paper.

WHAT THIS SHOWS
---------------
Recall rises monotonically with set size (its denominator is fixed by the gold
labels). Precision falls monotonically. The official score — Recall primary,
Precision tiebreak — therefore does NOT have an interior optimum in k alone;
that is exactly why the ``ratio`` rule matters: it buys recall on ambiguous
queries without paying precision on confident ones, so it can dominate every
fixed k simultaneously.

Read the plot for:
  * where Recall's slope flattens — past that, extra documents cost precision
    and buy almost nothing;
  * whether any ``ratio`` alpha sits above the top-k curve. If it does, a
    variable-length answer set is strictly better and you should never submit a
    fixed k again.

RE-RUN THIS AFTER EVERY MODEL CHANGE. The optimal cutoff moves whenever the
score distribution moves, and post-reranking distributions are much sharper.

USAGE
  python phases/1_bm25/cutoff_sweep.py --run work/experiments/runs/<id>.jsonl --plot
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(  # repo root: phases/<n>_<name>/ -> ../..
    os.path.join(os.path.dirname(__file__), "..", "..")))

from src import cutoff, io_utils  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--queries", default="data/processed/queries_dev.jsonl")
    ap.add_argument("--max-k", type=int, default=20)
    ap.add_argument("--cap", type=int, default=50)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default="work/analysis/fig_cutoff_sweep.png")
    ap.add_argument("--md", default="work/analysis/cutoff_sweep.md")
    a = ap.parse_args()

    run = io_utils.load_run(a.run)
    qrels = io_utils.qrels(io_utils.load_queries(a.queries))
    rows = cutoff.sweep(run, qrels, max_k=a.max_k, cap=a.cap)

    hdr = f"{'rule':<8} {'param':>7} {'recall':>8} {'prec':>8} {'f1':>8} {'|set|':>7}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        p = "" if r["param"] is None else (f"{r['param']:.2f}"
                                           if isinstance(r["param"], float)
                                           else str(r["param"]))
        print(f"{r['rule']:<8} {p:>7} {r['recall']:>8.4f} {r['precision']:>8.4f} "
              f"{r['f1']:>8.4f} {r['avg_set_size']:>7.2f}")

    b = rows[0]
    print(f"\nBEST: rule={b['rule']} param={b['param']} "
          f"recall={b['recall']:.4f} precision={b['precision']:.4f} "
          f"avg set size={b['avg_set_size']:.2f}")

    os.makedirs(os.path.dirname(a.md) or ".", exist_ok=True)
    with open(a.md, "w", encoding="utf-8") as f:
        f.write(f"# Cutoff sweep — `{os.path.basename(a.run)}`\n\n")
        f.write("| rule | param | recall | precision | f1 | avg set size |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['rule']} | {r['param']} | {r['recall']:.4f} | "
                    f"{r['precision']:.4f} | {r['f1']:.4f} | "
                    f"{r['avg_set_size']:.2f} |\n")
        f.write(f"\n**Best:** {b['rule']} @ {b['param']} — "
                f"R={b['recall']:.4f}, P={b['precision']:.4f}\n")
    print(f"wrote {a.md}")

    if a.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping plot", file=sys.stderr)
            return
        tk = sorted([r for r in rows if r["rule"] == "top_k"], key=lambda r: r["param"])
        rt = sorted([r for r in rows if r["rule"] == "ratio"], key=lambda r: r["param"])
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ks = [r["param"] for r in tk]
        ax[0].plot(ks, [r["recall"] for r in tk], "o-", label="Recall (primary)")
        ax[0].plot(ks, [r["precision"] for r in tk], "s-", label="Precision (tiebreak)")
        ax[0].plot(ks, [r["f1"] for r in tk], "^--", label="F1", alpha=.6)
        ax[0].set_xlabel("fixed top-k"); ax[0].set_ylabel("score")
        ax[0].set_title("Fixed-k cutoff"); ax[0].legend(); ax[0].grid(alpha=.3)
        al = [r["param"] for r in rt]
        ax[1].plot(al, [r["recall"] for r in rt], "o-", label="Recall")
        ax[1].plot(al, [r["precision"] for r in rt], "s-", label="Precision")
        ax2 = ax[1].twinx()
        ax2.plot(al, [r["avg_set_size"] for r in rt], "k:", label="avg |set|")
        ax2.set_ylabel("avg answer-set size")
        ax[1].set_xlabel("alpha (keep docs >= alpha x top score)")
        ax[1].set_title("Score-ratio cutoff"); ax[1].legend(loc="lower left")
        ax[1].grid(alpha=.3)
        fig.suptitle(f"Answer-set cutoff — {os.path.basename(a.run)}")
        fig.tight_layout()
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        fig.savefig(a.out, dpi=150)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
