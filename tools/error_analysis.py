#!/usr/bin/env python3
"""Pull dev failures for manual categorisation. Run at the end of EVERY phase.

WHY 20 IS ENOUGH
----------------
BTC explicitly asked for analysis of *why* a method underperformed and what the
next method fixed. Twenty labelled failures per phase is enough to write that
section, and few enough that you will actually do it. A hundred is enough that
you will not.

HOW TO USE
----------
1. Run it. It writes a markdown worksheet with the worst-scoring dev queries,
   showing what you returned, what was gold, and where the gold document actually
   ranked.
2. Fill in the CATEGORY column by hand. Categorising is the work; the script only
   saves you the lookup.
3. Tally the categories at the bottom. That tally IS the paper section.

THE `rank of gold` COLUMN IS THE DIAGNOSIS
------------------------------------------
  rank 1-3 but not returned  -> cutoff problem, not a model problem
  rank 20-100                -> ranking problem: the reranker should fix it
  not in top-100 at all      -> retrieval problem: the reranker cannot help
  gold id not in corpus      -> harness bug, stop everything and fix ingest

STANDARD CATEGORIES (from the roadmap — extend as you find new ones)
--------------------------------------------------------------------
  granularity      right văn bản, wrong điều (or vice versa)
  lexical-mismatch query and source use different words for the same concept
  multi-article    answer requires several điều; we returned only one
  negation         the retrieved article states the opposite condition
  numeric          wrong article/decree number matched (12 vs 112)
  ambiguous-query  the question is under-specified; several answers defensible
  label-noise      our answer looks correct and the gold looks wrong
  cutoff           gold ranked highly but fell outside the answer set

USAGE
  python tools/error_analysis.py --run work/experiments/runs/<id>.jsonl --phase 1 -n 20
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import cutoff as cutoff_mod  # noqa: E402
from src import io_utils, metrics  # noqa: E402

CATEGORIES = ["granularity", "lexical-mismatch", "multi-article", "negation",
              "numeric", "ambiguous-query", "label-noise", "cutoff"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--queries", default="data/processed/queries_dev.jsonl")
    ap.add_argument("--corpus", default="data/processed/corpus_article.jsonl")
    ap.add_argument("--phase", default="x")
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--cutoff", default="ratio")
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--snippet", type=int, default=180)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    run = io_utils.load_run(a.run)
    queries = io_utils.load_queries(a.queries)
    qrels = io_utils.qrels(queries)
    preds = cutoff_mod.apply_to_run(run, rule=a.cutoff, alpha=a.alpha, k=a.k)
    # Text lookup is keyed by BOTH chunk id and parent id: a run may have been
    # aggregated to parent level (aggregate_to_parent) while the corpus file is
    # still chunk-level, and gold ids then legitimately match only the parents.
    dtext, known = {}, set()
    if os.path.exists(a.corpus):
        recs = list(io_utils.read_jsonl(a.corpus))
        for r in recs:
            dtext[r["doc_id"]] = r["text"]
            known.add(r["doc_id"])
            pid = r.get("meta", {}).get("parent_id")
            if pid:
                known.add(pid)
                dtext.setdefault(pid, r["text"])   # first chunk as a stand-in

    scored = []
    for q in queries:
        rel = qrels.get(q["qid"], set())
        if not rel:
            continue
        got = set(preds.get(q["qid"], []))
        recall = len(got & rel) / len(rel)
        ranked = [d for d, _ in run.get(q["qid"], [])]
        pos = {d: (ranked.index(d) + 1 if d in ranked else None) for d in rel}
        scored.append((recall, q, rel, preds.get(q["qid"], []), pos))
    scored.sort(key=lambda x: x[0])
    worst = scored[:a.n]

    out = a.out or f"work/analysis/error_analysis_phase{a.phase}.md"
    L = [f"# Error analysis — phase {a.phase}", "",
         f"Run: `{os.path.basename(a.run)}`  ·  cutoff: {a.cutoff} α={a.alpha}  ·  "
         f"{len(worst)} worst dev queries", "",
         "Fill in the **CATEGORY** column by hand. Categories: "
         + ", ".join(f"`{c}`" for c in CATEGORIES), "",
         "> Diagnosis from `rank of gold`: 1–3 = cutoff problem · 20–100 = ranking "
         "problem (reranker) · not in top-100 = retrieval problem (reranker cannot "
         "help) · not in corpus = harness bug.", ""]

    for i, (recall, q, rel, got, pos) in enumerate(worst, 1):
        L += [f"## {i}. `{q['qid']}` — recall {recall:.2f}", "",
              f"**Query:** {q['text']}", "",
              "| gold doc | rank in our run | in answer set? |", "|---|---|---|"]
        for d in sorted(rel):
            r = pos.get(d)
            if r:
                rs = str(r)
            elif not known or d in known:
                rs = "**not in top-%d**" % len(run.get(q["qid"], []))
            else:
                rs = "**NOT IN CORPUS — harness bug**"
            L.append(f"| `{d}` | {rs} | {'yes' if d in got else 'no'} |")
        L += ["", f"**We returned ({len(got)}):** " +
              ", ".join(f"`{d}`" for d in got[:10]) +
              (" …" if len(got) > 10 else ""), ""]
        if dtext:
            for d in sorted(rel)[:1]:
                if d in dtext:
                    L += [f"<details><summary>gold text `{d}`</summary>", "",
                          "> " + dtext[d][:a.snippet].replace("\n", " ") + " …", "",
                          "</details>", ""]
            top1 = got[0] if got else None
            if top1 and top1 in dtext and top1 not in rel:
                L += [f"<details><summary>what we returned first `{top1}`</summary>", "",
                      "> " + dtext[top1][:a.snippet].replace("\n", " ") + " …", "",
                      "</details>", ""]
        L += ["**CATEGORY:** _______________", "",
              "**Note:** ", "", "---", ""]

    L += ["## Tally (fill in after categorising)", "",
          "| category | count | what would fix it |", "|---|---|---|"]
    for c in CATEGORIES:
        L.append(f"| {c} | | |")
    L += ["", "## One paragraph for the paper", "",
          "_Why was this method insufficient, and what should the next method fix?_",
          "", "> "]

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    s = metrics.official(preds, qrels)
    print(f"overall dev: R={s['primary_recall']:.4f} P={s['tiebreak_precision']:.4f}")
    print(f"{len(worst)} worst queries -> {out}")
    zero = sum(1 for r, *_ in scored if r == 0.0)
    print(f"{zero}/{len(scored)} dev queries scored ZERO recall")
    if known:
        missing = {d for _, _, rel, _, _ in scored for d in rel if d not in known}
        if missing:
            print(f"\n*** {len(missing)} gold ids are NOT IN THE CORPUS "
                  f"(e.g. {sorted(missing)[:3]}) — this is a harness bug, not a "
                  f"model problem. Fix ingest.py before analysing anything else. ***")
    print("\nNow open the file and fill in the CATEGORY fields by hand. "
          "That is the actual work.")


if __name__ == "__main__":
    main()
