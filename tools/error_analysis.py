#!/usr/bin/env python3
"""Full error analysis: where every lost point of Recall went, and what fixes it.

Run at the end of EVERY phase. The output is three things at once:

  1. an ALARM section — integrity problems that invalidate everything below them
  2. the LOSS DECOMPOSITION — the missing recall split into four causes that need
     four different fixes, with the components provably summing to the gap
  3. a WORKSHEET — the worst N questions, fully diagnosed, for you to categorise
     by hand

(2) is the part that decides what you work on next. (3) is the part BTC asked for
in the paper: *why was this method insufficient, and what did the next one fix?*

THE DECOMPOSITION, IN ONE LINE
------------------------------
    recall + loss_cap + loss_retrieval + loss_ranking + loss_cutoff + loss_zeroed = 1

    loss_cap        gold beyond the 5-slot cap        IMPOSSIBLE
    loss_retrieval  never retrieved                   fix the RETRIEVER
    loss_ranking    retrieved but ranked below 5      fix the RERANKER
    loss_cutoff     in the top-5, not returned        fix the CUTOFF RULE (free)
    loss_zeroed     empty or >5 ids, scored 0         SELF-INFLICTED

See `src/analysis.py` for the derivation.

USAGE
  python tools/error_analysis.py --run work/experiments/runs/<id>.jsonl --phase 1
  python tools/error_analysis.py --run <run> --cutoff ratio --alpha 0.85 -n 30
  python tools/error_analysis.py --run <run> --compare work/experiments/runs/<older>.jsonl
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import analysis, cutoff as cutoff_mod, io_utils, metrics  # noqa: E402

CATEGORIES = [
    ("granularity", "right văn bản, wrong điều (or the reverse)"),
    ("lexical-mismatch", "query and source use different words for the same thing"),
    ("numeric", "wrong article/decree number matched (Điều 12 vs Điều 112)"),
    ("multi-article", "answer spans several documents; we returned one"),
    ("negation", "retrieved text states the opposite condition"),
    ("too-general", "retrieved a broad law where a specific decree was wanted"),
    ("too-specific", "retrieved a narrow provision where the parent was wanted"),
    ("ambiguous-query", "question under-specified; several answers defensible"),
    ("label-noise", "our answer looks right and the gold looks wrong"),
    ("cutoff", "gold ranked inside the top-5 but we did not return it"),
    ("impossible", "more than 5 gold documents — the cap forbids full recall"),
]


def pct(x):
    return f"{100 * x:5.1f}%"


def bucket_rank(r):
    b = r["best_gold_rank"]
    if b is None:
        return "not retrieved"
    for lo, hi in ((1, 1), (2, 5), (6, 10), (11, 25), (26, 50), (51, 100)):
        if lo <= b <= hi:
            return f"{lo}-{hi}" if lo != hi else "1"
    return ">100"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--queries", default="data/processed/queries_dev.jsonl")
    ap.add_argument("--corpus", default="data/processed/corpus_article.jsonl")
    ap.add_argument("--phase", default="x")
    ap.add_argument("-n", type=int, default=20, help="questions in the worksheet")
    ap.add_argument("--cutoff", default="ratio",
                    choices=["top_k", "ratio", "threshold", "gap", "mine"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--compare", default=None,
                    help="an earlier run, to show what the current one fixed/broke")
    ap.add_argument("--snippet", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    run = io_utils.load_run(a.run)
    queries = io_utils.load_queries(a.queries)
    qrels = io_utils.qrels(queries)
    qtext = {q["qid"]: q["text"] for q in queries}

    dtext, known = {}, set()
    if os.path.exists(a.corpus):
        for r in io_utils.read_jsonl(a.corpus):
            dtext[r["doc_id"]] = r["text"]
            known.add(r["doc_id"])
            pid = r.get("meta", {}).get("parent_id")
            if pid:
                known.add(pid)
                dtext.setdefault(pid, r["text"])

    preds = cutoff_mod.apply_to_run(run, rule=a.cutoff, k=a.k, alpha=a.alpha)
    rows = analysis.diagnose_all(run, qrels, preds, known or None)
    agg = analysis.aggregate(rows)
    official = metrics.official(preds, qrels)

    out = a.out or f"work/analysis/error_analysis_phase{a.phase}.md"
    L = [f"# Error analysis — phase {a.phase}", "",
         f"Run: `{os.path.basename(a.run)}`  ·  cutoff: `{a.cutoff}` "
         f"(k={a.k}, α={a.alpha})  ·  {agg['n_queries']} dev questions", ""]

    # ---------------- 1. headline -----------------------------------------
    L += ["## 1. Headline", "",
          "| | |", "|---|---|",
          f"| **Recall (primary)** | **{official['primary_recall']:.4f}** |",
          f"| Precision (tiebreak) | {official['tiebreak_precision']:.4f} |",
          f"| questions with perfect recall | {agg['n_perfect']} / {agg['n_queries']} |",
          f"| questions with zero recall | {agg['n_zero_recall']} / {agg['n_queries']} |",
          f"| avg slots used (of {metrics.MAX_DOCS_PER_QUERY}) | {agg['avg_slots_used']:.2f} |",
          f"| avg slots wasted on non-relevant | {agg['avg_slots_wasted']:.2f} |", ""]

    # ---------------- 2. alarms -------------------------------------------
    alarms = analysis.verdict(agg)
    integrity = [x for x in alarms if x.startswith(("HARNESS", "INVALID",
                                                    "SELF-INFLICTED", "WASTE",
                                                    "INCONSISTENT"))]
    L += ["## 2. Integrity alarms", ""]
    if integrity:
        L += ["> **Fix these before reading anything else.** They invalidate the "
              "numbers below.", ""]
        L += [f"- {x}" for x in integrity] + [""]
    else:
        L += ["None. No missing gold ids, no chunk ids, no over-cap or empty "
              "answers, no duplicates.", ""]

    # ---------------- 3. the decomposition --------------------------------
    gap = agg["loss_total"]
    L += ["## 3. Where the missing recall went", "",
          f"Recall is **{agg['recall']:.4f}**, so **{gap:.4f}** is missing. "
          f"Every point of it is attributed below; the components sum to the gap "
          f"exactly, by construction.", "",
          "| cause | lost recall | share of gap | what fixes it |",
          "|---|---|---|---|"]
    for key in sorted(analysis.LOSS_COMPONENTS, key=lambda k: -agg[k]):
        v = agg[key]
        share = (100 * v / gap) if gap > 1e-12 else 0.0
        L.append(f"| `{key.replace('loss_', '')}` | {v:.4f} | {share:4.1f}% | "
                 f"{analysis.FIXES[key]} |")
    L += [f"| **total** | **{gap:.4f}** | 100% | |", "",
          f"Sanity: recall {agg['recall']:.4f} + gap {gap:.4f} = "
          f"{agg['recall'] + gap:.4f}", ""]

    # nested ceilings
    L += ["### Nested ceilings", "",
          "Each row is the best recall achievable if everything below it were perfect.",
          "",
          "| ceiling | value | meaning |", "|---|---|---|",
          f"| cap ceiling | {agg['ceiling_cap']:.4f} | best possible given ≤5 ids per question |",
          f"| retrieval ceiling | {agg['ceiling_retrieval']:.4f} | ...and given what the retriever returned |",
          f"| prefix ceiling | {agg['ceiling_prefix']:.4f} | ...and given the current ranking, with a perfect cutoff |",
          f"| **achieved** | **{agg['recall']:.4f}** | ...and given the cutoff you used |", ""]
    headroom = agg["ceiling_prefix"] - agg["recall"]
    if headroom > 1e-9:
        L.append(f"> **{headroom:.4f} recall is available from the cutoff alone** — "
                 f"the documents are already in your top-5. That is a sweep, not a "
                 f"model change.\n")

    # ---------------- 4. what-if: every cutoff ----------------------------
    L += ["## 4. What every cutoff would have scored", "",
          "| rule | param | recall | precision | avg set |", "|---|---|---|---|---|"]
    for r in cutoff_mod.sweep(run, qrels):
        p = "" if r["param"] is None else (f"{r['param']:.2f}"
                                           if isinstance(r["param"], float)
                                           else str(r["param"]))
        L.append(f"| {r['rule']} | {p} | {r['recall']:.4f} | "
                 f"{r['precision']:.4f} | {r['avg_set_size']:.2f} |")
    L.append("")

    # ---------------- 5. segments -----------------------------------------
    L += ["## 5. Breakdowns", "", "### By number of gold documents", "",
          "| \\|gold\\| | questions | recall | precision | cap loss | retrieval loss | ranking loss | cutoff loss |",
          "|---|---|---|---|---|---|---|---|"]
    for s in analysis.segment(rows, lambda r: r["n_gold"], "n_gold"):
        L.append(f"| {s['n_gold']} | {s['n_queries']} | {s['recall']:.4f} | "
                 f"{s['precision']:.4f} | {s['loss_cap']:.4f} | "
                 f"{s['loss_retrieval']:.4f} | {s['loss_ranking']:.4f} | "
                 f"{s['loss_cutoff']:.4f} |")
    L += ["", "> Questions with more than 5 gold documents can never reach recall 1.0. "
              "If that row is large, your headline recall has a hard ceiling below 1.", ""]

    L += ["### By where the best gold document ranked", "",
          "| best gold rank | questions | recall | what it means |", "|---|---|---|---|"]
    meaning = {
        "1": "already top — pure cutoff/precision question",
        "2-5": "inside the cap — cutoff decides whether you get it",
        "6-10": "just outside — a reranker should reach these",
        "11-25": "reranker territory",
        "26-50": "deep; reranking at depth 50 needed",
        "51-100": "very deep; check retrieval quality",
        "not retrieved": "RETRIEVER failure — reranking cannot help",
    }
    for s in analysis.segment(rows, bucket_rank, "bucket"):
        L.append(f"| {s['bucket']} | {s['n_queries']} | {s['recall']:.4f} | "
                 f"{meaning.get(s['bucket'], '')} |")
    L.append("")

    L += ["### By question length (words)", "",
          "| length | questions | recall | precision |", "|---|---|---|---|"]
    def lbucket(r):
        n = len(qtext.get(r["qid"], "").split())
        return "1-10" if n <= 10 else "11-20" if n <= 20 else "21-40" if n <= 40 else ">40"
    for s in analysis.segment(rows, lbucket, "bucket"):
        L.append(f"| {s['bucket']} | {s['n_queries']} | {s['recall']:.4f} | "
                 f"{s['precision']:.4f} |")
    L.append("")

    L += ["### By answer-set size actually returned", "",
          "| slots used | questions | recall | precision |", "|---|---|---|---|"]
    for s in analysis.segment(rows, lambda r: r["slots_used"], "slots"):
        L.append(f"| {s['slots']} | {s['n_queries']} | {s['recall']:.4f} | "
                 f"{s['precision']:.4f} |")
    L.append("")

    # ---------------- 6. comparison ---------------------------------------
    if a.compare and os.path.exists(a.compare):
        old_run = io_utils.load_run(a.compare)
        old_preds = cutoff_mod.apply_to_run(old_run, rule=a.cutoff, k=a.k,
                                            alpha=a.alpha)
        old_rows = {r["qid"]: r for r in
                    analysis.diagnose_all(old_run, qrels, old_preds, known or None)}
        new_rows = {r["qid"]: r for r in rows}
        fixed = [q for q in new_rows
                 if new_rows[q]["recall"] > old_rows.get(q, {}).get("recall", 0)]
        broke = [q for q in new_rows
                 if new_rows[q]["recall"] < old_rows.get(q, {}).get("recall", 0)]
        old_agg = analysis.aggregate(list(old_rows.values()))
        L += [f"## 6. Versus `{os.path.basename(a.compare)}`", "",
              f"| | previous | current | Δ |", "|---|---|---|---|",
              f"| recall | {old_agg['recall']:.4f} | {agg['recall']:.4f} | "
              f"{agg['recall'] - old_agg['recall']:+.4f} |",
              f"| precision | {old_agg['precision']:.4f} | {agg['precision']:.4f} | "
              f"{agg['precision'] - old_agg['precision']:+.4f} |"]
        for k in analysis.LOSS_COMPONENTS:
            L.append(f"| {k.replace('loss_','')} loss | {old_agg[k]:.4f} | "
                     f"{agg[k]:.4f} | {agg[k] - old_agg[k]:+.4f} |")
        L += ["", f"**{len(fixed)} questions improved, {len(broke)} regressed.**", ""]
        if broke:
            L += ["Regressions (investigate these — a net gain can hide a systematic "
                  "loss):", ""]
            for q in broke[:10]:
                L.append(f"- `{q}` {old_rows[q]['recall']:.2f} → "
                         f"{new_rows[q]['recall']:.2f} — {qtext.get(q,'')[:70]}")
            L.append("")

    # ---------------- 7. worksheet ----------------------------------------
    worst = sorted(rows, key=lambda r: (r["recall"], -r["loss_cutoff"]))[:a.n]
    L += [f"## 7. Worksheet — {len(worst)} worst questions", "",
          "Fill in **CATEGORY** by hand. Categories: "
          + ", ".join(f"`{c}`" for c, _ in CATEGORIES), "",
          "> `dominant loss` already tells you which *kind* of fix applies. Your job "
          "is to say *why* the model made that mistake — that is what a script "
          "cannot do and what the paper needs.", ""]

    for i, r in enumerate(worst, 1):
        dom = max(analysis.LOSS_COMPONENTS, key=lambda k: r[k])
        L += [f"### {i}. `{r['qid']}` — recall {r['recall']:.2f}, "
              f"precision {r['precision']:.2f}", "",
              f"**Q:** {qtext.get(r['qid'], '')}", "",
              f"- state: `{r['state']}`"
              + ("  **← scored ZERO**" if r["state"] != analysis.OK else ""),
              f"- gold: {r['n_gold']}  ·  retrieved: {r['n_gold_retrieved']}  ·  "
              f"in top-5: {r['n_gold_in_cap']}  ·  returned: {r['n_hit']}",
              f"- slots: used {r['slots_used']}, wasted {r['slots_wasted']}, "
              f"free {r['slots_free']}",
              f"- dominant loss: **{dom.replace('loss_','')}** → {analysis.FIXES[dom]}",
              ""]
        if r["impossible"]:
            L.append("> **IMPOSSIBLE**: more than 5 gold documents. Max achievable "
                     f"recall for this question is {r['ceiling_cap']:.2f}.\n")
        L += ["| gold doc | rank in run | in answer? |", "|---|---|---|"]
        for d, rk in sorted(r["gold_ranks"].items()):
            if rk is None:
                mark = ("**NOT IN CORPUS — harness bug**"
                        if d in r["gold_missing_from_corpus"]
                        else f"**not in top-{r['run_depth']}**")
            else:
                mark = str(rk)
            L.append(f"| `{d}` | {mark} | "
                     f"{'yes' if rk and rk <= r['slots_used'] else 'no'} |")
        L.append("")
        returned = preds.get(r["qid"], [])
        L.append(f"**Returned ({len(returned)}):** "
                 + (", ".join(f"`{d}`" for d in returned) or "_nothing_"))
        L.append("")
        if dtext:
            for d in sorted(r["gold_ranks"])[:1]:
                if d in dtext:
                    L += [f"<details><summary>gold `{d}`</summary>", "",
                          "> " + dtext[d][:a.snippet].replace("\n", " ") + " …", "",
                          "</details>", ""]
            wrong = [d for d in returned if d not in r["gold_ranks"]]
            if wrong and wrong[0] in dtext:
                L += [f"<details><summary>we returned `{wrong[0]}` instead</summary>",
                      "", "> " + dtext[wrong[0]][:a.snippet].replace("\n", " ") + " …",
                      "", "</details>", ""]
        L += ["**CATEGORY:** ______________    **WHY:** ", "", "---", ""]

    # ---------------- 8. tally + paragraph --------------------------------
    L += ["## 8. Tally (fill in after categorising)", "",
          "| category | count | what would fix it |", "|---|---|---|"]
    for c, desc in CATEGORIES:
        L.append(f"| `{c}` | | _{desc}_ |")
    L += ["", "## 9. One paragraph for the paper", "",
          "_Which method was insufficient, why, and what should the next one fix?_",
          "", "Template — replace the numbers with yours:", "",
          f"> At phase {a.phase} the system reached Recall {agg['recall']:.4f}. "
          f"Of the {gap:.4f} shortfall, {agg['loss_retrieval']:.4f} was documents "
          f"the retriever never returned, {agg['loss_ranking']:.4f} was documents "
          f"retrieved but ranked outside the five available slots, "
          f"{agg['loss_cutoff']:.4f} was documents present in the top five that the "
          f"cutoff rule discarded, and {agg['loss_cap']:.4f} was unreachable because "
          f"the question has more than five relevant documents. Manual inspection of "
          f"{len(worst)} failures attributed them chiefly to ___. This motivated ___ "
          f"in the next phase.", "", "> "]

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    # ---------------- terminal summary ------------------------------------
    print(f"Recall {agg['recall']:.4f}  Precision {agg['precision']:.4f}   "
          f"({agg['n_queries']} questions)")
    print(f"\nwhere the {gap:.4f} gap went:")
    for key in sorted(analysis.LOSS_COMPONENTS, key=lambda k: -agg[k]):
        if agg[key] > 1e-9:
            print(f"  {agg[key]:.4f}  {key.replace('loss_',''):<10} "
                  f"{analysis.FIXES[key]}")
    print()
    for line in alarms:
        print(f"  ! {line}")
    print(f"\nwrote {out}")
    print(f"Now open it and fill in the CATEGORY fields for the "
          f"{len(worst)} worst questions. That part is the actual work.")


if __name__ == "__main__":
    main()
