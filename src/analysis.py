"""Per-query diagnosis: where every lost point of Recall actually went.

WHY THIS EXISTS
---------------
"Recall is 0.62" tells you nothing you can act on. This module decomposes the
missing 0.38 into four causes that need four different fixes, and it does so
EXACTLY — the four components provably sum to the gap, per query and in
aggregate. That decomposition is the answer to BTC's paper requirement: *why was
this method insufficient, and what did the next one fix?*

THE DECOMPOSITION
-----------------
For one question, let

    G  = number of gold documents
    R  = how many of them the retriever returned ANYWHERE in the run
    T5 = how many of them are in the top-5 of the run
    A  = how many of them we actually submitted (hits)

BTC lets us submit at most 5 ids, and a cutoff rule returns a PREFIX of the
ranking, so these are nested:

    A  <=  T5  <=  min(R, 5)  <=  min(G, 5)  <=  G

Each step is a distinct, separately-fixable loss:

    loss_cap       = (G - min(G,5)) / G     gold beyond the 5-slot capacity.
                                            IMPOSSIBLE. No method can fix it.
    loss_retrieval = (min(G,5) - min(R,5)) / G
                                            never retrieved at all.
                                            Fix: the RETRIEVER. A reranker cannot
                                            reach a document that is not in the list.
    loss_ranking   = (min(R,5) - T5) / G    retrieved, but ranked below position 5.
                                            Fix: the RERANKER (or fusion weights).
    loss_cutoff    = (T5 - A) / G           sitting in the top-5 and we did not
                                            return it. Fix: the CUTOFF RULE —
                                            free, one command, no model change.

and by construction

    recall + loss_cap + loss_retrieval + loss_ranking + loss_cutoff = 1

A fifth term appears when BTC's scorer zeroes a question outright (empty answer,
or more than 5 ids). Then the hits we did have are discarded:

    loss_zeroed    = hits / G               SELF-INFLICTED. Fix the cutoff clamp.

Averaging each component over questions decomposes the macro-Recall gap itself,
because macro Recall is the mean of per-question recalls. So the aggregate table
reads: "of the 38 points we are missing, 6 are impossible, 19 need a better
retriever, 9 need a better reranker, 4 are a cutoff we can retune this afternoon."

READ THE AGGREGATE TABLE BEFORE CHOOSING WHAT TO WORK ON. It is the single most
decision-relevant artefact in the repo.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .metrics import MAX_DOCS_PER_QUERY

Ranked = Sequence[Tuple[str, float]]

# Scoring states, mirroring BTC's scorer exactly.
OK = "ok"
ZEROED_EMPTY = "zeroed_empty"
ZEROED_OVER_CAP = "zeroed_over_cap"

LOSS_COMPONENTS = ["loss_cap", "loss_retrieval", "loss_ranking",
                   "loss_cutoff", "loss_zeroed"]

FIXES = {
    "loss_cap": "IMPOSSIBLE — more than 5 gold documents; the cap forbids full recall",
    "loss_retrieval": "RETRIEVER — the document is not in the run at all",
    "loss_ranking": "RERANKER / FUSION — retrieved but ranked below position 5",
    "loss_cutoff": "CUTOFF RULE — in the top-5 and not returned; free to fix",
    "loss_zeroed": "SELF-INFLICTED — empty or >5 ids; clamp the cutoff",
}


def diagnose_query(
    qid: str,
    gold: Set[str],
    ranked: Ranked,
    predicted: Sequence[str],
    corpus_ids: Optional[Set[str]] = None,
    cap: int = MAX_DOCS_PER_QUERY,
) -> Dict[str, Any]:
    """Full diagnosis of one question. See the module docstring for the maths."""
    gold = set(gold)
    G = len(gold)
    ranked = list(ranked)
    predicted = list(predicted)
    doc_order = [d for d, _ in ranked]

    # --- scoring state, exactly as BTC computes it -------------------------
    if len(predicted) == 0:
        state = ZEROED_EMPTY
    elif len(predicted) > cap:
        state = ZEROED_OVER_CAP
    else:
        state = OK

    raw_hits = len(set(predicted) & gold)
    hits = raw_hits if state == OK else 0
    recall = hits / G if G else 0.0
    # BTC's precision denominator is the RAW list length (duplicates included)
    precision = (hits / len(predicted)) if (state == OK and predicted) else 0.0

    # --- where each gold document sits in the ranking ----------------------
    rank_of = {}
    for d in sorted(gold):
        rank_of[d] = doc_order.index(d) + 1 if d in doc_order else None
    ranks = [r for r in rank_of.values() if r is not None]

    R = len(ranks)                                   # gold retrieved anywhere
    T5 = sum(1 for r in ranks if r <= cap)           # gold inside the top-5

    # The chain below assumes A <= T5 <= min(R,cap) <= min(G,cap) <= G, which holds
    # when the prediction is a prefix of this run. It may NOT hold if the run and
    # prediction files disagree (different splits, a hand-edited submission, or a
    # prediction produced by some other pipeline). Rather than emit a negative
    # component we raise each ceiling to at least what was actually achieved --
    # honest, because achieving A hits proves A hits were achievable -- and flag it.
    prefix_violation = raw_hits > T5

    # `raw_hits` (not the zeroed `hits`) drives the chain: a zeroed question still
    # FOUND those documents, and forfeiting them is accounted for separately by
    # loss_zeroed. That keeps the identity exact in both scoring states.
    A_eff = raw_hits
    T5_eff = max(T5, A_eff)
    capR_eff = max(min(R, cap), T5_eff)
    capG_eff = max(min(G, cap), capR_eff)

    # --- the exact decomposition ------------------------------------------
    # recall + loss_cap + loss_retrieval + loss_ranking + loss_cutoff + loss_zeroed = 1
    if G == 0:
        losses = {k: 0.0 for k in LOSS_COMPONENTS}
    else:
        losses = {
            "loss_cap": (G - capG_eff) / G,
            "loss_retrieval": (capG_eff - capR_eff) / G,
            "loss_ranking": (capR_eff - T5_eff) / G,
            "loss_cutoff": (T5_eff - A_eff) / G,
            "loss_zeroed": (raw_hits / G) if state != OK else 0.0,
        }


    # --- slot economics ----------------------------------------------------
    slots_used = len(predicted)
    slots_wasted = len([d for d in predicted if d not in gold])
    slots_free = max(0, cap - slots_used)
    # could we have spent a free slot usefully?
    recoverable_in_top5 = max(0, T5_eff - A_eff)

    # --- data-integrity flags ---------------------------------------------
    dupes = [d for d in set(predicted) if predicted.count(d) > 1]
    unknown = ([d for d in predicted if d not in corpus_ids]
               if corpus_ids is not None else [])
    chunky = [d for d in predicted if "#" in str(d) or "::" in str(d)]
    gold_missing_from_corpus = ([d for d in gold if d not in corpus_ids]
                                if corpus_ids is not None else [])

    # --- score-shape features (why the cutoff behaved as it did) ----------
    scores = [s for _, s in ranked]
    top1 = scores[0] if scores else 0.0
    top2 = scores[1] if len(scores) > 1 else 0.0
    margin = top1 - top2
    margin_ratio = (top2 / top1) if top1 > 0 else 0.0
    tail_ratio = (scores[min(cap, len(scores)) - 1] / top1) if top1 > 0 and scores else 0.0

    return {
        "qid": qid,
        "n_gold": G,
        "n_pred": len(predicted),
        "n_hit": hits,
        "n_hit_raw": raw_hits,
        "recall": recall,
        "precision": precision,
        "state": state,
        "impossible": G > cap,
        # retrieval geometry
        "n_gold_retrieved": R,
        "n_gold_in_cap": T5,
        "best_gold_rank": min(ranks) if ranks else None,
        "worst_gold_rank": max(ranks) if ranks else None,
        "gold_ranks": rank_of,
        "run_depth": len(ranked),
        # ceilings, nested
        "ceiling_cap": (min(G, cap) / G) if G else 0.0,
        "ceiling_retrieval": (min(R, cap) / G) if G else 0.0,
        "ceiling_prefix": (T5 / G) if G else 0.0,
        # exact loss attribution
        **losses,
        "loss_total": sum(losses.values()),
        # slots
        "slots_used": slots_used,
        "slots_wasted": slots_wasted,
        "slots_free": slots_free,
        "recoverable_in_top5": recoverable_in_top5,
        # integrity
        "has_duplicates": bool(dupes),
        "duplicates": dupes,
        "unknown_ids": unknown,
        "chunk_ids": chunky,
        "gold_missing_from_corpus": gold_missing_from_corpus,
        "prefix_violation": prefix_violation,
        # score shape
        "top1_score": top1,
        "score_margin": margin,
        "score_margin_ratio": margin_ratio,
        "score_tail_ratio": tail_ratio,
    }


def diagnose_all(
    run: Dict[str, Ranked],
    qrels: Dict[str, Set[str]],
    predictions: Dict[str, Sequence[str]],
    corpus_ids: Optional[Set[str]] = None,
    cap: int = MAX_DOCS_PER_QUERY,
) -> List[Dict[str, Any]]:
    """Diagnose every question in ``qrels`` (missing ones count as empty)."""
    return [
        diagnose_query(qid, gold, run.get(qid, []), predictions.get(qid, []),
                       corpus_ids, cap)
        for qid, gold in qrels.items()
    ]


def aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Macro aggregate. The loss components decompose the macro-Recall gap."""
    n = len(rows) or 1
    out: Dict[str, Any] = {"n_queries": len(rows)}
    out["recall"] = sum(r["recall"] for r in rows) / n
    out["precision"] = sum(r["precision"] for r in rows) / n
    for k in LOSS_COMPONENTS:
        out[k] = sum(r[k] for r in rows) / n
    out["loss_total"] = sum(out[k] for k in LOSS_COMPONENTS)
    out["ceiling_cap"] = sum(r["ceiling_cap"] for r in rows) / n
    out["ceiling_retrieval"] = sum(r["ceiling_retrieval"] for r in rows) / n
    out["ceiling_prefix"] = sum(r["ceiling_prefix"] for r in rows) / n
    for k in ("slots_used", "slots_wasted", "slots_free", "recoverable_in_top5"):
        out[f"avg_{k}"] = sum(r[k] for r in rows) / n
    out["n_zeroed_empty"] = sum(1 for r in rows if r["state"] == ZEROED_EMPTY)
    out["n_zeroed_over_cap"] = sum(1 for r in rows if r["state"] == ZEROED_OVER_CAP)
    out["n_impossible"] = sum(1 for r in rows if r["impossible"])
    out["n_perfect"] = sum(1 for r in rows if r["recall"] >= 1.0)
    out["n_zero_recall"] = sum(1 for r in rows if r["recall"] <= 0.0)
    out["n_with_duplicates"] = sum(1 for r in rows if r["has_duplicates"])
    out["n_with_unknown_ids"] = sum(1 for r in rows if r["unknown_ids"])
    out["n_with_chunk_ids"] = sum(1 for r in rows if r["chunk_ids"])
    out["n_gold_missing_corpus"] = sum(1 for r in rows if r["gold_missing_from_corpus"])
    out["n_prefix_violation"] = sum(1 for r in rows if r["prefix_violation"])
    return out


def verdict(agg: Dict[str, Any]) -> List[str]:
    """Turn the aggregate into ranked, actionable statements."""
    lines = []
    # integrity first — these invalidate everything below them
    if agg["n_gold_missing_corpus"]:
        lines.append(
            f"HARNESS BUG: {agg['n_gold_missing_corpus']} questions have a gold id "
            f"that is not in the corpus. Your doc_id construction disagrees with "
            f"BTC's labels — recall is capped below 1.0 and no model can fix it. "
            f"Fix ingest.py before reading anything else on this page.")
    if agg["n_with_chunk_ids"]:
        lines.append(
            f"INVALID SUBMISSION: {agg['n_with_chunk_ids']} questions return "
            f"chunk-style ids. Add --aggregate max; BTC has never seen these ids.")
    if agg["n_zeroed_over_cap"]:
        lines.append(
            f"SELF-INFLICTED: {agg['n_zeroed_over_cap']} questions return more than "
            f"{MAX_DOCS_PER_QUERY} ids and are scored ZERO. Clamp the cutoff.")
    if agg["n_zeroed_empty"]:
        lines.append(
            f"SELF-INFLICTED: {agg['n_zeroed_empty']} questions return nothing and "
            f"are scored ZERO. Raise min_k to 1.")
    if agg["n_with_duplicates"]:
        lines.append(
            f"WASTE: {agg['n_with_duplicates']} questions contain duplicate ids. "
            f"Each duplicate lowers precision and burns one of five slots.")
    if agg["n_prefix_violation"]:
        lines.append(
            f"INCONSISTENT INPUT: {agg['n_prefix_violation']} predictions are not a "
            f"prefix of the run file — the prediction and run files disagree. "
            f"Loss attribution for those questions is approximate.")

    # then where the remaining recall went, largest first
    ranked = sorted(((agg[k], k) for k in LOSS_COMPONENTS), reverse=True)
    gap = agg["loss_total"]
    if gap > 1e-9:
        for value, key in ranked:
            if value <= 1e-9:
                continue
            share = 100 * value / gap
            lines.append(f"{value:.4f} of the {gap:.4f} recall gap ({share:.0f}%) "
                         f"-> {FIXES[key]}")
    if agg["avg_slots_free"] > 0.5 and agg["loss_cutoff"] > 1e-9:
        lines.append(
            f"CHEAPEST WIN: {agg['avg_slots_free']:.1f} of {MAX_DOCS_PER_QUERY} slots "
            f"are unused on average while {agg['loss_cutoff']:.4f} of recall sits in "
            f"the top-{MAX_DOCS_PER_QUERY} unreturned. Re-sweep the cutoff first.")
    return lines


def segment(rows: Sequence[Dict[str, Any]], key_fn, name: str) -> List[Dict[str, Any]]:
    """Group rows by ``key_fn`` and aggregate each group. For breakdown tables."""
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(key_fn(r), []).append(r)
    out = []
    for k in sorted(groups, key=lambda x: (x is None, x)):
        a = aggregate(groups[k])
        a[name] = k
        out.append(a)
    return out
