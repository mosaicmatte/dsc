"""Answer-set cutoff rules: turning a *ranking* into a variable-length answer set.

THE HARD CONSTRAINT
-------------------
BTC's scorer gives a question ZERO on both Recall and Precision if it returns
more than **5** document_ids (or zero of them). Not a truncation — a zeroing.
So every rule here is clamped to ``1 <= |answer set| <= MAX_DOCS_PER_QUERY``,
and ``max_k`` must never be raised above 5 for a real submission.

Within that 1..5 window the cutoff is still a genuine model component. Recall is
primary and Precision breaks ties, so on a query where the retriever is confident
you want to return 1 document (precision 1.0 costs you no recall), and where it
is unsure you want all 5. A fixed k cannot do both; the ``ratio`` rule can.

That is the whole game on this task: five slots per question, spent well.

The four rules
--------------
``top_k``       return a fixed k (1-5). Baseline; never the best.
``ratio``       keep docs scoring >= alpha * top_score. Adapts per query:
                a confident query with one dominant score returns 1 document,
                an ambiguous one returns many. Usually the biggest single win.
``threshold``   keep docs with score >= tau. Only sane when scores are
                calibrated across queries (cross-encoder logits are; raw BM25
                scores are NOT, they scale with query length).
``gap``         cut at the largest relative drop between consecutive scores.

All rules are clamped by ``min_k``/``max_k``, defaulting to 1 and 5. Both bounds
are load-bearing: an empty answer set scores zero, and a 6-document one also
scores zero.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .metrics import MAX_DOCS_PER_QUERY

Ranked = Sequence[Tuple[str, float]]


def apply_cutoff(
    ranked: Ranked,
    rule: str = "top_k",
    k: int = 10,
    alpha: float = 0.8,
    tau: float = 0.0,
    min_k: int = 1,
    max_k: int = MAX_DOCS_PER_QUERY,
) -> List[str]:
    if not ranked:
        return []
    if max_k > MAX_DOCS_PER_QUERY:
        raise ValueError(
            f"max_k={max_k} exceeds BTC's hard limit of {MAX_DOCS_PER_QUERY} "
            f"documents per question. Anything above it scores ZERO on both "
            f"metrics. Use metrics.set_scores(enforce_cap=False) if you are "
            f"deliberately running an above-the-cap analysis.")
    ranked = sorted(ranked, key=lambda x: -x[1])
    docs = [d for d, _ in ranked]
    scores = [s for _, s in ranked]

    if rule == "top_k":
        keep = k
    elif rule == "ratio":
        top = scores[0]
        if top <= 0:  # degenerate/negative scores -> ratio is meaningless
            keep = k
        else:
            keep = sum(1 for s in scores if s >= alpha * top)
    elif rule == "threshold":
        keep = sum(1 for s in scores if s >= tau)
    elif rule == "gap":
        keep = len(scores)
        best_drop, denom = -1.0, max(abs(scores[0]), 1e-9)
        for i in range(min(len(scores), max_k) - 1):
            drop = (scores[i] - scores[i + 1]) / denom
            if drop > best_drop:
                best_drop, keep = drop, i + 1
    else:
        raise ValueError(f"unknown cutoff rule: {rule!r}")

    # de-duplicate first: BTC's precision denominator is the RAW list length, so a
    # repeated doc_id both lowers precision and burns one of the five slots.
    docs = list(dict.fromkeys(docs))
    keep = max(min_k, min(keep, max_k, len(docs)))
    return docs[:keep]


def apply_to_run(run: Dict[str, Ranked], **kw) -> Dict[str, List[str]]:
    return {qid: apply_cutoff(r, **kw) for qid, r in run.items()}


# --------------------------------------------------------------------------
# sweeping
# --------------------------------------------------------------------------

def sweep(
    run: Dict[str, Ranked],
    qrels: Dict[str, set],
    max_k: int = MAX_DOCS_PER_QUERY,
    alphas: Sequence[float] = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98),
    min_k: int = 1,
    cap: int = MAX_DOCS_PER_QUERY,
) -> List[Dict]:
    """Score every cutoff configuration on dev. Returns rows sorted best-first.

    The search space is small by construction — k in 1..5 plus a handful of
    alphas — because BTC caps the answer set at 5. That makes an exhaustive
    sweep cheap, so re-run it after EVERY model change.

    The resulting Precision/Recall-vs-cutoff plot belongs in the paper;
    ``phases/1_bm25/cutoff_sweep.py`` writes it out.
    """
    from . import metrics

    rows: List[Dict] = []
    for k in range(1, max_k + 1):
        preds = apply_to_run(run, rule="top_k", k=k, min_k=min_k, max_k=cap)
        s = metrics.official(preds, qrels)
        rows.append({"rule": "top_k", "param": k, **_slim(s)})
    for a in alphas:
        preds = apply_to_run(run, rule="ratio", alpha=a, min_k=min_k, max_k=cap)
        s = metrics.official(preds, qrels)
        rows.append({"rule": "ratio", "param": a, **_slim(s)})
    preds = apply_to_run(run, rule="gap", min_k=min_k, max_k=cap)
    rows.append({"rule": "gap", "param": None, **_slim(metrics.official(preds, qrels))})
    rows.sort(key=lambda r: (r["recall"], r["precision"]), reverse=True)
    return rows


def _slim(s: Dict) -> Dict:
    return {
        "recall": s["primary_recall"],
        "precision": s["tiebreak_precision"],
        "f1": s.get("macro_f1", 0.0),
        "avg_set_size": s["avg_pred_size"],
    }


def best(rows: List[Dict]) -> Dict:
    return rows[0] if rows else {}
