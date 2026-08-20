"""Answer-set cutoff rules: turning a *ranking* into a variable-length *answer set*.

Why this file is a model component and not a utility
-----------------------------------------------------
Task 1 reports Precision AND Recall. If every submission returned a fixed top-k,
Precision would be a deterministic function of Recall (P = R * |rel| / k averaged
out) and reporting both would be redundant. Reporting both only makes sense if
teams may return *different numbers of documents per query*. So the cutoff rule
is a thing you tune on dev, and re-tune after every change to the scorer,
because the score distribution moves every time the model does.

The four rules
--------------
``top_k``       return a fixed k. Baseline; never the best.
``ratio``       keep docs scoring >= alpha * top_score. Adapts per query:
                a confident query with one dominant score returns 1 document,
                an ambiguous one returns many. Usually the biggest single win.
``threshold``   keep docs with score >= tau. Only sane when scores are
                calibrated across queries (cross-encoder logits are; raw BM25
                scores are NOT, they scale with query length).
``gap``         cut at the largest relative drop between consecutive scores.

All rules are clamped by ``min_k``/``max_k``. ``min_k >= 1`` matters: an empty
answer set scores zero recall on that query and can never be recovered.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Ranked = Sequence[Tuple[str, float]]


def apply_cutoff(
    ranked: Ranked,
    rule: str = "top_k",
    k: int = 10,
    alpha: float = 0.8,
    tau: float = 0.0,
    min_k: int = 1,
    max_k: int = 50,
) -> List[str]:
    if not ranked:
        return []
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
    max_k: int = 20,
    alphas: Sequence[float] = (0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98),
    min_k: int = 1,
    cap: int = 50,
) -> List[Dict]:
    """Score every cutoff configuration on dev. Returns rows sorted best-first.

    The resulting Precision/Recall/official-score-vs-cutoff plot belongs in the
    paper; ``phases/1_bm25/cutoff_sweep.py`` writes it out.
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
