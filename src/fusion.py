"""Combining a lexical run and a dense run.

Two families, and they fail differently:

**Reciprocal Rank Fusion (RRF)** uses only ranks:
    score(d) = sum_i w_i / (K + rank_i(d))
Scale-free, so BM25 scores (unbounded, query-length dependent) and cosine
similarities (in [-1, 1]) can be mixed without calibration. K=60 is the standard
constant; it damps the influence of the very top rank so one over-confident
system cannot dominate. RRF is the safe default and usually within a point of
the best tuned alternative.

**Weighted score fusion** normalises each system's scores per query, then takes
a weighted sum. It can beat RRF because it keeps score *margins* (the gap
between rank 1 and rank 2 carries information that ranks discard) -- and it is
what makes the ``ratio`` cutoff rule work well downstream. It needs per-query
normalisation, because raw BM25 scores are not comparable across queries.

Tune the weight on dev. Do not assume 0.5.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Run = Dict[str, List[Tuple[str, float]]]


def rrf(runs: Sequence[Run], weights: Sequence[float] | None = None,
        K: int = 60, top_k: int = 100) -> Run:
    weights = list(weights or [1.0] * len(runs))
    if len(weights) != len(runs):
        raise ValueError("weights and runs differ in length")
    qids = {q for r in runs for q in r}
    out: Run = {}
    for qid in qids:
        acc: Dict[str, float] = {}
        for run, w in zip(runs, weights):
            for rank, (doc, _) in enumerate(run.get(qid, []), 1):
                acc[doc] = acc.get(doc, 0.0) + w / (K + rank)
        out[qid] = sorted(acc.items(), key=lambda x: -x[1])[:top_k]
    return out


def _normalize(scores: Sequence[float], method: str = "minmax") -> List[float]:
    if not scores:
        return []
    if method == "minmax":
        lo, hi = min(scores), max(scores)
        rng = hi - lo
        return [1.0] * len(scores) if rng <= 1e-12 else [(s - lo) / rng for s in scores]
    if method == "zscore":
        n = len(scores)
        mu = sum(scores) / n
        var = sum((s - mu) ** 2 for s in scores) / n
        sd = var ** 0.5
        return [0.0] * n if sd <= 1e-12 else [(s - mu) / sd for s in scores]
    if method == "sum":
        tot = sum(abs(s) for s in scores) or 1.0
        return [s / tot for s in scores]
    raise ValueError(f"unknown normalisation: {method!r}")


def weighted(runs: Sequence[Run], weights: Sequence[float],
             method: str = "minmax", top_k: int = 100,
             missing: float = 0.0) -> Run:
    """Per-query score normalisation, then weighted sum.

    ``missing`` is the score assigned to a document absent from a system's run.
    0.0 with min-max means "as bad as that system's worst retrieved doc", which
    is the right prior for a truncated run.
    """
    if len(weights) != len(runs):
        raise ValueError("weights and runs differ in length")
    qids = {q for r in runs for q in r}
    out: Run = {}
    for qid in qids:
        acc: Dict[str, float] = {}
        seen: Dict[str, int] = {}
        for run, w in zip(runs, weights):
            lst = run.get(qid, [])
            if not lst:
                continue
            norm = _normalize([s for _, s in lst], method)
            for (doc, _), ns in zip(lst, norm):
                acc[doc] = acc.get(doc, 0.0) + w * ns
                seen[doc] = seen.get(doc, 0) + 1
        # documents retrieved by only some systems get `missing` from the rest
        total_w = sum(weights)
        for doc, cnt in seen.items():
            if cnt < len(runs):
                acc[doc] += missing * (total_w - sum(
                    w for w, r in zip(weights, runs) if doc in dict(r.get(qid, []))))
        out[qid] = sorted(acc.items(), key=lambda x: -x[1])[:top_k]
    return out


def sweep_weight(dense: Run, lexical: Run, qrels: Dict[str, set],
                 grid: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                          0.6, 0.7, 0.8, 0.9, 1.0),
                 depth: int = 100, method: str = "minmax") -> List[Dict]:
    """Tune the dense weight w (lexical gets 1-w). Reports Recall@depth."""
    from . import metrics

    rows = []
    for w in grid:
        fused = weighted([dense, lexical], [w, 1.0 - w], method=method, top_k=depth)
        rows.append({
            "w_dense": w,
            f"recall@{depth}": metrics.recall_at_k(fused, qrels, depth),
            "recall@10": metrics.recall_at_k(fused, qrels, 10),
            "mrr@10": metrics.mrr(fused, qrels, 10),
        })
    rows.sort(key=lambda r: -r[f"recall@{depth}"])
    return rows
