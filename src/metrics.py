"""Retrieval metrics.

Two families, and the difference matters for this competition:

* **Set-based** (Precision, Recall, F1, F2) score an *answer set* of whatever
  size you chose. Order inside the set is irrelevant.
* **Rank-based** (Recall@k, MRR, MAP, nDCG) score a *ranking* truncated at a
  fixed k. They are diagnostics for the retriever, not the leaderboard metric.

Task 1 is scored set-based (Recall primary, Precision tiebreak), which is why
``src/cutoff.py`` exists at all: choosing how many documents to return per query
is a model component, not a formatting detail.

Micro vs macro is NOT a detail either. Until Phase 0 has read BTC's evaluation
source, we compute both and report both; ``phases/0_harness/eval_code_notes.md``
records the answer and ``OFFICIAL_AVERAGING`` below gets set to match.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Set

# Set in Phase 0 from BTC's published evaluation code: "micro" or "macro".
# TODO(BLOCKER/phase0-B2): confirm against BTC's evaluation source and change if
# needed. Look for `sum(hits)/sum(rel)` (micro) vs `mean(hits_i/rel_i)` (macro).
# Every dev number in the repo is reported under this setting; getting it wrong
# means tuning against a metric nobody is scoring you on.
OFFICIAL_AVERAGING = "macro"


# --------------------------------------------------------------------------
# set-based
# --------------------------------------------------------------------------

def _prf(n_hit: int, n_pred: int, n_rel: int, beta: float = 1.0):
    p = n_hit / n_pred if n_pred else 0.0
    r = n_hit / n_rel if n_rel else 0.0
    if p + r == 0:
        f = 0.0
    else:
        b2 = beta * beta
        f = (1 + b2) * p * r / (b2 * p + r)
    return p, r, f


def set_scores(
    predictions: Dict[str, Sequence[str]],
    qrels: Dict[str, Set[str]],
    beta: float = 1.0,
) -> Dict[str, float]:
    """Micro- and macro-averaged Precision/Recall/F over all queries in qrels.

    A query present in ``qrels`` but missing from ``predictions`` counts as an
    empty prediction (precision undefined -> 0, recall 0). Silently dropping it
    would inflate the score, which is exactly the bug that makes a dev harness
    disagree with the leaderboard.
    """
    tp = npred = nrel = 0
    macro_p = macro_r = macro_f = 0.0
    n = 0
    for qid, rel in qrels.items():
        pred = list(dict.fromkeys(predictions.get(qid, [])))  # dedupe, keep order
        hit = len(set(pred) & rel)
        tp += hit
        npred += len(pred)
        nrel += len(rel)
        p, r, f = _prf(hit, len(pred), len(rel), beta)
        macro_p += p
        macro_r += r
        macro_f += f
        n += 1
    n = max(n, 1)
    mi_p, mi_r, mi_f = _prf(tp, npred, nrel, beta)
    return {
        "micro_precision": mi_p, "micro_recall": mi_r, f"micro_f{beta:g}": mi_f,
        "macro_precision": macro_p / n, "macro_recall": macro_r / n,
        f"macro_f{beta:g}": macro_f / n,
        "n_queries": float(n),
        "avg_pred_size": npred / n,
        "avg_rel_size": nrel / n,
    }


def official(
    predictions: Dict[str, Sequence[str]],
    qrels: Dict[str, Set[str]],
    averaging: str | None = None,
) -> Dict[str, float]:
    """Task 1 leaderboard view: Recall primary, Precision tiebreak.

    Returns ``primary`` and ``tiebreak`` plus a ``sort_key`` tuple so runs can
    be ordered exactly the way the leaderboard orders them.
    """
    avg = averaging or OFFICIAL_AVERAGING
    s = set_scores(predictions, qrels)
    rec = s[f"{avg}_recall"]
    prec = s[f"{avg}_precision"]
    s.update({"primary_recall": rec, "tiebreak_precision": prec,
              "averaging": avg, "sort_key": (rec, prec)})
    return s


def compare(a: Dict[str, float], b: Dict[str, float]) -> int:
    """Leaderboard ordering: 1 if a beats b, -1 if b beats a, 0 if identical."""
    ka, kb = a["sort_key"], b["sort_key"]
    return (ka > kb) - (ka < kb)


# --------------------------------------------------------------------------
# rank-based (retriever diagnostics)
# --------------------------------------------------------------------------

def _topk(ranked: Sequence, k: int) -> List[str]:
    return [d for d, _ in ranked[:k]] if ranked and isinstance(ranked[0], (list, tuple)) \
        else list(ranked[:k])


def recall_at_k(run: Dict[str, Sequence], qrels: Dict[str, Set[str]], k: int) -> float:
    """The retrieval CEILING at depth k: no reranker over the top-k can beat it."""
    tot = 0.0
    for qid, rel in qrels.items():
        if not rel:
            continue
        top = set(_topk(run.get(qid, []), k))
        tot += len(top & rel) / len(rel)
    return tot / max(len(qrels), 1)


def precision_at_k(run: Dict[str, Sequence], qrels: Dict[str, Set[str]], k: int) -> float:
    tot = 0.0
    for qid, rel in qrels.items():
        top = _topk(run.get(qid, []), k)
        tot += (len(set(top) & rel) / k) if k else 0.0
    return tot / max(len(qrels), 1)


def mrr(run: Dict[str, Sequence], qrels: Dict[str, Set[str]], k: int = 10) -> float:
    tot = 0.0
    for qid, rel in qrels.items():
        for i, d in enumerate(_topk(run.get(qid, []), k), 1):
            if d in rel:
                tot += 1.0 / i
                break
    return tot / max(len(qrels), 1)


def average_precision(ranked: Sequence, rel: Set[str], k: int = 100) -> float:
    if not rel:
        return 0.0
    hits = 0
    tot = 0.0
    for i, d in enumerate(_topk(ranked, k), 1):
        if d in rel:
            hits += 1
            tot += hits / i
    return tot / min(len(rel), k)


def mean_ap(run: Dict[str, Sequence], qrels: Dict[str, Set[str]], k: int = 100) -> float:
    return sum(average_precision(run.get(q, []), r, k)
               for q, r in qrels.items()) / max(len(qrels), 1)


def ndcg_at_k(run: Dict[str, Sequence], qrels: Dict[str, Set[str]], k: int = 10) -> float:
    """Binary-gain nDCG (BTC's labels are binary relevance)."""
    tot = 0.0
    for qid, rel in qrels.items():
        if not rel:
            continue
        dcg = sum(1.0 / math.log2(i + 1)
                  for i, d in enumerate(_topk(run.get(qid, []), k), 1) if d in rel)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
        tot += dcg / idcg if idcg else 0.0
    return tot / max(len(qrels), 1)


def diagnostics(run: Dict[str, Sequence], qrels: Dict[str, Set[str]],
                ks: Iterable[int] = (1, 5, 10, 20, 50, 100)) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(run, qrels, k)
        out[f"precision@{k}"] = precision_at_k(run, qrels, k)
    out["mrr@10"] = mrr(run, qrels, 10)
    out["map@100"] = mean_ap(run, qrels, 100)
    out["ndcg@10"] = ndcg_at_k(run, qrels, 10)
    return out
