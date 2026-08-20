"""Retrieval metrics — matched to BTC's published scorer.

THE OFFICIAL RULE (verbatim from phases/0_harness/btc_eval/scoring_legalir.py)
------------------------------------------------------------------------------
    recall    = mean_k [ |truth_k & pred_k| / |truth_k|  if 0 < len(pred_k) <= 5 else 0 ]
    precision = mean_k [ |truth_k & pred_k| / len(pred_k) if 0 < len(pred_k) <= 5 else 0 ]

Four consequences, each of which has a way of costing you the whole submission:

1. **At most 5 document_ids per question.** Return 6 and that question scores
   ZERO on *both* metrics — not a truncation, a zeroing. `src/cutoff.py` clamps
   at MAX_DOCS_PER_QUERY for this reason and you should never raise it.

2. **MACRO averaging.** `.mean()` over per-question ratios; every question
   weighs the same regardless of how many relevant documents it has.

3. **The precision denominator is `len(pred_k)`, NOT `len(set(pred_k))`.**
   A duplicated doc_id inflates the denominator, lowers precision, and still
   counts toward the cap. De-duplicate before submitting — `set_scores` below
   deliberately does NOT de-duplicate, so our numbers match theirs.

4. **A missing or extra question is a hard failure, not a zero.** Their code
   raises if `len(pred) != len(truth)`, and indexes `y_pred[k]` for every gold
   k. An incomplete submission errors out rather than scoring badly.

Ranking: Recall is primary, Precision breaks ties (confirmed by BTC on
02/08/2026 after an earlier email stated the reverse).

Rank-based metrics further down (Recall@k, MRR, MAP, nDCG) are retriever
DIAGNOSTICS, not the leaderboard metric. Recall@k in particular is the ceiling
that every downstream stage inherits.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Set

# CONFIRMED against BTC's scorer (btc_eval/scoring_legalir.py): `.mean()` over
# per-question ratios. Verified by `evaluate.py --cross-check`.
OFFICIAL_AVERAGING = "macro"

# Confirmed against BTC's scorer: `len(pred) <= 5` else the question scores 0.
MAX_DOCS_PER_QUERY = 5


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
    enforce_cap: bool = True,
) -> Dict[str, float]:
    """Official (macro) scores plus micro variants for diagnosis.

    Matches BTC's scorer exactly when ``enforce_cap`` is True (the default):
    a question with 0 or >MAX_DOCS_PER_QUERY predictions contributes 0 to both
    metrics, and the precision denominator is the RAW list length.

    ``enforce_cap=False`` is for analysis only — e.g. asking "what would recall
    be if we were allowed 20 documents?" Never report that number as a score.
    """
    tp = npred = nrel = 0
    macro_p = macro_r = macro_f = 0.0
    n = n_capped = n_empty = 0
    for qid, rel in qrels.items():
        pred = list(predictions.get(qid, []))      # NO dedup: matches BTC
        over = enforce_cap and len(pred) > MAX_DOCS_PER_QUERY
        empty = len(pred) == 0
        if over:
            n_capped += 1
        if empty:
            n_empty += 1

        if over or empty:
            p = r = f = 0.0
            hit = 0
        else:
            hit = len(set(pred) & rel)
            p, r, f = _prf(hit, len(pred), len(rel), beta)

        tp += hit
        npred += len(pred)
        nrel += len(rel)
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
        "n_over_cap": float(n_capped),
        "n_empty": float(n_empty),
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


def check_submittable(predictions: Dict[str, Sequence[str]],
                      qids: Sequence[str]) -> List[str]:
    """Return the reasons BTC's scorer would REJECT this submission (empty = ok).

    Their scorer raises rather than scoring badly, so these are hard errors:
      * a different number of questions than the reference
      * a gold question absent from the submission
    Plus the two silent score-killers we can also detect here.
    """
    problems = []
    missing = [q for q in qids if q not in predictions]
    extra = [q for q in predictions if q not in set(qids)]
    if missing:
        problems.append(f"{len(missing)} questions missing from the submission "
                        f"(e.g. {missing[:5]}) — BTC's scorer RAISES on this, "
                        f"the submission fails outright")
    if extra:
        problems.append(f"{len(extra)} questions in the submission that are not in "
                        f"the reference (e.g. {extra[:5]}) — count mismatch, "
                        f"the submission fails outright")
    over = [q for q, v in predictions.items() if len(v) > MAX_DOCS_PER_QUERY]
    if over:
        problems.append(f"{len(over)} questions return more than "
                        f"{MAX_DOCS_PER_QUERY} documents (e.g. {over[:5]}) — "
                        f"each scores ZERO on both metrics")
    dup = [q for q, v in predictions.items() if len(v) != len(set(v))]
    if dup:
        problems.append(f"{len(dup)} questions contain duplicate doc_ids "
                        f"(e.g. {dup[:5]}) — duplicates inflate the precision "
                        f"denominator and count toward the {MAX_DOCS_PER_QUERY} cap")
    empty = [q for q, v in predictions.items() if not v]
    if empty:
        problems.append(f"{len(empty)} questions have an empty prediction "
                        f"(e.g. {empty[:5]}) — each scores ZERO; raise min_k")
    # BTC's gold doc_ids are JSON *strings* ("280282") while the corpus files
    # carry them as JSON *ints* ("id": 740). Their scorer intersects raw sets:
    # {740} & {"740"} == set(). Ints therefore score a silent, total zero — no
    # error, no warning, just 0.0 on both metrics. Confirmed against the real
    # public-test data on 20/08/2026.
    nonstr = [q for q, v in predictions.items()
              if any(not isinstance(d, str) for d in v)]
    if nonstr:
        problems.append(f"{len(nonstr)} questions predict non-string doc_ids "
                        f"(e.g. {nonstr[:5]}) — BTC's gold ids are strings, so "
                        f"the set intersection is EMPTY and every such question "
                        f"scores 0 with no error message. Cast with str().")
    return problems


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
