# 01 — Metrics, in more depth

Prerequisite: `phases/0_harness/README.md` Part A.

---

## 1. The monotonicity argument, written out

For a query with relevant set `R` and predicted set `P`:

```
Recall    = |P ∩ R| / |R|         denominator FIXED by the gold labels
Precision = |P ∩ R| / |P|         denominator CHOSEN by you
```

Add one document `d` to `P`:

- if `d ∈ R`: numerator +1 in both. Recall strictly increases. Precision increases only if
  the new document's "hit rate" (1) exceeds the current precision — which it does whenever
  precision < 1.
- if `d ∉ R`: numerator unchanged. **Recall unchanged.** Precision strictly decreases.

So Recall is **monotonically non-decreasing** in `|P|` and Precision is **non-increasing**
once you are past the relevant documents. There is no set size that maximises both.

This is why "Recall primary, Precision tiebreak" is not a mild preference. Taken literally
and alone, it says: *return everything*. That it is not the winning strategy tells you
something else is constraining the answer set — a cap in the code, or a rule in the
regulations. **Finding out which is Phase 0 Task B2**, and it is worth doing properly
because the answer sets your entire cutoff strategy.

## 2. Micro vs macro, worked

Three queries. Query A has 10 relevant documents and you find 8. Queries B and C have 1
each and you find 0.

```
micro recall = (8 + 0 + 0) / (10 + 1 + 1) = 8/12 = 0.667
macro recall = (0.8 + 0 + 0) / 3                  = 0.267
```

A 40-point gap on the same predictions. Micro says you are doing fine; macro says you are
failing two thirds of your users.

**Which is "right" depends on what the organiser wants to reward**, which is why you read
their code instead of assuming. As a rule of thumb: micro rewards systems that do well on
information-rich queries; macro rewards systems that do well on *every* query. Legal IR
datasets are usually dominated by single-answer queries, so macro is the harsher and more
common choice.

**Diagnostic value of the gap.** `src/metrics.set_scores` always reports both. Reading them
together:

| pattern | meaning |
|---|---|
| micro ≫ macro | strong on multi-answer queries, weak on single-answer ones |
| macro ≫ micro | strong on single-answer queries; missing documents on the rich ones |
| roughly equal | performance is uniform across query types |

## 3. Why rank-based metrics still matter when you are scored set-based

You are scored on a set, but you *build* a ranking. The rank metrics diagnose the ranking:

- **`recall@k`** — the ceiling. Everything downstream inherits it. This is the single most
  important diagnostic number in the whole project.
- **`MRR@10`** — how good the top of the list is. If MRR is high and your set score is low,
  the ranking is fine and the **cutoff** is wrong. That is a five-minute fix, so check it first.
- **`nDCG@10`** — ranking quality with position discount. Mostly a sanity check here, since
  labels are binary and the metric is set-based.
- **`MAP`** — overall ranking quality across all hits. Most useful on multi-answer queries.

A useful habit: when the official score drops, look at `recall@100` *first*. If it is
unchanged, you did not break retrieval — you broke ranking or cutoff, and the search space
for the bug is much smaller.

## 4. Edge cases that silently corrupt a harness

`src/metrics.py` handles each of these explicitly. Yours must match BTC's, which is
Phase 0 Task B2 Q4.

| Case | What we do | Why it matters |
|---|---|---|
| query in gold, missing from predictions | counted as empty: P=0, R=0 | silently dropping it **inflates** your score and is the #1 cause of dev/leaderboard divergence |
| duplicate doc_id in a prediction | de-duplicated, order kept | otherwise `\|P\|` is inflated and precision under-reports |
| unknown doc_id in a prediction | counted in `\|P\|`, never a hit | a chunk id that escaped `aggregate_to_parent` shows up here |
| empty prediction | P=0, R=0 | this is why `min_k ≥ 1`: an empty set is unrecoverable |
| query with zero relevant docs | R=0 by convention | check whether BTC excludes these instead |

## 5. How much is a 1-point difference worth?

With a dev split of ~600 queries, the standard error on a proportion near 0.7 is roughly
`sqrt(0.7·0.3/600) ≈ 0.019`. So **a 1-point dev difference is inside the noise** and a
3-point difference is marginal.

Practical consequences:
- Do not chase 0.5-point dev improvements. They are not real.
- When two configs are within a point, prefer the simpler/cheaper one and say so.
- Report the dev split size in the paper so a reader can do this arithmetic.
- The leaderboard has its own noise from a different sample; a 1-point leaderboard move is
  not evidence of anything either.

---

## Check yourself

1. You add 5 documents to every answer set and macro recall does not move at all. What does
   that tell you about ranks 11–15 of your rankings?
2. Your micro recall is 0.55 and macro recall is 0.72. Which query type are you failing,
   and does that make the cutoff too large or too small?
3. `recall@100` is 0.93 and your official recall is 0.62. Name the two possible causes and
   the metric that distinguishes them.

<details><summary>answers</summary>

1. Ranks 11–15 contained **no relevant documents at all** for any query — recall's numerator
   never changed. You paid precision for nothing. Your cutoff is already past the useful depth.
2. Macro > micro means you do well on single-answer queries and are **missing documents on
   multi-answer queries**. Those need *larger* sets, so a uniform cutoff is too small for
   them — which is exactly the case a `ratio` rule handles and a fixed k cannot.
3. Either (a) the ranking is bad — relevant documents sit deep in the top-100 — or (b) the
   cutoff is discarding them. **`MRR@10` distinguishes them**: high MRR means the top of the
   list is good and the cutoff is at fault; low MRR means the ranking is at fault.
</details>
