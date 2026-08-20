# 05 — Reranking and the ceiling argument

Prerequisite: `phases/3_rerank/README.md` Part A.

---

## 1. The ceiling, stated precisely

A reranker is a permutation of a fixed candidate list. Permutations do not add elements.
Therefore, for any reranker `f` and retrieval depth `k`:

```
Recall(final)  ≤  Recall@k(retriever)
```

with equality only if the cutoff keeps every relevant document that was retrieved. This is
not an empirical tendency — it is arithmetic, and it has a strong practical consequence:

> Before spending a day on the reranker, compute `recall@depth`. That number is the maximum
> possible payoff of the entire day's work.

## 2. The two-number diagnosis

Take `ceiling = recall@depth(retriever)` and `final = official recall after reranking`.

| ceiling | gap = ceiling − final | diagnosis | action |
|---|---|---|---|
| low (< 0.80) | any | relevant documents are never retrieved | **retriever**; reranking is capped and cannot help |
| high | large (> 0.10) | documents are present but ranked or cut badly | **reranker or cutoff** |
| high | small (< 0.03) | reranker is extracting nearly everything available | **retriever**, or stop |

`phases/3_rerank/ceiling_table.py` computes all of this and prints the verdict.

**Check the cutoff before blaming the reranker.** A large gap right after introducing a
cross-encoder is usually the cutoff, not the model — see §4.

## 3. Depth as a hyperparameter

Deeper reranking raises the ceiling and costs linearly more compute, but adds candidates
that are increasingly likely to be distractors.

Read the Δ column of the ceiling table:

```
depth 10  → recall 0.81
depth 50  → recall 0.91   (+0.10 — worth reranking deeper)
depth 100 → recall 0.93   (+0.02 — 2× the compute for 2 points of ceiling)
```

If `recall@100 − recall@50` is small, depth 100 is a poor trade. And note that the *ceiling*
gain is an upper bound on the *realised* gain — you will get less than the Δ.

## 4. Why the cutoff must be re-swept after reranking

Cross-encoders separate relevant from irrelevant far more decisively than retrievers,
because they attend across the pair at every layer. Their score distribution is much sharper.

The `ratio` rule keeps documents scoring above `α × top_score`. Under a sharper
distribution the same `α` keeps **fewer** documents. So a pipeline that was returning ~6
documents per query might drop to ~2 — recall falls, and it looks like the reranker made
things worse.

It did not; the cutoff is now mis-tuned. One command:

```bash
python phases/1_bm25/cutoff_sweep.py --run experiments/runs/rerank-best.jsonl --plot
```

Post-rerank optima are typically **smaller sets at a lower α**, but measure rather than
assume — the direction depends on your score distribution, and BCE-trained cross-encoders
behave differently from listwise ones.

## 5. Training objectives, and why the default here is BCE

**Binary cross-entropy on pairs.** Label 1 for gold, 0 for mined negatives; the model emits
a relevance logit. Scores are **calibrated across queries**, so a single `threshold` is
meaningful everywhere — which keeps a whole cutoff rule available to you.

**Listwise / localised contrastive.** Softmax over one positive and `n` negatives *for the
same query*. Usually a little better at ordering, but the scores are only comparable within
a query, so cross-query thresholding is meaningless and `ratio` becomes your only adaptive rule.

Given that the cutoff is a first-class model component in this competition, keeping an extra
rule viable is worth a small accuracy cost. `train_cross_encoder.py` uses BCE. **Test both**
if time allows, and report the comparison — it is a genuine ablation row.

**Class balance** (`--neg-per-pos`) matters: too few negatives and the model calls everything
relevant; too many and it collapses to "irrelevant". 4–8 is the usual range. Log what you used.

## 6. Why the negatives must come from your own retriever

At inference the reranker sees **only** the retriever's top-k. Train it on random corpus
documents and you train it for a job it never does: distinguishing labour law from traffic
law, which the retriever already solved.

The hard call it actually faces is two adjacent điều of the *same* statute, one of which
applies to the query's condition. If it never saw that during training, it reorders close to
randomly among them.

**Diagnostic signature:** excellent training metrics, near-zero dev improvement. If you see
that, check where your negatives came from before touching anything else.

## 7. Why `rerank.py` keeps the un-reranked tail

Documents beyond `--depth` are appended below everything reranked, with descending
placeholder scores, rather than dropped. Two reasons:

1. The run stays a **complete ranking**, so `recall@100` remains measurable and the ceiling
   analysis still works. Truncating at depth makes every downstream diagnostic report the
   depth as your ceiling.
2. You keep the ability to ask "would reranking deeper have helped?" without re-running
   retrieval.

---

## Check yourself

1. Retriever recall@50 = 0.88. Your reranked pipeline scores 0.88. Is the reranker perfect?
2. You add a cross-encoder and recall drops 9 points. What do you check first, and why is it
   not "the reranker is bad"?
3. Why does the ceiling table need *both* the retriever run and the reranked run, rather than
   just the final score?

<details><summary>answers</summary>

1. **Only in the weak sense that it lost nothing.** It means every relevant document in the
   top-50 survived into the answer set — the reranker extracted 100% of what was available.
   It says nothing about ordering quality, and it says nothing about the 12% of relevant
   documents that were never in the top-50. Your next move is the retriever, not the reranker.
2. **The cutoff.** Cross-encoder scores are much sharper, so the same `ratio` α now keeps far
   fewer documents; recall falls purely from smaller answer sets. Check `avg_pred_size`
   before and after — if it collapsed, re-sweep. Only if the cutoff is already optimal is the
   reranker itself suspect.
3. Because the final score alone cannot distinguish "the reranker is bad" from "the relevant
   documents were never retrieved". The ceiling comes from the *retriever* run; the gap
   between it and the final score is the only quantity that assigns blame.
</details>
