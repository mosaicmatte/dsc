# Phase 3 — self-check answer key

### 1. Recall@50 = 0.91, Recall@100 = 0.93, final Recall after reranking top-50 = 0.78. Where is the loss?

> The candidate list contains 91% of the relevant documents; you are delivering
> 78%. **13 points are being lost after retrieval** — the documents are there and
> are being ranked or cut away.
>
> Two suspects, and check them in this order because one is free:
>
> 1. **The cutoff rule, first.** Cross-encoder scores are far sharper than
>    retriever scores, so the same α now keeps far fewer documents. If your answer
>    sets shrank from ~6 documents to ~2 after reranking, that alone explains the
>    drop, and re-sweeping fixes it in one command. This is the most common cause
>    and costs nothing to rule out.
> 2. **The reranker itself.** If the cutoff is already optimal, the cross-encoder
>    is genuinely mis-ordering. Fine-tune it on your retriever's errors (Task B2).
>
> Note that Recall@100 is only 0.02 above Recall@50 — so reranking deeper would
> buy at most 2 points of ceiling for double the compute. Not where the money is.

### 2. Same retriever, but final Recall is 0.90. Now what?

> You are at 99% of the ceiling. **The reranker has nothing left to extract** —
> further reranker work is capped at +0.01 no matter how good it gets.
>
> The only way up is to **raise the ceiling**: improve the retriever. Options, in
> descending order of usual payoff: re-tune the hybrid fusion weight, chunk finer
> so relevant text is not buried in a long unit, another round of hard-negative
> mining, or deeper retrieval (though @100 is only +0.02 here, so that is capped
> too).
>
> This is precisely the diagnosis the ceiling table exists to produce, and it is
> also the honest answer to BTC's research question: at this point additional
> deep-learning capacity in the reranker stops paying off.

### 3. Why must reranker negatives come from your own retriever, and what goes wrong otherwise?

> **Because that is the only distribution the reranker will ever see at
> inference.** It is applied exclusively to the retriever's top-k. Training it on
> random corpus documents trains it for a job it never does.
>
> Concretely, what goes wrong: with random negatives the training task is
> "distinguish a labour-law article from a traffic-law article" — which is trivial,
> which the retriever already solved, and which the model learns in a few hundred
> steps before its loss goes flat. It never sees the hard call: two adjacent điều
> of the *same* statute, one of which applies to the query's condition and one of
> which does not. So at inference it is confronted entirely with pairs it was
> never trained on, and it reorders close to randomly among them.
>
> The symptom is diagnostic: excellent training metrics, near-zero improvement on
> dev. If you see that pattern, check where your negatives came from.

---

### Bonus

**Why does `rerank.py` keep the un-reranked tail below everything reranked instead of dropping it?**
> So the run file stays a complete ranking and `recall@100` is preserved for the
> ceiling analysis. If you truncate at the rerank depth, every downstream
> diagnostic silently reports the depth as your ceiling, and you lose the ability
> to ask "would reranking deeper have helped?"

**BCE vs listwise for the cross-encoder — why does the default matter here specifically?**
> BCE logits are calibrated *across* queries, so a single `threshold` cutoff is
> meaningful for every query. Listwise scores are only comparable *within* a
> query, which leaves `ratio` as the only viable adaptive rule. Given that the
> cutoff rule is a first-class model component in this competition, keeping an
> extra rule available is worth the small accuracy cost — but test both.
