# Phase 0 — self-check answer key

Attempt the questions in the README before reading. If you cannot answer these
from memory, you will misread your own dev numbers for the next four weeks.

---

### 1. State `Recall@k` from memory.

> For one query: of all documents labelled relevant, what fraction appear in the
> top-k of your ranking.
> `Recall@k = |top-k ∩ relevant| / |relevant|`, averaged over queries.
>
> The property that matters: **it is a ceiling.** If your retriever's Recall@50
> is 0.85, then a reranker that only ever sees those 50 documents can never
> produce a final Recall above 0.85, no matter how good it is. This is the whole
> basis of the Phase 3 ceiling table.

### 2. Why could returning 100 documents per query raise Recall and lower Precision, and what does BTC's code do about it?

> Recall's denominator is `|relevant|` — fixed by the gold labels. Adding
> documents can only add hits, so Recall is **monotonically non-decreasing** in
> set size. Precision's denominator is `|predicted|` — which you control — so
> every non-relevant document you add strictly lowers it.
>
> **What BTC does about it: answer this from their source, not from here.**
> Record it in `02_eval_code_notes.md` with a line reference. The three
> possibilities are (a) a hard cap on set size, (b) Precision entering the
> ranking so the exploit self-penalises, (c) nothing in the code and a rule in
> the regulations instead.

### 3. Dev Recall 0.72 micro but 0.61 macro — what does that gap tell you?

> Micro pools all queries (`Σhits / Σrelevant`), so queries with many relevant
> documents dominate it. Macro averages per query, so every query counts once.
>
> **Micro > macro means you are doing well on queries with many relevant
> documents and badly on queries with few.** Since most legal-IR queries have
> one or two relevant articles, that is the opposite of where you want to be
> strong — and if BTC scores macro, this gap is 11 points of leaderboard you are
> leaving behind.
>
> The fix is usually the cutoff rule, not the model: a fixed large k finds the
> extra documents on multi-answer queries while diluting single-answer ones.

---

### Bonus — you should also be able to answer these

**Why is the dev split stratified by |relevant| rather than randomly sampled?**
> Because the optimal cutoff is a direct function of that distribution. A dev
> split with an unrepresentative mixture hands you a cutoff tuned for the wrong
> data, and it will not transfer to the leaderboard.

**Your dev score goes up 3 points but the leaderboard drops 1. What do you do?**
> Stop modelling. Something in the harness disagrees with theirs: a different
> averaging, dropped queries, a doc_id mismatch, or dev leakage. Check
> `src/exp_log.correlation()`, then re-run `evaluate.py --cross-check`. A dev
> split that does not predict the leaderboard is worse than no dev split,
> because it makes you confident while you are wrong.
