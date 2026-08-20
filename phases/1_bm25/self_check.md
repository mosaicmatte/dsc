# Phase 1 — self-check answer key

### 1. What does `b = 0` mean, mechanically?

> The length-normalisation factor `k1·(1 − b + b·|d|/avgdl)` collapses to `k1`.
> Document length disappears from the formula entirely: a 3000-word văn bản and a
> 40-word khoản are treated identically.
>
> Consequence: **long documents win**, because they contain more terms by
> accident and therefore match more query terms by accident. `b = 0` is almost
> never right on a corpus with mixed unit lengths.

### 2. Legal articles vary enormously in length. Argue for or against `b < 0.75`.

There is a real case on both sides; what matters is that you know *which regime
you are in*, and that granularity decides it.

> **Against `b < 0.75` (i.e. keep strong normalisation) — at document granularity.**
> A Bộ luật is two orders of magnitude longer than a Nghị định. Without strong
> normalisation the Bộ luật matches nearly every query on accidental term
> overlap, and it drowns short, precisely-relevant documents. Here `b` should be
> at or near 1.0.
>
> **For `b < 0.75` — at điều granularity.** Units are now length-comparable, and
> the remaining variation is *informative*: a long điều genuinely covers more
> ground and is genuinely more likely to contain the answer. Aggressive
> normalisation actively penalises the detailed articles that answer detailed
> questions. Here `b` around 0.3–0.5 often wins.
>
> **The real answer: this is an empirical question and it is question B3 in the
> task list.** The point of being able to argue both sides is that you will
> recognise which story the grid is telling you instead of just reading off a
> number.

### 3. The ratio cutoff returns 1 document for some queries and 30 for others. Bug?

> **No — that is the entire point.** The score-ratio rule adapts to the shape of
> the score distribution per query. A query whose top hit scores far above
> everything else is one the retriever is confident about, and returning a single
> document maximises Precision at no Recall cost. A query with 30 near-tied
> scores is one the retriever cannot separate, and casting a wider net is the
> only way to capture the relevant document.
>
> The clamp is not optional here: `1 ≤ |set| ≤ 5`. An empty set scores zero, and
> so does a 6-document set — BTC's scorer zeroes both metrics for that question.
> `src/cutoff.py` enforces it and raises if you try to set `max_k` above 5.
>
> **What would be a bug:** the set size not varying at all. That means the scores
> are degenerate — usually a sign that the query matched no indexed terms and
> everything scored zero.

---

### Bonus

**Why keep `/`, `.` and `-` inside BM25 tokens?**
> So `100/2019/nđ-cp` survives as one term. Split into `100`, `2019`, `nđ`, `cp`
> it becomes four low-IDF tokens that match thousands of documents; kept whole it
> is a near-unique, maximum-IDF term that identifies exactly one document.

**Why is word segmentation an ablation rather than an obvious win for BM25?**
> Segmentation makes matches more precise (`học_sinh` cannot match a stray `học`)
> but more brittle — one segmenter error breaks a match that a syllable-level
> index would have caught. Dense encoders with PhoBERT backbones *require* it;
> BM25 merely *might* benefit. Measure it.
