# BM25 grid — corpus_article.jsonl (segmenter=none)

Metric: **official** (Precision shown as tiebreak). Cutoff: ratio alpha=0.85.

| k1 \ b | 0.3 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|
| **0.9** | 0.9167 | 0.9167 | 0.9167 | 0.9167 |
| **1.2** | 0.9167 | 0.9167 | 0.9167 | 0.9167 |
| **1.5** | 0.9167 | 0.9167 | 0.9167 | 0.9167 |
| **2.0** | 0.9167 | 0.9167 | 0.9167 | **1.0000** |

Best: **k1=2.0, b=1.0** -> official=1.0000, P=0.6944

## Interpretation (fill in)

- Is the grid flat? If so BM25 is not the bottleneck — move on.
- Where did b land, and does that match the granularity argument?
- Where did k1 land, and what does that say about repetition in this corpus?
