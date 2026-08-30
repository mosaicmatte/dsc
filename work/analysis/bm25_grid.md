# BM25 grid — corpus_article.jsonl (segmenter=pyvi)

Metric: **official** (Precision shown as tiebreak). Cutoff: top_k alpha=0.85.

| k1 \ b | 0.3 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|
| **0.9** | 0.7804 | 0.7888 | 0.7976 | 0.8016 |
| **1.2** | 0.7856 | 0.7953 | 0.8031 | 0.8072 |
| **1.5** | 0.7882 | 0.7986 | 0.8091 | 0.8107 |
| **2.0** | 0.7886 | 0.8001 | 0.8092 | **0.8134** |

Best: **k1=2.0, b=1.0** -> official=0.8134, P=0.1727

## Interpretation (fill in)

- Is the grid flat? If so BM25 is not the bottleneck — move on.
- Where did b land, and does that match the granularity argument?
- Where did k1 land, and what does that say about repetition in this corpus?
