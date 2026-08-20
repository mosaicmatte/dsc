# Reference — depth beyond what the phases require

The phase READMEs teach exactly what that phase needs. These notes go further: the
derivations, the failure modes, and the "why is it like that" answers. **Nothing here is
required to complete a phase.** Read a note when the phase README raised a question it did
not answer, or when a result surprises you and you want to know whether it *should* have.

| Note | Read it when |
|---|---|
| [01 — Metrics](01_metrics.md) | a score moves and you cannot explain why; before arguing about micro vs macro |
| [02 — BM25](02_bm25.md) | the grid search is flat, or `b` lands somewhere unexpected |
| [03 — Dense retrieval](03_dense_retrieval.md) | fine-tuning does not help; before spending a day on hard negatives |
| [04 — Vietnamese NLP](04_vietnamese_nlp.md) | anything text-preprocessing shaped, especially before changing the tokeniser |
| [05 — Reranking & ceilings](05_reranking.md) | deciding between retriever and reranker work |
| [06 — RAG and Task 2](06_rag_task2.md) | the generator produces confident nonsense |
| [07 — Hardware & runtime](07_hardware_runtime.md) | planning a day's work, or something is taking suspiciously long |
| [08 — Reading list](08_reading_list.md) | you want the primary sources for the paper's related-work section |
| [**09 — Official rules and specs**](09_official_rules.md) | **any question about what BTC actually requires — this page overrides every other page** |
| [**10 — What the real data looks like**](10_data_facts.md) | **before you design anything — measured facts about the actual corpus, and the four ways it silently costs you recall** |

Each note ends with **Check yourself** — questions whose answers are in the note. If you
can answer them, you have got what the note is for.
