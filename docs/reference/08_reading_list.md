# 08 — Reading list

For the paper's related-work section, and for anyone who wants the primary sources rather
than my summaries. Ordered by how directly each one bears on what you are building.

> Verify every citation before it goes in the paper — check the venue, year and author list
> against the actual paper. Titles and authors below are given from memory and are a starting
> point for your search, not a bibliography you should copy verbatim.

---

## Directly load-bearing

**BM25 and probabilistic retrieval**
Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (Foundations
and Trends in IR, 2009). The definitive treatment of where `k1` and `b` come from. Read §3
if you read nothing else; it is the source of the length-normalisation argument in
`docs/reference/02_bm25.md`.

**Dense passage retrieval**
Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question Answering* (EMNLP 2020).
The paper that established bi-encoder retrieval with in-batch negatives as a standard
recipe. Its negative-sampling ablation is the direct ancestor of Phase 2's three rounds.

**Sentence embeddings**
Reimers & Gurevych, *Sentence-BERT* (EMNLP 2019). Why you pool a transformer into one vector
and train it with a similarity objective — the foundation of the `sentence-transformers`
library the repo uses.

**Contrastive learning**
van den Oord et al., *Representation Learning with Contrastive Predictive Coding* (2018).
The origin of the InfoNCE loss. Read for the loss derivation; the domain is not ours.

**Hard negatives**
Xiong et al., *ANCE: Approximate Nearest Neighbor Negative Contrastive Learning* (ICLR 2021)
— mining negatives from the model's own index, iteratively. This is exactly Phase 2 round 3.
Qu et al., *RocketQA* (NAACL 2021) — treats the false-negative problem head-on with
cross-encoder denoising. Read it before deciding how aggressive `--skip-top` should be.

**Rank fusion**
Cormack, Clarke & Buettcher, *Reciprocal Rank Fusion outperforms Condorcet and individual
rank learning methods* (SIGIR 2009). Short. The source of `K = 60`.

**Multilingual retrieval**
Chen et al., *BGE-M3* (2024). The backbone behind `Vietnamese_Embedding`,
`Vietnamese_Reranker` and `bge-reranker-v2-m3`. Relevant for the long-context and
multi-granularity claims.

**Vietnamese pretraining**
Nguyen & Nguyen, *PhoBERT: Pre-trained language models for Vietnamese* (Findings of EMNLP
2020). **Read the preprocessing section specifically** — it is the authority for why
PhoBERT-backbone models need segmented input, which is the highest-value fact in this repo.

**Long-context reading**
Liu et al., *Lost in the Middle: How Language Models Use Long Contexts* (TACL 2024). The
mechanism behind Phase 4's non-monotonic context-size result. Cite it when your top-5 result
comes in below top-3.

**Parameter-efficient fine-tuning**
Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (ICLR 2022). Note what it
actually claims — trainable-parameter reduction, not inference-parameter reduction. Relevant
to the eligibility argument in `docs/reference/06_rag_task2.md` §6.

## Useful background

- Khattab & Zaharia, *ColBERT* (SIGIR 2020) — late interaction, the middle ground between
  bi- and cross-encoders. Worth a paragraph in related work as an approach you did not take,
  with the parameter budget as the reason.
- Thakur et al., *BEIR* (NeurIPS Datasets & Benchmarks 2021) — zero-shot retrieval
  benchmarking; useful framing for why zero-shot numbers deserve to be ablation rows.
- Nogueira & Cho, *Passage Re-ranking with BERT* (2019) — the original neural reranking
  result; the cleanest statement of the retrieve-then-rerank argument.
- Lewis et al., *Retrieval-Augmented Generation* (NeurIPS 2020) — the RAG framing for Task 2.

## Vietnamese legal IR specifically

Search for prior work on **Zalo AI Challenge legal text retrieval** and the **ALQAC**
(Automated Legal Question Answering Competition) series — both are Vietnamese legal retrieval
shared tasks with published participant papers, and they are the closest prior art to this
competition. Several of the models you are registering were trained on that data.

This is the section of your related work that reviewers will care most about, because it is
where you can say what is genuinely different about DSC@UIT 2026: the 4B ceiling and the
variable-length answer set.

## How to use this list in the paper

Do not cite everything. Cite what you actually relied on:

| Your claim | Cite |
|---|---|
| BM25 formulation and parameter roles | Robertson & Zaragoza |
| bi-encoder + in-batch negatives | Karpukhin et al.; Reimers & Gurevych |
| iterative hard-negative mining | Xiong et al. (ANCE); Qu et al. (RocketQA) for false negatives |
| RRF with K=60 | Cormack et al. |
| retrieve-then-rerank | Nogueira & Cho |
| segmentation requirement | Nguyen & Nguyen (PhoBERT) |
| non-monotonic context size | Liu et al. |
| LoRA does not reduce inference parameters | Hu et al. |

A related-work section that cites eight papers you genuinely used beats one that cites thirty
you skimmed.
