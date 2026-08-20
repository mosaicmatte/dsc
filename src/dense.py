"""Dense encoders: model registry, encoding, and vector search.

THE MODEL REGISTRY IS THE POINT OF THIS FILE
--------------------------------------------
Every Vietnamese encoder falls into one of two families, and feeding one the
other's input format costs 5-15 points **without raising an exception**:

  PhoBERT backbone -> input MUST be word-segmented ("người_lao_động")
  BGE-M3 / XLM-R   -> input MUST NOT be segmented

``REGISTRY`` records this per model so no call site has to remember. Add a model
here before using it anywhere, and register it with BTC (see
phases/0_harness/00_model_registration.md) before submitting with it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from . import normalize


@dataclass
class ModelSpec:
    name: str
    backbone: str
    segmented: bool          # does this model require pre-segmented input?
    max_seq: int
    params: int
    # Is it on BTC's public approved list? A submission using an unapproved model
    # is NOT recognised. List:
    # https://docs.google.com/spreadsheets/d/1c5jzsYezWho1WGLRfMKWOaFPLIk_GTnXP5vV8AOWM2Q
    approved: bool = True
    query_prefix: str = ""   # some models expect an instruction prefix
    doc_prefix: str = ""
    notes: str = ""
    tags: List[str] = field(default_factory=list)


# TODO(BLOCKER/phase0-B0): verify `segmented`, `max_seq` and `params` for every
# entry against its model card before first use. `segmented` is the dangerous one:
# a wrong value costs 5-15 points and raises no exception. Check the card's usage
# example -- if it shows `ViTokenizer.tokenize(text)` or underscored words, it is
# segmented=True. Also check for a required query prefix/instruction and put it in
# `query_prefix` (several retrieval models expect one and quietly underperform
# without it).
REGISTRY: Dict[str, ModelSpec] = {
    m.name: m for m in [
        ModelSpec("bkai-foundation-models/vietnamese-bi-encoder", "PhoBERT-base",
                  True, 256, 135_000_000,
                  notes="trained partly on Zalo legal retrieval; small and fast; "
                        "256-token limit makes chunking mandatory",
                  tags=["bi-encoder"]),
        ModelSpec("dangvantuan/vietnamese-embedding", "PhoBERT-base",
                  True, 256, 135_000_000, approved=False,
                  notes="NOT on BTC's approved list as of 20/08 — register it "
                        "before use, or use another bi-encoder",
                  tags=["bi-encoder"]),
        ModelSpec("AITeamVN/Vietnamese_Embedding", "BGE-M3",
                  False, 8192, 568_000_000,
                  notes="long context suits full điều without truncation",
                  tags=["bi-encoder"]),
        ModelSpec("BAAI/bge-m3", "XLM-R-large", False, 8192, 568_000_000,
                  notes="multilingual baseline for contrast", tags=["bi-encoder"]),
        ModelSpec("AITeamVN/Vietnamese_Reranker", "BGE-M3",
                  False, 8192, 568_000_000, tags=["cross-encoder"]),
        ModelSpec("BAAI/bge-reranker-v2-m3", "XLM-R-large",
                  False, 8192, 568_000_000, tags=["cross-encoder"]),
        ModelSpec("itdainb/PhoRanker", "PhoBERT-base",
                  True, 256, 135_000_000, tags=["cross-encoder"]),
        ModelSpec("namdp-ptit/ViRanker", "PhoBERT-base",
                  True, 256, 135_000_000, tags=["cross-encoder"]),
    ]
}


def approved_or_raise(name: str) -> ModelSpec:
    """Fail loudly before a run rather than after an invalid submission."""
    sp = spec(name)
    if not sp.approved:
        raise ValueError(
            f"{name!r} is NOT on BTC's approved model list. A submission using it "
            f"is not recognised. Register it at https://forms.gle/HWE7tcxzWq63Kxv28 "
            f"and wait for approval, or pick an approved model — see "
            f"phases/0_harness/00_model_registration.md")
    return sp


def spec(name: str) -> ModelSpec:
    if name not in REGISTRY:
        raise KeyError(
            f"{name!r} is not in src/dense.REGISTRY.\n"
            "Add a ModelSpec for it — in particular decide `segmented`, because "
            "getting that wrong degrades scores silently rather than crashing.")
    return REGISTRY[name]


def prepare(texts: Sequence[str], model_name: str, is_query: bool = False) -> List[str]:
    """Normalise + segment (or not) + prefix, according to the model's spec."""
    sp = spec(model_name)
    pre = sp.query_prefix if is_query else sp.doc_prefix
    return [pre + normalize.encoder_text(t, sp.segmented) for t in texts]


# --------------------------------------------------------------------------

def load_encoder(model_name: str, device: str | None = None, **kw):
    from sentence_transformers import SentenceTransformer  # type: ignore
    sp = spec(model_name)
    m = SentenceTransformer(model_name, device=device, **kw)
    m.max_seq_length = min(m.max_seq_length or sp.max_seq, sp.max_seq)
    return m


def encode(model, texts: Sequence[str], model_name: str, is_query: bool = False,
           batch_size: int = 64, normalize_embeddings: bool = True,
           show_progress: bool = True):
    """Encode with the model's required input format applied automatically.

    ``normalize_embeddings=True`` makes the dot product a cosine similarity, so
    downstream code can use one similarity function everywhere. Keep it on
    unless you have measured that it hurts.
    """
    prepped = prepare(texts, model_name, is_query=is_query)
    return model.encode(prepped, batch_size=batch_size,
                        normalize_embeddings=normalize_embeddings,
                        show_progress_bar=show_progress,
                        convert_to_numpy=True)


def search(query_emb, doc_emb, doc_ids: Sequence[str], qids: Sequence[str],
           top_k: int = 100, batch: int = 256
           ) -> Dict[str, List[Tuple[str, float]]]:
    """Exact brute-force search. Fine up to ~1M docs; swap in FAISS beyond that.

    Exact search is the right default for a competition corpus: an approximate
    index introduces a recall loss you then spend days mistaking for a model
    problem.
    """
    import numpy as np

    out: Dict[str, List[Tuple[str, float]]] = {}
    k = min(top_k, len(doc_ids))
    for i in range(0, len(qids), batch):
        sims = query_emb[i:i + batch] @ doc_emb.T
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        for r, qid in enumerate(qids[i:i + batch]):
            cols = idx[r]
            cols = cols[np.argsort(-sims[r, cols], kind="stable")]
            out[qid] = [(doc_ids[c], float(sims[r, c])) for c in cols]
    return out


def build_faiss(doc_emb, use_gpu: bool = False):
    """Flat inner-product index. Embeddings must already be L2-normalised."""
    import faiss  # type: ignore
    index = faiss.IndexFlatIP(doc_emb.shape[1])
    if use_gpu:
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(doc_emb.astype("float32"))
    return index
