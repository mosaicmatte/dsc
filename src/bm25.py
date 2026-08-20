"""BM25 over an inverted index.

Written by hand rather than wrapping ``rank_bm25`` for one reason: the index is
built ONCE and ``k1``/``b`` are supplied at *scoring* time. A 4x4 grid search
therefore costs one tokenisation pass, not sixteen. (``rank_bm25`` bakes the
parameters into the constructor.)

The scoring function
--------------------
    score(q, d) = sum_t idf(t) * qtf(t) * tf(t,d)*(k1+1)
                  / ( tf(t,d) + k1*(1 - b + b*|d|/avgdl) )

    idf(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

* **k1** controls term-frequency saturation. k1=0 makes term frequency binary
  (present/absent). Large k1 -> raw counts. Legal text repeats defined terms, so
  saturation is helpful: a document is not 10x more relevant for saying
  "người lao động" ten times.
* **b** controls length normalisation. b=0 = none (long documents win, because
  they contain more terms by accident). b=1 = full (score divided by relative
  length). Legal articles vary enormously in length -- a two-line khoản next to
  a three-page điều -- so b is the parameter to sweep hardest.
* The ``log(1 + ...)`` IDF variant keeps IDF positive for terms appearing in
  >half the corpus, unlike the classic Robertson form which goes negative and
  lets a common term *subtract* score.
"""
from __future__ import annotations

import pickle
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np


class BM25Index:
    def __init__(self, tokenized_docs: Sequence[Sequence[str]], doc_ids: Sequence[str]):
        if len(tokenized_docs) != len(doc_ids):
            raise ValueError("tokenized_docs and doc_ids differ in length")
        self.doc_ids = list(doc_ids)
        self.N = len(doc_ids)
        self.doc_len = np.array([len(d) for d in tokenized_docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.N else 0.0

        postings: Dict[str, List[Tuple[int, int]]] = {}
        for i, toks in enumerate(tokenized_docs):
            for term, tf in Counter(toks).items():
                postings.setdefault(term, []).append((i, tf))

        self.postings: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.idf: Dict[str, float] = {}
        for term, plist in postings.items():
            idx = np.fromiter((p[0] for p in plist), dtype=np.int32, count=len(plist))
            tfs = np.fromiter((p[1] for p in plist), dtype=np.float32, count=len(plist))
            self.postings[term] = (idx, tfs)
            df = len(plist)
            self.idf[term] = float(np.log(1.0 + (self.N - df + 0.5) / (df + 0.5)))

    # ------------------------------------------------------------------
    def score(self, query_tokens: Sequence[str], k1: float = 1.2,
              b: float = 0.75) -> np.ndarray:
        scores = np.zeros(self.N, dtype=np.float32)
        norm = k1 * (1.0 - b + b * self.doc_len / (self.avgdl or 1.0))
        for term, qtf in Counter(query_tokens).items():
            post = self.postings.get(term)
            if post is None:
                continue
            idx, tfs = post
            contrib = self.idf[term] * qtf * (tfs * (k1 + 1.0)) / (tfs + norm[idx])
            np.add.at(scores, idx, contrib)
        return scores

    def search(self, query_tokens: Sequence[str], top_k: int = 100,
               k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
        s = self.score(query_tokens, k1, b)
        top_k = min(top_k, self.N)
        if top_k <= 0:
            return []
        part = np.argpartition(-s, top_k - 1)[:top_k]
        part = part[np.argsort(-s[part], kind="stable")]
        return [(self.doc_ids[i], float(s[i])) for i in part if s[i] > 0]

    def batch_search(self, queries: Dict[str, Sequence[str]], top_k: int = 100,
                     k1: float = 1.2, b: float = 0.75,
                     progress: bool = True) -> Dict[str, List[Tuple[str, float]]]:
        items = list(queries.items())
        if progress:
            try:
                from tqdm import tqdm
                items = tqdm(items, desc=f"bm25 k1={k1} b={b}")
            except ImportError:
                pass
        return {qid: self.search(toks, top_k, k1, b) for qid, toks in items}

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> "BM25Index":
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        return (f"BM25Index(N={self.N}, vocab={len(self.postings)}, "
                f"avgdl={self.avgdl:.1f})")
