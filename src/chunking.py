"""Corpus granularity -- ablation #1, and usually worth more than model choice.

The tension
-----------
Vietnamese legal documents nest: văn bản > chương > mục > điều > khoản > điểm.

* Index whole **văn bản**: every relevant article is inside some indexed unit, so
  recall of the *unit* is easy -- but the unit is huge, BM25 length
  normalisation fights you, and a bi-encoder must compress a 3000-token
  document into one vector. Precision suffers badly.
* Index **điều**: units are the size of an actual answer, embeddings are sharp,
  precision is much better -- but if the gold label is at văn bản level you must
  map chunks back up before scoring, and if an answer spans several điều you now
  need several hits instead of one.

Which is right depends on what BTC's gold labels point at. Phase 0 answers that
from the data overview + evaluation code; this module then supports BOTH so the
comparison is a measurement, not an argument.

``aggregate_to_parent`` is the bridge: retrieve at fine granularity (sharp
matching), then score at the granularity the labels use.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

# Each pattern MUST expose a group named ``label`` holding the bare identifier
# (the number/letter), because that is what ends up in the chunk's doc_id.
# "Điều 12.", "Điều 12 -", "ĐIỀU 12:"  at the start of a line
ARTICLE_RE = re.compile(r"(?im)^[ \t]*điều\s+(?P<label>\d+)\s*[\.\-:–]?", re.UNICODE)
# "1.", "2)"  at the start of a line -> khoản
CLAUSE_RE = re.compile(r"(?m)^[ \t]*(?P<label>\d{1,2})\s*[\.\)]\s+", re.UNICODE)
# "a)", "b)"  -> điểm
POINT_RE = re.compile(r"(?m)^[ \t]*(?P<label>[a-zđ])\s*\)\s+", re.UNICODE)


def _split_on(text: str, pattern: re.Pattern) -> List[Tuple[str, str]]:
    """Split ``text`` at each match. Returns [(label, chunk_text), ...].

    Text before the first match is returned under the label ``"_preamble"`` --
    dropping it is a classic silent recall leak, because the preamble of a văn
    bản carries its title and scope.
    """
    matches = list(pattern.finditer(text))
    if not matches:
        return [("_whole", text)]
    out: List[Tuple[str, str]] = []
    if matches[0].start() > 0:
        pre = text[: matches[0].start()].strip()
        if pre:
            out.append(("_preamble", pre))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        label = (m.groupdict().get("label") or m.group(0)).strip()
        out.append((label, text[m.start():end].strip()))
    return out


def split_articles(text: str) -> List[Tuple[str, str]]:
    """Split a văn bản into điều."""
    return _split_on(text, ARTICLE_RE)


def split_clauses(text: str) -> List[Tuple[str, str]]:
    """Split a điều into khoản."""
    return _split_on(text, CLAUSE_RE)


def sliding_window(text: str, size: int = 256, stride: int = 192) -> List[str]:
    """Whitespace-token windows, for units longer than the encoder's max length.

    Overlap (stride < size) exists so a relevant sentence is never cut in half
    across two windows with neither half retaining enough context to match.
    """
    words = text.split()
    if len(words) <= size:
        return [text]
    return [" ".join(words[i:i + size])
            for i in range(0, max(len(words) - size + stride, 1), stride)]


# --------------------------------------------------------------------------

def build_corpus(
    records: Iterable[Dict[str, Any]],
    granularity: str = "document",
    window: int | None = None,
    stride: int = 192,
    min_chars: int = 30,
) -> List[Dict[str, Any]]:
    """Canonical corpus records -> chunked canonical corpus records.

    granularity: ``document`` | ``article`` | ``clause``
    Each output record keeps ``meta.parent_id`` pointing at the input doc_id, so
    ``aggregate_to_parent`` can undo the split at scoring time.
    """
    if granularity not in {"document", "article", "clause"}:
        raise ValueError(f"unknown granularity: {granularity!r}")
    out: List[Dict[str, Any]] = []
    for rec in records:
        pid = str(rec["doc_id"])
        text = rec.get("text", "") or ""
        meta = dict(rec.get("meta", {}))
        title = meta.get("title", "")

        if granularity == "document":
            pieces = [(pid, text)]
        else:
            arts = split_articles(text)
            if granularity == "article":
                pieces = [(f"{pid}#dieu{lab}", t) for lab, t in arts]
            else:
                pieces = []
                for lab, atext in arts:
                    for clab, ctext in split_clauses(atext):
                        pieces.append((f"{pid}#dieu{lab}#khoan{clab}", ctext))

        # A real Vietnamese legal document can carry the same article number many
        # times over — an amending decree restates "Điều 1" once per law it
        # touches (document 224467 does it 49 times). Labels are therefore NOT
        # unique within a document, and a bare "{pid}#dieu1" collides. Suffix the
        # repeats so every chunk keeps a distinct id; parent_id is untouched, so
        # aggregation back to the document is unaffected.
        seen: Dict[str, int] = {}
        for cid, ctext in pieces:
            n = seen.get(cid, 0)
            seen[cid] = n + 1
            if n:
                cid = f"{cid}@{n}"
            ctext = ctext.strip()
            if len(ctext) < min_chars:
                continue
            # Prepending the parent title recovers context a bare khoản loses
            # ("...trong trường hợp quy định tại khoản 1" means nothing alone).
            body = f"{title}\n{ctext}" if title and granularity != "document" else ctext
            if window:
                for wi, w in enumerate(sliding_window(body, window, stride)):
                    out.append({"doc_id": f"{cid}::w{wi}", "text": w,
                                "meta": {**meta, "parent_id": pid,
                                         "granularity": granularity}})
            else:
                out.append({"doc_id": cid, "text": body,
                            "meta": {**meta, "parent_id": pid,
                                     "granularity": granularity}})
    return out


def parent_map(corpus: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    return {r["doc_id"]: r.get("meta", {}).get("parent_id", r["doc_id"])
            for r in corpus}


def aggregate_to_parent(
    run: Dict[str, List[Tuple[str, float]]],
    pmap: Dict[str, str],
    how: str = "max",
    top_k: int = 100,
) -> Dict[str, List[Tuple[str, float]]]:
    """Collapse a chunk-level ranking to a parent-document ranking.

    ``max``  -- a document is as relevant as its single best chunk. Correct
                default: one strongly matching điều IS the reason the văn bản is
                relevant.
    ``sum``  -- rewards documents with many weakly matching chunks. This
                systematically favours long documents; only use it if dev says so.
    ``mean`` -- length-neutral but dilutes a single decisive match.
    """
    out: Dict[str, List[Tuple[str, float]]] = {}
    for qid, ranked in run.items():
        acc: Dict[str, List[float]] = {}
        for doc, score in ranked:
            acc.setdefault(pmap.get(doc, doc), []).append(score)
        if how == "max":
            merged = {p: max(v) for p, v in acc.items()}
        elif how == "sum":
            merged = {p: sum(v) for p, v in acc.items()}
        elif how == "mean":
            merged = {p: sum(v) / len(v) for p, v in acc.items()}
        else:
            raise ValueError(f"unknown aggregation: {how!r}")
        out[qid] = sorted(merged.items(), key=lambda x: -x[1])[:top_k]
    return out


def stats(corpus: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    lens = [len(r["text"].split()) for r in corpus]
    lens_sorted = sorted(lens)
    n = len(lens) or 1
    def pct(p): return lens_sorted[min(int(p * n), n - 1)] if lens_sorted else 0
    return {
        "n_chunks": len(corpus),
        "n_parents": len({r.get("meta", {}).get("parent_id", r["doc_id"])
                          for r in corpus}),
        "len_mean": sum(lens) / n,
        "len_p50": pct(0.50), "len_p90": pct(0.90),
        "len_p99": pct(0.99), "len_max": max(lens) if lens else 0,
    }
