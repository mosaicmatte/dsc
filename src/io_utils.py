"""JSONL readers/writers for the canonical formats."""
from __future__ import annotations

import gzip
import json
import os
from typing import Any, Dict, Iterable, Iterator, List, Tuple


def _open(path: str, mode: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with _open(path, "r") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln}: bad JSON ({e})") from e


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    n = 0
    with _open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------- corpus / queries ----------

def load_corpus(path: str) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Return (doc_ids, texts, metas) in a stable order."""
    doc_ids, texts, metas = [], [], []
    for r in read_jsonl(path):
        doc_ids.append(str(r["doc_id"]))
        texts.append(r["text"])
        metas.append(r.get("meta", {}))
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError(f"{path}: duplicate doc_id values")
    return doc_ids, texts, metas


def load_queries(path: str) -> List[Dict[str, Any]]:
    qs = [dict(r) for r in read_jsonl(path)]
    for q in qs:
        q["qid"] = str(q["qid"])
        q["relevant"] = [str(d) for d in q.get("relevant", [])]
    return qs


def qrels(queries: List[Dict[str, Any]]) -> Dict[str, set]:
    return {q["qid"]: set(q["relevant"]) for q in queries}


# ---------- runs / predictions ----------

def write_run(path: str, ranked: Dict[str, List[Tuple[str, float]]]) -> int:
    return write_jsonl(
        path,
        (
            {"qid": qid, "ranked": [[d, float(s)] for d, s in lst]}
            for qid, lst in ranked.items()
        ),
    )


def load_run(path: str) -> Dict[str, List[Tuple[str, float]]]:
    out: Dict[str, List[Tuple[str, float]]] = {}
    for r in read_jsonl(path):
        out[str(r["qid"])] = [(str(d), float(s)) for d, s in r["ranked"]]
    return out


def write_predictions(path: str, preds: Dict[str, List[str]]) -> int:
    return write_jsonl(
        path, ({"qid": qid, "predicted": list(docs)} for qid, docs in preds.items())
    )


def load_predictions(path: str) -> Dict[str, List[str]]:
    return {
        str(r["qid"]): [str(d) for d in r["predicted"]] for r in read_jsonl(path)
    }
