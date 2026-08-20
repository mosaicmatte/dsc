#!/usr/bin/env python3
"""Task B3 — raw BTC files -> canonical format.

=============================================================================
THIS IS THE ONLY FILE IN THE REPO THAT KNOWS BTC'S RAW LAYOUT. Keep it that way.
When BTC ships a data patch, exactly one file needs editing.
=============================================================================

WHAT YOU NEED TO DO
-------------------
1. Put the raw files in ``data/raw/``.
2. Run with ``--inspect`` first. It prints the detected field names and one
   sample record without writing anything.
3. If the auto-detected mapping is right, drop ``--inspect`` and run it.
   If it is wrong, pass the field names explicitly (``--text-field noi_dung``
   etc.). Do not "fix" it by editing the JSON by hand — the mapping must be
   reproducible from the command line.
4. Run ``--validate`` and fix anything it complains about.

HOW IT WORKS
------------
Canonical output (see root README):
  data/processed/corpus_document.jsonl  {"doc_id","text","meta"}
  data/processed/queries_train.jsonl    {"qid","text","relevant":[doc_id,...]}

Usage:
  # inspect first — prints detected fields and one record, writes nothing
  python phases/0_harness/ingest.py --inspect \
      --raw-corpus data/raw/selected-contexts.zip

  # Task 1
  python phases/0_harness/ingest.py \
      --raw-corpus  data/raw/selected-contexts.zip \
      --raw-queries data/raw/train.json \
      --raw-test    data/raw/public_official.json

  # Task 2 (same corpus; --task 2 also picks up the gold prose answer)
  python phases/0_harness/ingest.py --task 2 \
      --raw-queries data/raw/task2_train.json

  python phases/0_harness/ingest.py --validate
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(  # repo root: phases/<n>_<name>/ -> ../..
    os.path.join(os.path.dirname(__file__), "..", "..")))

from src import io_utils  # noqa: E402
from src.normalize import normalize  # noqa: E402

PROC = "data/processed"

# Candidate names, most-likely first. Extend these lists rather than editing
# the detection logic.
# Ordered most-likely-first. The DSC2026 shapes are listed first because they are
# what BTC actually ships (confirmed from the Task 1/2 overview documents):
#   corpus  context_*.json : {"id": 740, "name": ..., "link": ..., "passage": ...}
#   queries train.json     : {"147194": {"question": "...", "answer": ["177504"]}}
ID_FIELDS = ["id", "doc_id", "law_id", "article_id", "cid", "_id", "index"]
TEXT_FIELDS = ["passage", "text", "content", "noi_dung", "article_text", "body"]
TITLE_FIELDS = ["name", "title", "tieu_de", "law_title", "heading"]
QID_FIELDS = ["__key__", "qid", "question_id", "id", "query_id", "sample_id"]
QUERY_FIELDS = ["question", "query", "cau_hoi", "text", "q"]
# For Task 1 "answer" holds the list of relevant document ids.
REL_FIELDS = ["answer", "relevant_id", "relevant", "relevant_laws", "positive",
              "labels", "relevant_articles", "answer_id", "context_id", "gold"]


def load_raw(path: str):
    """BTC ships several shapes. All of them land here.

    * ``selected-contexts.zip`` or a directory of ``context_*.json`` — the corpus.
      Each file holds one record or a list of records.
    * ``train.json`` / ``public_official.json`` — a JSON OBJECT keyed by question
      id: ``{"147194": {"question": ..., "answer": [...]}}``. The key becomes
      ``__key__`` on each record and is picked up as the qid.
    * a plain list of records, or ``{"data": [...]}``.
    * ``.jsonl``.
    """
    if os.path.isdir(path) or path.endswith(".zip"):
        return _load_context_files(path)
    if path.endswith(".jsonl"):
        return list(io_utils.read_jsonl(path))
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # unwrap {"data": [...]} / {"items": [...]} style containers
        for k in ("data", "items", "corpus", "questions", "samples"):
            if isinstance(data.get(k), list):
                return data[k]
        # or a plain {id: record} mapping
        if all(isinstance(v, (dict, str)) for v in data.values()):
            return [{"__key__": k, **(v if isinstance(v, dict) else {"text": v})}
                    for k, v in data.items()]
        raise ValueError(f"{path}: dict with no recognisable record list")
    return data


def _load_context_files(path: str):
    """Read every context_*.json out of a directory or a zip."""
    import zipfile

    recs = []
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist()
                     if n.endswith(".json") and not n.startswith("__MACOSX")]
            if not names:
                raise SystemExit(f"{path}: no .json members found")
            for n in sorted(names):
                with z.open(n) as f:
                    obj = json.load(f)
                recs.extend(obj if isinstance(obj, list) else [obj])
    else:
        names = sorted(fn for fn in os.listdir(path) if fn.endswith(".json"))
        if not names:
            raise SystemExit(f"{path}: no .json files found")
        for n in names:
            with open(os.path.join(path, n), encoding="utf-8") as f:
                obj = json.load(f)
            recs.extend(obj if isinstance(obj, list) else [obj])
    print(f"  read {len(recs)} records from {len(names)} context file(s)")
    return recs


def detect(records, candidates, required=True, what=""):
    keys = set()
    for r in records[:50]:
        keys |= set(r.keys())
    for c in candidates:
        if c in keys:
            return c
    if required:
        raise SystemExit(
            f"could not detect the {what} field.\n"
            f"  available keys: {sorted(keys)}\n"
            f"  pass it explicitly, e.g. --{what.replace('_','-')} <key>\n"
            f"  (and add it to {what.upper()}_FIELDS in this file)")
    return None


def as_list(v):
    """Relevance labels appear as a scalar, a list, or a list of dicts."""
    if v is None:
        return []
    if isinstance(v, (str, int)):
        return [str(v)]
    if isinstance(v, list):
        out = []
        for x in v:
            if isinstance(x, dict):
                # e.g. {"law_id": "...", "article_id": "..."} -> composite key
                out.append("#".join(str(x[k]) for k in sorted(x)))
            else:
                out.append(str(x))
        return out
    raise ValueError(f"cannot interpret relevance value: {v!r}")


def ingest_corpus(path, id_field, text_field, title_field, out):
    recs = load_raw(path)
    id_field = id_field or detect(recs, ID_FIELDS, what="id_field")
    text_field = text_field or detect(recs, TEXT_FIELDS, what="text_field")
    title_field = title_field or detect(recs, TITLE_FIELDS, False, "title_field")
    rows, seen = [], set()
    for i, r in enumerate(recs):
        did = str(r.get(id_field, r.get("__key__", i)))
        if did in seen:
            # Duplicate ids are common when the corpus is article-level and the
            # id is only unique within a law. Make it unique, keep the original.
            did = f"{did}::{i}"
        seen.add(did)
        title = str(r.get(title_field, "") or "") if title_field else ""
        body = str(r.get(text_field, "") or "")
        meta = {k: v for k, v in r.items()
                if k not in {id_field, text_field, "__key__"}
                and isinstance(v, (str, int, float, bool))}
        if title:
            meta["title"] = title
        # keep_newlines: corpus text is chunked later by src.chunking, whose
        # regexes anchor on line starts. Flattening here would silently break it.
        rows.append({"doc_id": did,
                     "text": normalize(f"{title}\n{body}".strip(), lower=False,
                                       keep_newlines=True),
                     "meta": meta})
    n = io_utils.write_jsonl(out, rows)
    print(f"[corpus] {n} records -> {out}  "
          f"(id={id_field}, text={text_field}, title={title_field})")
    return rows


ANSWER_FIELDS = ["answer", "cau_tra_loi", "answer_text", "response", "output",
                 "label", "gold_answer"]


def ingest_queries(path, qid_field, query_field, rel_field, out, split,
                   answer_field=None, task=1):
    """Queries for both tasks. Task 2 additionally carries a gold ``answer``.

    Any field we do not recognise is preserved verbatim on the record, so a
    Task 2 answer, a difficulty label or a category tag survives ingest even if
    nobody thought to add it to the field lists above.
    """
    recs = load_raw(path)
    qid_field = qid_field or detect(recs, QID_FIELDS, what="qid_field")
    query_field = query_field or detect(recs, QUERY_FIELDS, what="query_field")
    if task == 2:
        # In Task 2 "answer" is the gold PROSE answer, not a list of doc ids, so
        # it must not be consumed as the relevance field.
        answer_field = answer_field or detect(recs, ANSWER_FIELDS,
                                              "train" in split, "answer_field")
        rel_candidates = [f for f in REL_FIELDS if f != answer_field]
        rel_field = rel_field or detect(recs, rel_candidates, False, "rel_field")
    else:
        rel_field = rel_field or detect(recs, REL_FIELDS, split == "train",
                                        "rel_field")
    known = {qid_field, query_field, rel_field, answer_field, "__key__"}
    rows = []
    for i, r in enumerate(recs):
        row = {
            "qid": str(r.get(qid_field, i)),
            "text": normalize(str(r.get(query_field, "")), lower=False),
            "relevant": as_list(r.get(rel_field)) if rel_field else [],
        }
        if answer_field and r.get(answer_field) is not None:
            row["answer"] = normalize(str(r[answer_field]), lower=False)
        # keep anything else we did not explicitly map
        for k, v in r.items():
            if k not in known and k not in row and isinstance(v, (str, int, float, bool)):
                row[k] = v
        rows.append(row)
    n = io_utils.write_jsonl(out, rows)
    print(f"[queries:{split}] {n} records -> {out}  "
          f"(qid={qid_field}, query={query_field}, rel={rel_field}"
          + (f", answer={answer_field}" if answer_field else "") + ")")
    if task == 2 and answer_field:
        have = sum(1 for r in rows if r.get("answer"))
        print(f"                 {have}/{n} records carry a gold answer")
    return rows


def validate(corpus_path=f"{PROC}/corpus_document.jsonl",
             query_paths=(f"{PROC}/queries_train.jsonl",
                          f"{PROC}/task2_train.jsonl")):
    """The checks that catch a broken harness before it costs you a week."""
    ok = True
    doc_ids, texts, _ = io_utils.load_corpus(corpus_path)
    ids = set(doc_ids)
    print(f"corpus: {len(doc_ids)} docs, {len(ids)} unique ids")
    empty = sum(1 for t in texts if not t.strip())
    if empty:
        print(f"  FAIL: {empty} documents have empty text")
        ok = False
    for qp in query_paths:
        if not os.path.exists(qp):
            continue
        qs = io_utils.load_queries(qp)
        n_rel = sum(len(q["relevant"]) for q in qs)
        dangling = {d for q in qs for d in q["relevant"] if d not in ids}
        noq = [q["qid"] for q in qs if not q["text"].strip()]
        norel = [q["qid"] for q in qs if not q["relevant"]]
        print(f"{qp}: {len(qs)} queries, {n_rel} labels, "
              f"{n_rel/max(len(qs),1):.2f} rel/query")
        if dangling:
            # THE most important check: a gold id that is not in the corpus means
            # your doc_id construction disagrees with BTC's labels. Recall is
            # capped below 1.0 and no model can fix it.
            print(f"  FAIL: {len(dangling)} gold ids not present in corpus, "
                  f"e.g. {list(dangling)[:5]}")
            ok = False
        if noq:
            print(f"  FAIL: {len(noq)} queries with empty text, e.g. {noq[:5]}")
            ok = False
        if norel:
            print(f"  WARN: {len(norel)} queries with no relevant docs "
                  f"(check whether this is a labelling gap or intentional)")
    print("VALIDATE:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-corpus")
    ap.add_argument("--raw-queries")
    ap.add_argument("--raw-test")
    ap.add_argument("--split", default="train")
    ap.add_argument("--task", type=int, default=1, choices=[1, 2],
                    help="2 also ingests the gold answer field (LegalQA)")
    ap.add_argument("--answer-field", default=None)
    ap.add_argument("--id-field"); ap.add_argument("--text-field")
    ap.add_argument("--title-field"); ap.add_argument("--qid-field")
    ap.add_argument("--query-field"); ap.add_argument("--rel-field")
    ap.add_argument("--out-dir", default=PROC)
    ap.add_argument("--inspect", action="store_true",
                    help="print detected fields and a sample record, write nothing")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()

    if a.validate:
        sys.exit(0 if validate() else 1)

    if a.inspect:
        for label, path in (("corpus", a.raw_corpus), ("queries", a.raw_queries),
                            ("test", a.raw_test)):
            if not path:
                continue
            recs = load_raw(path)
            print(f"\n=== {label}: {path} — {len(recs)} records ===")
            print("keys:", sorted({k for r in recs[:50] for k in r}))
            print(json.dumps(recs[0], ensure_ascii=False, indent=2)[:1200])
        return

    os.makedirs(a.out_dir, exist_ok=True)
    if a.raw_corpus:
        ingest_corpus(a.raw_corpus, a.id_field, a.text_field, a.title_field,
                      f"{a.out_dir}/corpus_document.jsonl")
    stem = "queries" if a.task == 1 else "task2"
    if a.raw_queries:
        ingest_queries(a.raw_queries, a.qid_field, a.query_field, a.rel_field,
                       f"{a.out_dir}/{stem}_{a.split}.jsonl", a.split,
                       a.answer_field, a.task)
    if a.raw_test:
        ingest_queries(a.raw_test, a.qid_field, a.query_field, None,
                       f"{a.out_dir}/{stem}_public_test.jsonl", "public_test",
                       None, a.task)
    print("\nnow run:  python phases/0_harness/ingest.py --validate")
    if a.task == 2:
        print("then:     python phases/0_harness/build_dev_split.py "
              f"--queries {a.out_dir}/task2_train.jsonl --prefix task2")


if __name__ == "__main__":
    main()
