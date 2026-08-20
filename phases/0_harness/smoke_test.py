#!/usr/bin/env python3
"""Verify your install before the real data arrives.

Generates a small synthetic Vietnamese legal corpus, runs the whole Phase 0 + 1
chain over it (ingest -> validate -> dev split -> chunk -> BM25 -> grid -> cutoff
sweep -> submission pre-flight), and checks the numbers are sane.

It writes ONLY inside --workdir (default: a temporary directory) and never
touches data/processed or experiments/. Safe to run any time.

    python phases/0_harness/smoke_test.py

If this passes, your environment can run Phases 0-1. Phases 2-4 additionally need
torch / sentence-transformers, which this does not test.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(  # repo root: phases/<n>_<name>/ -> ../..
    os.path.join(os.path.dirname(__file__), "..", "..")))

sys.path.insert(0, os.path.abspath(  # tools/ lives at the repo root
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")))

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILURES.append(name)
    return cond


def make_raw(d):
    """Write the shared fixture (tools/make_fixture.py) into ``d``."""
    from make_fixture import build  # noqa: E402
    corpus, queries, _ = build()
    json.dump(corpus, open(f"{d}/corpus.json", "w"), ensure_ascii=False)
    json.dump({"data": queries}, open(f"{d}/train.json", "w"), ensure_ascii=False)
    return len(corpus), len(queries)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--keep", action="store_true", help="do not delete the workdir")
    a = ap.parse_args()

    work = a.workdir or tempfile.mkdtemp(prefix="dsc_smoke_")
    os.makedirs(work, exist_ok=True)
    raw, proc = f"{work}/raw", f"{work}/processed"
    os.makedirs(raw, exist_ok=True)

    from src import chunking, cutoff, fusion, io_utils, metrics, normalize
    from src.bm25 import BM25Index

    print(f"workdir: {work}\n")
    nd, nq = make_raw(raw)
    print(f"synthetic data: {nd} documents, {nq} queries\n")

    print("1. normalisation")
    check("tone placement hoà->hòa", normalize.normalize("Hoà giải") == "hòa giải")
    check("toàn NOT corrupted", normalize.normalize("toàn bộ") == "toàn bộ")
    check("decree id kept whole",
          "100/2019/nđ-cp" in normalize.tokenize("Nghị định 100/2019/NĐ-CP"))
    check("newlines survive when asked",
          "\n" in normalize.normalize("a\nb", keep_newlines=True))

    print("\n2. ingest")
    sys.argv = ["ingest", "--raw-corpus", f"{raw}/corpus.json",
                "--raw-queries", f"{raw}/train.json", "--out-dir", proc]
    import ingest  # noqa: E402  (same directory)
    ingest.main()
    ok = ingest.validate(f"{proc}/corpus_document.jsonl",
                         (f"{proc}/queries_train.jsonl",))
    check("ingest --validate passes", ok)

    print("\n3. chunking")
    recs = list(io_utils.read_jsonl(f"{proc}/corpus_document.jsonl"))
    arts = chunking.build_corpus(recs, "article")
    check("article split produced chunks", len(arts) > len(recs),
          f"{len(recs)} docs -> {len(arts)} chunks")
    pmap = chunking.parent_map(arts)
    check("chunks map back to parents",
          all(pmap[c["doc_id"]] in {r["doc_id"] for r in recs} for c in arts))

    print("\n4. BM25")
    ids = [c["doc_id"] for c in arts]
    idx = BM25Index([normalize.tokenize(c["text"]) for c in arts], ids)
    queries = io_utils.load_queries(f"{proc}/queries_train.jsonl")
    qtok = {q["qid"]: normalize.tokenize(q["text"]) for q in queries}
    run = idx.batch_search(qtok, top_k=50, progress=False)
    run = chunking.aggregate_to_parent(run, pmap, "max", 50)
    qrels = io_utils.qrels(queries)
    d = metrics.diagnostics(run, qrels, ks=(1, 10, 50))
    check("recall@50 > 0.5", d["recall@50"] > 0.5, f"recall@50={d['recall@50']:.3f}")
    check("k1/b actually change scores",
          idx.search(qtok["q0"], 5, b=0.0) != idx.search(qtok["q0"], 5, b=1.0))

    print("\n5. metrics")
    check("recall is monotone in set size",
          all(metrics.recall_at_k(run, qrels, k) <= metrics.recall_at_k(run, qrels, k + 5)
              for k in (1, 5, 10, 20)))
    p10 = cutoff.apply_to_run(run, rule="top_k", k=10)
    p1 = cutoff.apply_to_run(run, rule="top_k", k=1)
    s10, s1 = metrics.official(p10, qrels), metrics.official(p1, qrels)
    check("larger set: recall up, precision down",
          s10["primary_recall"] >= s1["primary_recall"]
          and s10["tiebreak_precision"] <= s1["tiebreak_precision"],
          f"R {s1['primary_recall']:.3f}->{s10['primary_recall']:.3f}, "
          f"P {s1['tiebreak_precision']:.3f}->{s10['tiebreak_precision']:.3f}")
    check("missing query scored as empty, not skipped",
          metrics.official({}, qrels)["primary_recall"] == 0.0)

    print("\n6. cutoff rules")
    rows = cutoff.sweep(run, qrels, max_k=10)
    check("sweep returns all rules",
          {r["rule"] for r in rows} == {"top_k", "ratio", "gap"})
    var = {len(v) for v in cutoff.apply_to_run(run, rule="ratio", alpha=0.9).values()}
    check("ratio produces variable-length sets", len(var) > 1, f"sizes={sorted(var)}")

    print("\n7. fusion")
    f_rrf = fusion.rrf([run, run], [0.5, 0.5], top_k=50)
    check("rrf preserves recall",
          abs(metrics.recall_at_k(f_rrf, qrels, 50) - d["recall@50"]) < 1e-9)
    f_w = fusion.weighted([run, run], [0.5, 0.5], top_k=50)
    check("weighted fusion runs", len(f_w) == len(run))

    print("\n8. experiment log")
    from src import exp_log
    log = f"{work}/runs.csv"
    for i, (dev, lb) in enumerate([(0.5, 0.48), (0.6, 0.59), (0.7, 0.71)]):
        exp_log.log_run({"run_id": f"t{i}", "dev_official": dev, "leaderboard": lb}, log)
    corr = exp_log.correlation(log)
    check("dev/leaderboard correlation gate works", corr["verdict"] == "healthy",
          f"spearman={corr['spearman']:.2f}")

    print("\n9. optional dependencies")
    for mod, why in [("yaml", "config freezing"), ("matplotlib", "cutoff plots"),
                     ("pyvi", "word segmentation"), ("torch", "phases 2-4"),
                     ("sentence_transformers", "phases 2-4"),
                     ("transformers", "phase 4")]:
        try:
            __import__(mod)
            print(f"  ok       {mod:<24} ({why})")
        except ImportError:
            print(f"  MISSING  {mod:<24} ({why}) — pip install -r requirements.txt")

    if not a.keep and not a.workdir:
        shutil.rmtree(work, ignore_errors=True)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"SMOKE TEST FAILED — {len(FAILURES)} check(s):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("SMOKE TEST PASSED — your environment can run Phases 0-1.")
    print("Phases 2-4 additionally need torch / sentence-transformers (see above).")


if __name__ == "__main__":
    main()
