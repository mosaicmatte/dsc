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


def _raises(fn, exc):
    try:
        fn()
        return False
    except exc:
        return True


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILURES.append(name)
    return cond


def make_raw(d):
    """Write the shared fixture (tools/make_fixture.py) into ``d``, in BTC shapes."""
    import zipfile
    from make_fixture import build  # noqa: E402
    corpus, queries, _ = build()
    os.makedirs(f"{d}/contexts", exist_ok=True)
    for i, rec in enumerate(corpus):
        json.dump(rec, open(f"{d}/contexts/context_{i}.json", "w"), ensure_ascii=False)
    with zipfile.ZipFile(f"{d}/selected-contexts.zip", "w") as z:
        for i in range(len(corpus)):
            z.write(f"{d}/contexts/context_{i}.json", f"context_{i}.json")
    json.dump(queries, open(f"{d}/train.json", "w"), ensure_ascii=False)
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
    sys.argv = ["ingest", "--raw-corpus", f"{raw}/selected-contexts.zip",
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
    q0 = next(iter(qtok))
    check("k1/b actually change scores",
          idx.search(qtok[q0], 5, b=0.0) != idx.search(qtok[q0], 5, b=1.0))

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

    print("\n6. BTC scoring rules (the ones that zero a submission)")
    capped = {q: (v * 3)[:6] for q, v in list(p10.items())[:2]}   # >5 ids
    padded = dict(p1)
    padded.update(capped)
    s_cap = metrics.official(padded, qrels)
    check("returning >5 ids zeroes those questions",
          s_cap["n_over_cap"] == len(capped) and
          s_cap["primary_recall"] < metrics.official(p1, qrels)["primary_recall"],
          f"{int(s_cap['n_over_cap'])} questions over cap")
    check("cutoff refuses max_k > 5",
          _raises(lambda: cutoff.apply_cutoff([("a", 1.0)], max_k=20), ValueError))
    check("cutoff de-duplicates",
          cutoff.apply_cutoff([("a", 5.0), ("a", 4.0), ("b", 3.0)],
                              rule="top_k", k=3) == ["a", "b"])
    dup_pred = {q: (v + v)[:5] for q, v in p1.items()}
    check("duplicates lower precision (BTC uses the raw list length)",
          metrics.official(dup_pred, qrels)["tiebreak_precision"]
          <= metrics.official(p1, qrels)["tiebreak_precision"])
    probs = metrics.check_submittable({list(qrels)[0]: ["x"]}, list(qrels))
    check("check_submittable catches a short submission", bool(probs))

    print("\n6b. parity with BTC's own scorer")
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from btc_eval.scoring_legalir import eval_retrieval
        full = {q: list(p10.get(q, [])) for q in qrels}
        theirs = eval_retrieval({q: {"answer": v} for q, v in full.items()},
                                {q: list(v) for q, v in qrels.items()})
        ours = metrics.official(full, qrels)
        check("our metrics == BTC's scoring program",
              abs(theirs["recall"] - ours["primary_recall"]) < 1e-12 and
              abs(theirs["precision"] - ours["tiebreak_precision"]) < 1e-12,
              f"R={theirs['recall']:.4f} P={theirs['precision']:.4f}")
    except ImportError as e:
        check("BTC scorer importable", False, str(e))

    print("\n7. cutoff rules")
    rows = cutoff.sweep(run, qrels, max_k=5)
    check("sweep returns all rules",
          {r["rule"] for r in rows} == {"top_k", "ratio", "gap"})
    var = {len(v) for v in cutoff.apply_to_run(run, rule="ratio", alpha=0.9).values()}
    check("ratio produces variable-length sets", len(var) > 1, f"sizes={sorted(var)}")

    print("\n8. fusion")
    f_rrf = fusion.rrf([run, run], [0.5, 0.5], top_k=50)
    check("rrf preserves recall",
          abs(metrics.recall_at_k(f_rrf, qrels, 50) - d["recall@50"]) < 1e-9)
    f_w = fusion.weighted([run, run], [0.5, 0.5], top_k=50)
    check("weighted fusion runs", len(f_w) == len(run))

    print("\n9. experiment log")
    from src import exp_log
    log = f"{work}/runs.csv"
    for i, (dev, lb) in enumerate([(0.5, 0.48), (0.6, 0.59), (0.7, 0.71)]):
        exp_log.log_run({"run_id": f"t{i}", "dev_official": dev, "leaderboard": lb}, log)
    corr = exp_log.correlation(log)
    check("dev/leaderboard correlation gate works", corr["verdict"] == "healthy",
          f"spearman={corr['spearman']:.2f}")

    print("\n10. exhaustive test cases")
    import subprocess
    tests = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "tests", "test_cases.py")
    r = subprocess.run([sys.executable, tests], capture_output=True, text=True)
    tail = [l for l in r.stdout.strip().split("\n") if "passed" in l]
    check("tests/test_cases.py all pass", r.returncode == 0,
          tail[-1] if tail else "see: python tests/test_cases.py")

    print("\n11. optional dependencies")
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
