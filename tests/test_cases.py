#!/usr/bin/env python3
"""Exhaustive test cases for scoring, cutoff and loss attribution.

WHY THIS FILE EXISTS
--------------------
The analysis is only worth reading if it is correct on every case, not just the
common one. This enumerates the whole space of things a question can do —
including the ones that look impossible until the day they happen — and asserts
what the repo should do about each.

It needs no data and no GPU:

    python tests/test_cases.py            # run everything
    python tests/test_cases.py -v         # list every case as it passes

Categories covered
  A. BTC parity          our metrics vs their scoring program, incl. edge cases
  B. The 5-id cap        at, below, above, and exactly on the boundary
  C. Degenerate input    empty, missing, extra, duplicate, unknown, chunk ids
  D. Loss attribution    the components sum to the gap, in every configuration
  E. Cutoff rules        clamping, de-duplication, tie handling, degenerate scores
  F. Ordering            leaderboard ordering, recall-then-precision tiebreak
  G. Text normalisation  tone marks, Unicode forms, legal identifiers
  H. Chunking            granularity, parent mapping, aggregation
  I. Property tests      randomised, to catch what enumeration misses
"""
from __future__ import annotations

import os
import random
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "phases", "0_harness"))

from src import analysis, chunking, cutoff, metrics, normalize  # noqa: E402

CAP = metrics.MAX_DOCS_PER_QUERY
PASS, FAIL = [], []
VERBOSE = "-v" in sys.argv


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    if VERBOSE or not cond:
        print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


def btc(preds, truth):
    """Score through BTC's own program."""
    from btc_eval.scoring_legalir import eval_retrieval
    return eval_retrieval({k: {"answer": list(v)} for k, v in preds.items()},
                          {k: list(v) for k, v in truth.items()})


def ours(preds, truth):
    s = metrics.official(preds, {k: set(v) for k, v in truth.items()})
    return {"recall": s["primary_recall"], "precision": s["tiebreak_precision"]}


def agree(preds, truth):
    t, o = btc(preds, truth), ours(preds, truth)
    return (abs(t["recall"] - o["recall"]) < 1e-12
            and abs(t["precision"] - o["precision"]) < 1e-12)


# ===========================================================================
def section(title):
    if VERBOSE:
        print(f"\n{title}")


def test_A_btc_parity():
    section("A. Parity with BTC's scoring program")
    cases = {
        "exact match":            ({"q": ["1", "2"]}, {"q": ["1", "2"]}),
        "partial":                ({"q": ["1"]},      {"q": ["1", "2"]}),
        "no overlap":             ({"q": ["9"]},      {"q": ["1"]}),
        "superset within cap":    ({"q": ["1", "2", "3"]}, {"q": ["1"]}),
        "exactly 5 ids":          ({"q": list("12345")}, {"q": ["1"]}),
        "6 ids (over cap)":       ({"q": list("123456")}, {"q": ["1"]}),
        "empty prediction":       ({"q": []},         {"q": ["1"]}),
        "duplicate ids":          ({"q": ["1", "1"]}, {"q": ["1"]}),
        "all duplicates":         ({"q": ["1", "1", "1"]}, {"q": ["1"]}),
        "unknown id":             ({"q": ["999"]},    {"q": ["1"]}),
        "gold has 1, we send 5":  ({"q": list("12345")}, {"q": ["5"]}),
        "gold has 6":             ({"q": list("12345")}, {"q": list("123456")}),
        "multi-question":         ({"a": ["1"], "b": []}, {"a": ["1", "2"], "b": ["3"]}),
        "order irrelevant":       ({"q": ["2", "1"]}, {"q": ["1", "2"]}),
    }
    for name, (p, t) in cases.items():
        check(f"A: {name}", agree(p, t))


def test_B_cap():
    section("B. The 5-id cap")
    truth = {"q": ["1", "2"]}
    for n in range(0, 9):
        preds = {"q": [str(i) for i in range(1, n + 1)]}
        r = ours(preds, truth)["recall"]
        if n == 0:
            check("B: 0 ids scores 0", r == 0.0)
        elif n <= CAP:
            check(f"B: {n} ids scored normally", r > 0.0, f"recall={r:.2f}")
        else:
            check(f"B: {n} ids zeroed", r == 0.0)
    check("B: boundary — exactly 5 is valid",
          ours({"q": list("12345")}, truth)["recall"] > 0)
    check("B: boundary — exactly 6 is zero",
          ours({"q": list("123456")}, truth)["recall"] == 0)
    check("B: over-cap zeroes PRECISION too",
          ours({"q": list("123456")}, truth)["precision"] == 0)
    check("B: one bad question does not zero the others",
          abs(ours({"a": list("123456"), "b": ["1"]},
                   {"a": ["1"], "b": ["1"]})["recall"] - 0.5) < 1e-12)


def test_C_degenerate():
    section("C. Degenerate and malformed input")
    qrels = {"a": {"1"}, "b": {"2"}}
    check("C: missing question counts as empty (not skipped)",
          metrics.official({"a": ["1"]}, qrels)["primary_recall"] == 0.5)
    probs = metrics.check_submittable({"a": ["1"]}, ["a", "b"])
    check("C: check_submittable flags a missing question", any("missing" in p for p in probs))
    probs = metrics.check_submittable({"a": ["1"], "b": ["2"], "c": ["3"]}, ["a", "b"])
    check("C: check_submittable flags an extra question", any("not in the reference" in p for p in probs))
    probs = metrics.check_submittable({"a": list("123456"), "b": ["2"]}, ["a", "b"])
    check("C: check_submittable flags over-cap", any("more than" in p for p in probs))
    probs = metrics.check_submittable({"a": ["1", "1"], "b": ["2"]}, ["a", "b"])
    check("C: check_submittable flags duplicates", any("duplicate" in p for p in probs))
    probs = metrics.check_submittable({"a": [], "b": ["2"]}, ["a", "b"])
    check("C: check_submittable flags empty", any("empty" in p for p in probs))
    check("C: clean submission has no problems",
          metrics.check_submittable({"a": ["1"], "b": ["2"]}, ["a", "b"]) == [])
    check("C: zero-gold question does not crash",
          metrics.official({"a": ["1"]}, {"a": set()})["primary_recall"] == 0.0)
    check("C: empty everything does not crash",
          metrics.official({}, {})["primary_recall"] == 0.0)


def test_D_attribution():
    section("D. Loss attribution sums exactly")
    scen = {
        "perfect":              ({"a", "b"}, [("a", 9), ("b", 8)], ["a", "b"]),
        "cutoff too small":     ({"a", "b"}, [("a", 9), ("b", 8)], ["a"]),
        "ranked out of cap":    ({"a", "b"}, [("x", 9)] * 0 + [("a", 9), ("x", 8),
                                              ("y", 7), ("z", 6), ("w", 5), ("b", 4)],
                                 ["a"]),
        "not retrieved":        ({"a", "b"}, [("a", 9)], ["a"]),
        "impossible (7 gold)":  (set("abcdefg"), [(c, 9 - i) for i, c in
                                                  enumerate("abcdefg")],
                                 list("abcde")),
        "zeroed over cap":      ({"a"}, [("a", 9)], list("abcdef")),
        "zeroed empty":         ({"a"}, [("a", 9)], []),
        "nothing retrieved":    ({"a"}, [("x", 9)], ["x"]),
        "no gold at all":       (set(), [("x", 9)], ["x"]),
    }
    for name, (gold, ranked, pred) in scen.items():
        r = analysis.diagnose_query("q", gold, ranked, pred)
        total = r["recall"] + r["loss_total"]
        ok = abs(total - (1.0 if gold else 0.0)) < 1e-12
        check(f"D: {name} sums to 1", ok, f"recall={r['recall']:.3f} "
                                          f"gap={r['loss_total']:.3f}")
        check(f"D: {name} no negative component",
              all(r[k] >= -1e-12 for k in analysis.LOSS_COMPONENTS))

    # attribution points at the right cause
    r = analysis.diagnose_query("q", {"a", "b"}, [("a", 9), ("b", 8)], ["a"])
    check("D: cutoff loss identified", r["loss_cutoff"] > 0 and r["loss_retrieval"] == 0)
    r = analysis.diagnose_query("q", {"a", "b"}, [("a", 9)], ["a"])
    check("D: retrieval loss identified", r["loss_retrieval"] > 0 and r["loss_cutoff"] == 0)
    r = analysis.diagnose_query("q", {"a", "b"},
                                [("a", 9), ("x", 8), ("y", 7), ("z", 6), ("w", 5), ("b", 4)],
                                ["a"])
    check("D: ranking loss identified", r["loss_ranking"] > 0)
    r = analysis.diagnose_query("q", set("abcdefg"),
                                [(c, 9) for c in "abcdefg"], list("abcde"))
    check("D: cap loss identified", r["loss_cap"] > 0 and r["impossible"])
    r = analysis.diagnose_query("q", {"a"}, [("a", 9)], list("abcdef"))
    check("D: zeroed loss identified", r["loss_zeroed"] > 0 and r["recall"] == 0)

    # aggregate decomposes the macro gap
    rows = [analysis.diagnose_query(f"q{i}", g, rk, p)
            for i, (g, rk, p) in enumerate(scen.values()) if g]
    agg = analysis.aggregate(rows)
    check("D: aggregate sums to the macro gap",
          abs(agg["recall"] + agg["loss_total"] - 1.0) < 1e-12,
          f"{agg['recall']:.4f}+{agg['loss_total']:.4f}")

    # integrity flags
    r = analysis.diagnose_query("q", {"a"}, [("a", 9)], ["a", "a"], {"a"})
    check("D: duplicates flagged", r["has_duplicates"])
    r = analysis.diagnose_query("q", {"a"}, [("a", 9)], ["zz"], {"a"})
    check("D: unknown id flagged", r["unknown_ids"] == ["zz"])
    r = analysis.diagnose_query("q", {"a"}, [("a", 9)], ["L1#dieu3"], {"a"})
    check("D: chunk id flagged", r["chunk_ids"] == ["L1#dieu3"])
    r = analysis.diagnose_query("q", {"zz"}, [("a", 9)], ["a"], {"a"})
    check("D: gold missing from corpus flagged", r["gold_missing_from_corpus"] == ["zz"])
    r = analysis.diagnose_query("q", {"a"}, [("x", 9)], ["a"])
    check("D: prefix violation flagged", r["prefix_violation"])


def test_E_cutoff():
    section("E. Cutoff rules")
    ranked = [("a", 10.0), ("b", 9.5), ("c", 9.0), ("d", 8.5),
              ("e", 8.0), ("f", 1.0), ("g", 0.5)]
    for rule in ("top_k", "ratio", "gap", "threshold"):
        got = cutoff.apply_cutoff(ranked, rule=rule, k=10, alpha=0.5, tau=0.0)
        check(f"E: {rule} never exceeds the cap", len(got) <= CAP, f"{len(got)}")
        check(f"E: {rule} never returns empty", len(got) >= 1)
    check("E: max_k above the cap raises",
          _raises(lambda: cutoff.apply_cutoff(ranked, max_k=CAP + 1), ValueError))
    check("E: duplicates removed",
          cutoff.apply_cutoff([("a", 5.0), ("a", 4.0), ("b", 3.0)],
                              rule="top_k", k=3) == ["a", "b"])
    check("E: empty ranking gives empty answer",
          cutoff.apply_cutoff([], rule="ratio") == [])
    check("E: single document works",
          cutoff.apply_cutoff([("a", 1.0)], rule="ratio") == ["a"])
    check("E: all-equal scores handled",
          len(cutoff.apply_cutoff([(c, 1.0) for c in "abcdefg"], rule="ratio")) <= CAP)
    check("E: all-zero scores handled",
          len(cutoff.apply_cutoff([(c, 0.0) for c in "abc"], rule="ratio")) >= 1)
    check("E: negative scores handled",
          len(cutoff.apply_cutoff([("a", -1.0), ("b", -2.0)], rule="ratio")) >= 1)
    check("E: ratio adapts to the distribution",
          len(cutoff.apply_cutoff([("a", 10.0), ("b", 1.0)], rule="ratio", alpha=0.8))
          < len(cutoff.apply_cutoff([(c, 10.0) for c in "abcde"],
                                    rule="ratio", alpha=0.8)))
    check("E: unsorted input is sorted first",
          cutoff.apply_cutoff([("b", 1.0), ("a", 9.0)], rule="top_k", k=1) == ["a"])
    check("E: unknown rule raises",
          _raises(lambda: cutoff.apply_cutoff(ranked, rule="nope"), ValueError))
    rows = cutoff.sweep({"q": ranked}, {"q": {"a"}})
    check("E: sweep stays within the cap",
          all(r["avg_set_size"] <= CAP for r in rows))


def test_F_ordering():
    section("F. Leaderboard ordering")
    q = {"a": {"1", "2"}}
    hi_r = metrics.official({"a": ["1", "2", "9"]}, q)
    hi_p = metrics.official({"a": ["1"]}, q)
    check("F: higher recall wins even with lower precision",
          metrics.compare(hi_r, hi_p) == 1,
          f"R {hi_r['primary_recall']:.2f}/{hi_p['primary_recall']:.2f}")
    a = metrics.official({"a": ["1"]}, q)
    b = metrics.official({"a": ["1", "9"]}, q)
    check("F: equal recall broken by precision", metrics.compare(a, b) == 1)
    check("F: identical is a draw", metrics.compare(a, dict(a)) == 0)


def test_G_normalisation():
    section("G. Vietnamese normalisation")
    cases = [
        ("Hoà giải", "hòa giải", "tone: open syllable rewritten"),
        ("Khoẻ mạnh", "khỏe mạnh", "tone: oe"),
        ("Thuỷ lợi", "thủy lợi", "tone: uy"),
        ("Toàn bộ", "toàn bộ", "tone: closed syllable NOT corrupted"),
        ("Hoàn thành", "hoàn thành", "tone: closed syllable NOT corrupted"),
    ]
    for src, want, why in cases:
        check(f"G: {why}", normalize.normalize(src) == want,
              f"{src!r}->{normalize.normalize(src)!r}")
    nfd = unicodedata.normalize("NFD", "Điều")
    check("G: NFD and NFC normalise identically",
          normalize.normalize(nfd) == normalize.normalize("Điều"))
    toks = normalize.tokenize("Nghị định 100/2019/NĐ-CP")
    check("G: decree id survives as one token", "100/2019/nđ-cp" in toks, str(toks))
    check("G: idempotent",
          normalize.normalize(normalize.normalize("Hoà")) == normalize.normalize("Hoà"))
    check("G: empty string safe", normalize.normalize("") == "")
    check("G: None safe", normalize.normalize(None) == "")
    check("G: newlines kept when asked",
          "\n" in normalize.normalize("a\nb", keep_newlines=True))
    check("G: newlines flattened by default",
          "\n" not in normalize.normalize("a\nb"))
    check("G: zero-width characters stripped",
          normalize.normalize("a​b") == "ab")


def test_H_chunking():
    section("H. Chunking")
    doc = {"doc_id": "740", "text": "Tiêu đề\nĐiều 1. A\n1. x\n2. y\nĐiều 2. B\n1. z",
           "meta": {"title": "T"}}
    arts = chunking.build_corpus([doc], "article", min_chars=1)
    check("H: article split finds both điều",
          {"740#dieu1", "740#dieu2"} <= {r["doc_id"] for r in arts},
          str([r["doc_id"] for r in arts]))
    pm = chunking.parent_map(arts)
    check("H: every chunk maps to its parent",
          all(v == "740" for v in pm.values()))
    run = {"q": [("740#dieu1", 5.0), ("740#dieu2", 3.0)]}
    agg = chunking.aggregate_to_parent(run, pm, "max")
    check("H: aggregation collapses to the parent id",
          agg["q"] == [("740", 5.0)], str(agg["q"]))
    check("H: sum aggregation adds",
          chunking.aggregate_to_parent(run, pm, "sum")["q"][0][1] == 8.0)
    check("H: no newlines -> no split (documented failure mode)",
          len(chunking.build_corpus(
              [{"doc_id": "1", "text": "Điều 1. A Điều 2. B", "meta": {}}],
              "article", min_chars=1)) == 1)
    check("H: document granularity is a no-op",
          len(chunking.build_corpus([doc], "document", min_chars=1)) == 1)


def test_I_property():
    section("I. Randomised property tests")
    rng = random.Random(1234)
    docs = [str(i) for i in range(40)]
    bad_parity = bad_sum = bad_neg = 0
    for _ in range(2000):
        qids = [f"q{i}" for i in range(rng.randint(1, 5))]
        truth = {q: rng.sample(docs, rng.randint(1, 8)) for q in qids}
        pred = {}
        for q in qids:
            n = rng.choice([0, 1, 2, 3, 4, 5, 6, 9])
            v = rng.sample(docs, n) if n else []
            if n >= 2 and rng.random() < 0.25:
                v[1] = v[0]                       # duplicate
            if rng.random() < 0.3 and truth[q]:
                v = (v + truth[q][:1])[:max(n, 1)]  # ensure some hits
            pred[q] = v
        if not agree(pred, truth):
            bad_parity += 1
        for q in qids:
            ranked = [(d, float(len(docs) - i)) for i, d in enumerate(docs)]
            rng.shuffle(ranked)
            r = analysis.diagnose_query(q, set(truth[q]), ranked, pred[q])
            if abs(r["recall"] + r["loss_total"] - 1.0) > 1e-9:
                bad_sum += 1
            if any(r[k] < -1e-12 for k in analysis.LOSS_COMPONENTS):
                bad_neg += 1
    check("I: 2000 random submissions match BTC exactly", bad_parity == 0,
          f"{bad_parity} mismatches")
    check("I: attribution always sums to the gap", bad_sum == 0, f"{bad_sum} bad")
    check("I: no negative loss components", bad_neg == 0, f"{bad_neg} bad")

    # cutoff never violates the contract, whatever the score distribution
    bad = 0
    for _ in range(2000):
        n = rng.randint(0, 12)
        ranked = [(str(i), rng.choice([0.0, -1.0, 1e-9, rng.random(), 1e6]))
                  for i in range(n)]
        for rule in ("top_k", "ratio", "gap", "threshold"):
            got = cutoff.apply_cutoff(ranked, rule=rule, k=rng.randint(1, 9),
                                      alpha=rng.random(), tau=rng.random())
            if len(got) > CAP or len(got) != len(set(got)):
                bad += 1
            if n and not got:
                bad += 1
    check("I: cutoff always <=5, de-duplicated, non-empty", bad == 0, f"{bad} bad")


def _raises(fn, exc):
    try:
        fn()
        return False
    except exc:
        return True


def main():
    print("Exhaustive test cases — scoring, cutoff, attribution, text, chunking\n")
    for fn in (test_A_btc_parity, test_B_cap, test_C_degenerate, test_D_attribution,
               test_E_cutoff, test_F_ordering, test_G_normalisation,
               test_H_chunking, test_I_property):
        fn()
    print(f"\n{'=' * 62}")
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        sys.exit(1)
    print("ALL CASES PASS")


if __name__ == "__main__":
    main()
