#!/usr/bin/env python3
"""Turn Task 2 predictions into a Codabench submission, and log it.

FORMAT (confirmed against BTC's scorer, btc_eval/scoring_legalqa.py, which does
``y_pred = {k: v['answer'] for k, v in y_pred.items()}``):

    submission.zip  ->  submission.json
    {
      "9001": {"answer": "Theo Điều 37 Nghị định 153/2020/NĐ-CP ..."},
      "9002": {"answer": "..."}
    }

A JSON object keyed by question id, whose "answer" is a STRING of prose.

HARD FAILURE MODES
------------------
Their scorer raises — the whole submission fails, it does not merely score low —
when the number of keys differs from the reference. So every question must be
present, even if the answer is empty. Pre-flight below refuses to write a file
that would fail.

SCORING REMINDER
----------------
METEOR is primary and is heavily recall-weighted; ROUGE-L is secondary. Neither
rewards brevity. An empty or one-line answer scores near zero even when it is
factually right, so never submit blanks — emit the retrieved passage as a
fallback instead.

USAGE
  python phases/4_task2_qa/make_submission_task2.py \
      --pred work/experiments/predictions/<run_id>.jsonl \
      --queries data/processed/task2_public_test.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.abspath(  # repo root: phases/<n>_<name>/ -> ../..
    os.path.join(os.path.dirname(__file__), "..", "..")))

from src import exp_log, io_utils  # noqa: E402

SUBMISSION_FILENAME = "submission.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True,
                    help="output of baseline_generative.py / baseline_extractive.py")
    ap.add_argument("--queries", default="data/processed/task2_public_test.jsonl")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out-dir", default="work/submissions")
    ap.add_argument("--fallback", default="",
                    help="text to use when a question has no answer "
                         "(better: pass --fallback-run to emit the top passage)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    queries = io_utils.load_queries(a.queries)
    preds = {r["qid"]: str(r.get("answer", "") or "")
             for r in io_utils.read_jsonl(a.pred)}

    answers = {q["qid"]: preds.get(q["qid"], a.fallback) for q in queries}

    problems = []
    missing = [q for q, v in answers.items() if q not in preds]
    if missing:
        problems.append(f"{len(missing)} questions have no prediction "
                        f"(e.g. {missing[:5]}) — filled with --fallback; BTC's "
                        f"scorer RAISES if any key is absent entirely")
    blank = [q for q, v in answers.items() if not v.strip()]
    if blank:
        problems.append(f"{len(blank)} answers are EMPTY (e.g. {blank[:5]}) — "
                        f"each scores ~0 METEOR. Set --fallback or fix generation")
    extra = [q for q in preds if q not in answers]
    if extra:
        problems.append(f"{len(extra)} predictions for questions not in the "
                        f"reference (e.g. {extra[:5]}) — key-count mismatch, "
                        f"the submission fails outright")
    for p in problems:
        print(f"PRE-FLIGHT ERROR: {p}", file=sys.stderr)
    if problems and not a.force:
        print("\nRefusing to write an invalid submission. Fix the above, or pass "
              "--force.", file=sys.stderr)
        sys.exit(1)

    run_id = a.run_id or os.path.basename(a.pred).replace(".jsonl", "")
    os.makedirs(a.out_dir, exist_ok=True)
    json_path = os.path.join(a.out_dir, f"{run_id}.json")
    zip_path = os.path.join(a.out_dir, f"{run_id}.zip")

    payload = {qid: {"answer": txt} for qid, txt in answers.items()}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, SUBMISSION_FILENAME)

    lens = [len(v.split()) for v in answers.values()]
    gold_lens = [len(str(q.get("answer", "")).split()) for q in queries
                 if q.get("answer")]
    print(f"\nrun_id      : {run_id}")
    print(f"questions   : {len(answers)}")
    print(f"answer words: mean {sum(lens)/max(len(lens),1):.0f}, "
          f"min {min(lens, default=0)}, max {max(lens, default=0)}")
    if gold_lens:
        gm = sum(gold_lens) / len(gold_lens)
        print(f"reference   : mean {gm:.0f} words")
        ratio = (sum(lens) / max(len(lens), 1)) / max(gm, 1e-9)
        print(f"length ratio: {ratio:.2f}x"
              + ("  <- much shorter than the reference; METEOR is recall-weighted, "
                 "this costs you points" if ratio < 0.6 else
                 "  <- much longer than the reference; precision drag on METEOR"
                 if ratio > 2.0 else ""))
    print(f"wrote       : {json_path}\n              {zip_path}")

    exp_log.log_run({
        "run_id": run_id, "phase": "4", "task": "2",
        "leaderboard": "PENDING",
        "notes": f"TASK2 SUBMITTED {os.path.basename(zip_path)}; "
                 f"mean_len={sum(lens)/max(len(lens),1):.0f}w; {a.notes}",
    })
    print("\nSubmit through the REGISTERED ORGANIZATION on "
          "https://www.codabench.org/competitions/17716/")


if __name__ == "__main__":
    main()
