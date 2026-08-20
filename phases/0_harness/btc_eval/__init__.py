"""BTC's published scoring programs, vendored VERBATIM. Do not edit.

Provenance
----------
Downloaded from the Drive links BTC circulated on 05/08/2026 (and re-sent with
the Public Test announcement on 06/08):
  Scoring-Program-Task-LegalIR.zip -> scoring_legalir.py
  Scoring-Program-Task-LegalQA.zip -> scoring_legalqa.py

Both are Codabench scoring programs: they read /app/input/{ref,res} and write
/app/output/scores.json. We never run `main()`; we import their `eval_retrieval`
/ `eval_qa` functions directly and hand them dicts, which is exactly what their
own `main()` does after loading the JSON files.

`src/metrics.py` reimplements these semantics for speed. `evaluate.py --cross-check`
verifies the two agree. If they ever disagree, OUR code is wrong.
"""
