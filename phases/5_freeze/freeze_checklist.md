# Freeze checklist — fill in as you go

> **TODO(TEAM/phase5-B1): fill in during the 15-18/09 freeze. Every unchecked box is a submission risk.**

## 1. The chosen run

| Field | Value |
|---|---|
| run_id | |
| git tag | |
| config | `configs/FINAL.yaml` |
| dev score (R / P) | / |
| **leaderboard score** | |
| chosen because | |

> Pick by **leaderboard**, not dev. Dev was for iterating; the leaderboard is your estimate
> of the Private Test. If they disagree about the winner, trust the leaderboard and write
> the disagreement into the paper — it is a finding about your dev split, not a nuisance.

## 2. Component inventory (must match the BTC registration exactly)

| Role | Model | HF revision (SHA) | Params | Registered? |
|---|---|---|---|---|
| segmenter | | | | |
| bi-encoder | | | | |
| cross-encoder | | | | |
| generator (Task 2) | | | | |
| | | **TOTAL** | | **< 4B?** |

```bash
python src/params.py
```

## 3. Determinism

- [ ] seed recorded in `configs/FINAL.yaml`
- [ ] ran the pipeline twice from scratch; submissions are byte-identical
      (`diff <(sha256sum sub1.json) <(sha256sum sub2.json)`)
- [ ] any non-deterministic step documented (GPU nondeterminism in embedding is
      usually harmless at 4 decimal places — confirm, do not assume)

## 4. Data provenance

- [ ] only BTC data used, no augmentation
- [ ] no external API called anywhere in the pipeline
- [ ] dev split regenerable from `build_dev_split.py --seed 42`
- [ ] no test-set labels touched at any point

## 5. Package contents

- [ ] all code
- [ ] README with exact reproduction commands, tested literally by a teammate
- [ ] `requirements-frozen.txt` from `pip freeze`
- [ ] model weights or documented download steps with revisions
- [ ] `configs/FINAL.yaml`
- [ ] `work/experiments/runs.csv`

## 6. Submission verification

| Date | run_id | file | Organization? | Recorded valid? | Score |
|---|---|---|---|---|---|
| | | | | | |

## 7. Sign-off

- [ ] a teammate reproduced the submission from a clean clone
- [ ] freeze declared at: ______ (date/time)
- [ ] no code changes after freeze except documented Private-Test compatibility fixes:
  - 
