# The experiment log

> **If you cannot regenerate a submission from its `run_id` four weeks later, the run did
> not happen.**

## The rule

One CSV row per run, appended automatically by the phase scripts:

```
run_id | date | task | phase | chunking | retriever | negatives | reranker |
cutoff_rule | dev_P | dev_R | dev_official | leaderboard | n_params | config |
git_tag | seed | notes
```

Paired with:
- `configs/<run_id>.yaml` — the exact frozen configuration
- a git tag per **submitted** run

Never edit `runs.csv` by hand except through `src.exp_log.update_leaderboard`.
Hand edits are how a log stops being trustworthy, and an untrustworthy log is worse than
none because you will cite it in the paper.

## Recording a leaderboard score

```bash
python -c "from src.exp_log import update_leaderboard as u; u('<run_id>', 0.7231)"
```

Do it the moment the score appears. A `PENDING` row that nobody fills in is a run you
cannot use in the ablation table.

## The correlation gate

```bash
python -c "from src.exp_log import correlation as c; print(c())"
```

After ≥3 submitted runs this reports the Spearman correlation between dev and leaderboard:

| verdict | meaning | action |
|---|---|---|
| `healthy` (ρ ≥ 0.8) | dev predicts the leaderboard | keep going |
| `SUSPECT` (0.5 ≤ ρ < 0.8) | something is off | inspect before trusting dev |
| `BROKEN` (ρ < 0.5) | dev is lying to you | **stop modelling, fix the harness** |

Usual causes of divergence, in order of frequency:
1. chunk-level doc_ids leaked into the submission (forgot `--aggregate max`)
2. queries missing from the submission — scored as zero
3. wrong averaging in `src/metrics.OFFICIAL_AVERAGING`
4. dev leakage — dev queries also present in the training data

## Directory contents

| Path | Committed? | What |
|---|---|---|
| `runs.csv` | **yes** | the log — this is the paper's ablation table |
| `runs/` | no | full rankings `{qid, ranked:[[doc,score],...]}` |
| `predictions/` | no | answer sets after a cutoff `{qid, predicted:[...]}` |

Runs and predictions are regenerable from `work/configs/`, so they are gitignored. The CSV is
not regenerable and is committed.

## Reading the log

```bash
python -c "
import pandas as pd; pd.set_option('display.width',200)
d = pd.read_csv('work/experiments/runs.csv')
print(d[['run_id','retriever','reranker','cutoff_rule','dev_R','dev_P','leaderboard']])"
```
