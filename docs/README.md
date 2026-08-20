# Documentation

| File | What it is | Read when |
|---|---|---|
| [`python_for_cpp.md`](python_for_cpp.md) | Python, mapped from C++ | **you know C++ but not Python — read this first** |
| [`onboarding.md`](onboarding.md) | how the repo fits together | first, after `START_HERE.md` |
| [`walkthrough.md`](walkthrough.md) | the whole pipeline on synthetic data, with real output | day one, before BTC's data arrives |
| [`glossary.md`](glossary.md) | every term used in the phase READMEs | keep it open in a tab |
| [`troubleshooting.md`](troubleshooting.md) | symptoms → diagnosis | something looks wrong |
| [`todo.md`](todo.md) | what is unfinished and who can do it | planning, and before every submission |
| [`docs/reference/`](reference/) | depth beyond what the phases require | a result surprised you |

## The three layers

The material is deliberately layered so nobody has to read everything:

1. **Phase READMEs** (`phases/<n>_*/README.md`) — exactly what that phase needs. Required.
2. **Self-checks** (`phases/<n>_*/self_check.md`) — worked answers. Use them to confirm you
   understood, not to skip Part A.
3. **[`docs/reference/`](reference/)** — derivations, failure modes, "why is it like that".
   Entirely optional; nothing in a phase depends on it.

If you find yourself confused by a phase README, the reference note for that topic probably
answers it. If you find yourself bored by one, skip to Part B.

## Writing code

The places where you write your own code are marked in the source and listed by:

```bash
python tools/todo.py --yours
```

They are deliberately small and self-contained — a cutoff rule, an abbreviation table, a
prompt, a negative-sampling strategy. Each block states what to write, gives ideas ordered
easiest-first, gives the command to test it, and notes the Python you will need if you are
coming from C++. Nothing else in the pipeline breaks while they are unimplemented.
