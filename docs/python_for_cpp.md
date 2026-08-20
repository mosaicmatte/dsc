# Python for C++ programmers

Written for someone who knows C++ and has not used Python. It covers **only what this
repo actually uses** — you do not need the rest of the language yet.

Read it once end to end (~25 minutes), then keep it open while you work.

---

## 1. The five differences that matter immediately

| | C++ | Python |
|---|---|---|
| Blocks | `{ }` | **indentation** (4 spaces). Wrong indentation is a syntax error. |
| Statement end | `;` | newline |
| Types | declared: `int x = 3;` | inferred: `x = 3`. Names have no type; *values* do. |
| Compile | `g++ main.cpp && ./a.out` | none — `python file.py` runs it directly |
| Errors | mostly compile time | mostly **run time**. A typo in a rarely-taken branch survives until that branch runs. |

That last row is the one that bites. C++ tells you about a misspelled variable before the
program runs; Python tells you an hour into a training job. This is why every script here
validates its inputs early and loudly.

```python
def f(x):
    if x > 0:          # ':' opens a block
        return x * 2   # indented = inside the block
    return 0           # dedented = outside the if
```

## 2. Types you will meet, and their C++ equivalents

| Python | C++ analogue | Literal |
|---|---|---|
| `list` | `std::vector<T>` (but heterogeneous) | `[1, 2, 3]` |
| `dict` | `std::unordered_map<K,V>` | `{"a": 1, "b": 2}` |
| `set` | `std::unordered_set<T>` | `{1, 2, 3}` |
| `tuple` | `std::pair` / fixed `struct`, immutable | `(doc_id, score)` |
| `str` | `std::string`, immutable, always Unicode | `"Điều 113"` |
| `None` | `nullptr` / "no value" | `None` |

```python
scores = {}                    # std::unordered_map<std::string, double>
scores["740"] = 1.5
scores.get("999", 0.0)         # like .count() then [] — returns default if absent
for doc_id, s in scores.items():   # structured binding over the map
    print(doc_id, s)
```

**`set` matters here.** The whole scoring rule is set intersection:

```python
len(set(truth) & set(pred))    # |truth ∩ pred|
```

## 3. Slicing — the notation you will see constantly

```python
xs = [10, 20, 30, 40, 50]
xs[0]      # 10        first
xs[-1]     # 50        last (negative counts from the end)
xs[:3]     # [10,20,30]   first three   -> "top-k"
xs[2:]     # [30,40,50]   from index 2
xs[1:3]    # [20,30]      half-open, like C++ iterators [begin, end)
```

`ranked[:5]` is "the top 5" and appears everywhere in this repo, because BTC allows at
most 5 documents per question.

## 4. Functions, default arguments, keyword arguments

```python
def apply_cutoff(ranked, rule="top_k", k=10, max_k=5):
    ...

apply_cutoff(run)                       # uses all the defaults
apply_cutoff(run, rule="ratio", k=3)    # named arguments, any order
```

C++ has default arguments too, but Python lets you pass them **by name**, which is why
calls in this repo read like `search(query, top_k=100, k1=1.2, b=0.75)`. You never have to
remember positional order.

> **One trap with no C++ equivalent:** never write `def f(xs=[])`. The default list is
> created *once* and shared between calls. Use `def f(xs=None)` then `xs = xs or []`.

## 5. Comprehensions — the loop you will read most often

```python
squares = [x * x for x in range(10)]                 # build a list
evens   = [x for x in xs if x % 2 == 0]              # with a filter
lookup  = {d: s for d, s in pairs}                   # build a dict
```

Equivalent C++:

```cpp
std::vector<int> squares;
for (int x = 0; x < 10; ++x) squares.push_back(x * x);
```

BTC's own scoring code is one big comprehension — read it slowly, it is worth it:

```python
recall = np.array([
    len(set(y_true[k]) & set(y_pred.get(k, set()))) / len(y_true[k])
    if len(y_pred.get(k)) > 0 and len(y_pred.get(k)) <= 5
    else 0
    for k in ids_truth
]).mean()
```

Read it inside-out: *for each question `k`, if the prediction has between 1 and 5 ids,
take |truth ∩ pred| / |truth|; otherwise 0. Then average.* That is the whole Task 1 metric.

## 6. Imports and modules

```python
from src import metrics          # like #include, but namespaced
from src.cutoff import apply_cutoff
import numpy as np               # alias
```

A `.py` file is a module; a directory with `__init__.py` is a package. There is no header
/ source split and no linker — the import runs the file the first time it is imported.

**`sys.path`** is the include path. Every script here starts with a few lines that add the
repo root to it, so `from src import ...` works no matter where you run it from.

## 7. Objects, briefly

```python
class BM25Index:
    def __init__(self, docs, doc_ids):   # constructor
        self.doc_ids = doc_ids           # `self` is `this`, but EXPLICIT

    def search(self, query, top_k=100):  # every method takes self first
        ...

idx = BM25Index(docs, ids)               # no `new`, no delete
hits = idx.search(tokens, top_k=5)
```

No `public`/`private` (a leading underscore like `_helper` means "internal, do not touch"),
no destructors you need to write, and no manual memory management — Python is
garbage-collected.

## 8. Errors and exceptions

```python
raise ValueError("max_k=20 exceeds BTC's limit of 5")

try:
    risky()
except FileNotFoundError as e:
    print("missing:", e)
```

Like C++ exceptions, but used far more routinely — for ordinary "bad input" cases too, not
only catastrophes. When a script here refuses to write an invalid submission, that is a
`SystemExit` raised on purpose.

## 9. Reading a traceback

Python prints the call stack **innermost last** — the opposite of what you might expect.
**Read the last line first**: it says what went wrong. Then read upward to see where.

```
Traceback (most recent call last):
  File "phases/1_bm25/bm25_baseline.py", line 88, in main
    run = idx.batch_search(qtok, top_k=a.depth)
  File "src/bm25.py", line 74, in batch_search
    return {qid: self.search(toks, top_k, k1, b) for qid, toks in items}
KeyError: 'q0'          <-- START HERE: a dict lookup for key 'q0' failed
```

The three you will see most:

| Error | Means | Usual cause |
|---|---|---|
| `KeyError: 'x'` | dict has no such key | id mismatch between two files |
| `FileNotFoundError` | path does not exist | ran a step out of order |
| `TypeError: ... NoneType` | used a value that was `None` | a `.get()` returned nothing |

## 10. Virtual environments

```bash
python3 -m venv .venv          # make an isolated set of libraries
source .venv/bin/activate      # use it  (prompt shows (.venv))
pip install -r requirements.txt
```

Closest C++ analogue: a per-project dependency directory instead of system-wide installs.
**If a script says a package is missing, you almost certainly forgot `source
.venv/bin/activate`.**

## 11. The libraries in this repo

| Library | What it is | C++ analogue |
|---|---|---|
| `numpy` | fast arrays, vectorised maths | Eigen / valarray |
| `json` | read/write JSON | nlohmann::json |
| `argparse` | command-line flags | manual `argv` parsing |
| `re` | regular expressions | `std::regex`, but actually pleasant |
| `torch` | tensors + neural nets, GPU | LibTorch |
| `transformers` / `sentence-transformers` | pretrained models | — |

**numpy is the one worth learning properly.** It replaces loops with whole-array
operations, and it is 10–100× faster than a Python loop:

```python
scores = np.zeros(N)                  # vector<double>(N, 0.0)
scores += idf * tf / (tf + norm)      # elementwise over the WHOLE array
top = np.argpartition(-scores, 5)[:5] # indices of the 5 largest
```

A Python `for` loop over 60,000 documents is slow; the same thing in numpy is instant.
When something in this repo is unexpectedly slow, the fix is usually "move the loop into
numpy", not "optimise the loop".

## 12. Style conventions used here

- `snake_case` for functions and variables, `CapWords` for classes, `UPPER_CASE` for
  constants — the opposite convention to a lot of C++ code.
- 4-space indentation, no tabs. Mixing them is an error.
- Docstrings (`"""..."""`) at the top of every file and function say *why*, not *what*.
- Type hints (`def f(x: str) -> int:`) are **documentation only** — Python does not check
  or enforce them at run time. Treat them as comments that tooling can read.

---

## Your first 20 minutes of Python, using this repo

```bash
source .venv/bin/activate
python                       # interactive prompt, like a REPL
```

```python
>>> import sys; sys.path.insert(0, '.')
>>> from src.normalize import normalize, tokenize
>>> normalize("Hoà giải tại Toà án")
'hòa giải tại tòa án'
>>> tokenize("Nghị định 100/2019/NĐ-CP")
['nghị', 'định', '100/2019/nđ-cp']

>>> from src.metrics import official
>>> truth = {"q1": {"740", "177504"}}
>>> official({"q1": ["740"]}, truth)["primary_recall"]      # found 1 of 2
0.5
>>> official({"q1": ["740", "177504"]}, truth)["primary_recall"]
1.0
>>> official({"q1": ["1","2","3","4","5","6"]}, truth)["primary_recall"]   # over the cap
0.0
>>> exit()
```

That last line is the single most important fact in the competition, and you just proved
it to yourself in one line of Python.

---

## Where to go when you are stuck

1. `python -c "help(str.split)"` — built-in documentation for anything.
2. [`docs/troubleshooting.md`](troubleshooting.md) — symptoms → diagnosis for this repo.
3. The docstring at the top of the script you are running. Every one explains what it does
   and how.

You do **not** need to be fluent in Python to contribute here. Phase 0 and Phase 1 are
mostly running commands and reading numbers. Write your first Python in the
`TODO(YOU/...)` blocks — they are small, marked, and each one tells you exactly what to
write and how to test it:

```bash
python tools/todo.py --yours
```
