# 04 — Vietnamese NLP specifics

Prerequisite: `phases/1_bm25/README.md` §A3. Read before changing anything in
`src/normalize.py`.

---

## 1. Unicode: the same word, three byte sequences

Vietnamese stacks a vowel-quality diacritic and a tone mark on one letter. `ế` can be:

- U+1EBF (precomposed, NFC)
- U+00EA + U+0301 (ê + combining acute, NFD)
- U+0065 + U+0302 + U+0301 (e + circumflex + acute, fully decomposed)

To BM25 these are three different strings. To a tokeniser they may be three different
tokens. **`unicodedata.normalize("NFC", …)` first, always, on both corpus and queries.**
This is the cheapest correctness win available and it is invisible when you skip it — the
score is simply lower.

We also strip zero-width characters (U+200B–U+200F, U+FEFF, soft hyphen), which arrive via
copy-paste from PDFs and Word documents and split tokens invisibly.

## 2. Tone-mark placement: two correct spellings

For syllables whose nucleus is `oa`, `oe`, `uy`, Vietnamese has two accepted conventions:

| traditional ("kiểu cũ") | modern ("kiểu mới") |
|---|---|
| hòa | hoà |
| khỏe | khoẻ |
| thủy | thuỷ |

Both appear in real corpora, sometimes in the same document. Unnormalised, `hòa giải` and
`hoà giải` never match.

**The rule is not a find-and-replace.** The two conventions only differ in *open* syllables
— those ending in the vowel cluster. When a final consonant follows, both write the mark on
the same letter:

```
hòa / hoà     differ      (open syllable)
toàn          identical   (closed by -n)
hoàn          identical   (closed by -n)
```

A naive `oà → òa` replacement corrupts `toàn` into `tòan`, which is not a word, and quietly
destroys every match on it. `src/normalize.normalize_tone` guards with a negative lookahead
requiring a non-letter after the cluster:

```python
re.compile(re.escape("oà") + "(?!" + VIETNAMESE_LETTER_CLASS + ")")
```

Verified in `phases/0_harness/smoke_test.py`: `hoà→hòa`, `khoẻ→khỏe`, `thuỷ→thủy`,
and `toàn` unchanged.

## 3. Word segmentation

Vietnamese writes syllables separated by spaces; words may be one syllable or several.
`học sinh` (student) is one word written as two syllables; `học` (study) and `sinh` (born)
are also words. Whitespace is not a word boundary.

**Segmenters** join multi-syllable words with underscores: `học sinh` → `học_sinh`.

| Tool | Notes |
|---|---|
| `pyvi` (`ViTokenizer.tokenize`) | fast, pure Python, the usual default |
| `underthesea` (`word_tokenize`) | broader toolkit, somewhat slower |
| VnCoreNLP | Java, model-based — **register it with BTC if you use it** |

Segmentation is **cached** in `src/normalize.py` (`lru_cache`, 200k entries) because
segmenting a corpus twice is a common and avoidable waste of an afternoon.

### The rule that costs points silently

| Backbone | Models | Input |
|---|---|---|
| PhoBERT | `vietnamese-bi-encoder`, `dangvantuan/vietnamese-embedding`, `PhoRanker`, `ViRanker` | **must be segmented** |
| XLM-R / BGE-M3 | `Vietnamese_Embedding`, `bge-m3`, `Vietnamese_Reranker`, `bge-reranker-v2-m3` | **must not be segmented** |

PhoBERT's vocabulary was built on segmented text; give it raw syllables and every
multi-syllable word fragments into pieces it was never trained on. BGE-M3's SentencePiece
vocabulary was built on raw text; give it underscores and you insert characters that do not
appear in its training distribution.

**Neither mistake raises an exception.** Both just score worse — typically 5–15 points, which
is easy to misread as "this model is not good for our data".

`src/dense.REGISTRY` records the flag per model and `dense.prepare()` applies it, so no call
site has to remember. Add a model to the registry before using it anywhere.

## 4. Legal-domain vocabulary that must survive tokenisation

| Pattern | Example | Why it matters |
|---|---|---|
| decree/law ids | `100/2019/NĐ-CP`, `45/2019/QH14` | near-unique, maximum IDF |
| article refs | `Điều 113`, `khoản 2`, `điểm a` | structural anchors, appear in both query and source |
| dates | `01/01/2021` | version disambiguation between amended texts |
| defined terms | `người lao động`, `hợp đồng lao động` | used verbatim, never paraphrased in the source |

`src/normalize.tokenize` keeps `/`, `.`, `-` inside tokens for the first three. Never
"clean" punctuation out of legal text without checking what it costs on dev.

## 5. Case

Vietnamese legal documents use ALL CAPS for headings and title case erratically. Lowercase
everything for BM25. For neural encoders, `src/normalize.encoder_text` keeps case — the
models were trained on cased text and casing carries some signal.

## 6. What we deliberately do not do

- **No accent removal.** Stripping diacritics (`hòa` → `hoa`) merges genuinely distinct
  words and loses more than it gains. It is sometimes used for noisy user-generated text;
  legal corpora are clean.
- **No stemming.** Vietnamese is analytic — words do not inflect — so there is nothing to stem.
- **Minimal stopwords.** Off by default. See `docs/reference/02_bm25.md` §4.

---

## Check yourself

1. Why does `toàn` need a guard in the tone-normalisation regex when `hoà` does not?
2. You switch from `Vietnamese_Embedding` to `vietnamese-bi-encoder` and the score drops 12
   points. Name the first thing to check, before concluding the model is worse.
3. Why is stemming irrelevant for Vietnamese but segmentation critical?

<details><summary>answers</summary>

1. Because `toàn` is a **closed** syllable — the `oa` cluster is followed by `n` — and both
   spelling conventions place the tone mark identically there. Rewriting it produces `tòan`,
   which is not Vietnamese, and destroys every match on the word. `hoà` is an open syllable
   where the conventions genuinely differ, so rewriting is correct. The guard is a negative
   lookahead for a following letter.
2. **Whether the input is being segmented.** `vietnamese-bi-encoder` is PhoBERT-backbone and
   requires `học_sinh`; `Vietnamese_Embedding` is BGE-M3 and requires raw text. If you
   carried the old preprocessing over, every multi-syllable word is now fragmenting. Check
   `src/dense.REGISTRY` and confirm `dense.prepare()` is what built the input. Also check
   PhoBERT's 256-token limit against your chunk length p90 — it may simply be truncating.
3. Vietnamese is **analytic**: words do not change form for tense, number, or case, so
   there are no inflected variants to collapse — stemming has nothing to do. But word
   *boundaries* are not marked by whitespace, so identifying which syllables form a word is
   a real, unsolved-by-punctuation problem. Segmentation does the job that in English is
   done for free by spaces.
</details>
