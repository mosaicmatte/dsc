"""Vietnamese text normalisation and tokenisation.

Rules of the house
------------------
1. Apply the *same* normaliser to the corpus and to the queries. A mismatch here
   silently costs more recall than any model choice.
2. PhoBERT-backbone models (``vietnamese-bi-encoder``, ``PhoRanker``) REQUIRE
   word-segmented input. BGE-M3-backbone models (``Vietnamese_Embedding``,
   ``Vietnamese_Reranker``, ``bge-m3``) must NOT be given segmented input.
   Getting this backwards is a silent 5-15 point drop, not a crash.
3. Never strip digits or article markers. "Điều 12", "Khoản 3", "Điểm a" and
   "Nghị định 100/2019/NĐ-CP" are the highest-signal tokens in legal retrieval.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Iterable, List, Optional

# Every Vietnamese letter, used for syllable-boundary lookaheads.
_VN_LOWER = (
    "abcdefghijklmnopqrstuvwxyz"
    "àáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
)
_VN_LETTERS = _VN_LOWER + _VN_LOWER.upper()
_VN_CLASS = "[" + re.escape(_VN_LETTERS) + "]"

# Tone-placement variants. Only "open" syllables (oa/oe/uy with no final
# consonant) actually differ between the two Vietnamese styles:
#   hoà ~ hòa,  khoẻ ~ khỏe,  thuỷ ~ thủy
# but "toàn", "hoàn" are spelled identically in both styles, so the rewrite is
# guarded by a lookahead that requires the syllable to END there.
_TONE_PAIRS = [
    ("oà", "òa"), ("oá", "óa"), ("oả", "ỏa"), ("oã", "õa"), ("oạ", "ọa"),
    ("oè", "òe"), ("oé", "óe"), ("oẻ", "ỏe"), ("oẽ", "õe"), ("oẹ", "ọe"),
    ("uỳ", "ùy"), ("uý", "úy"), ("uỷ", "ủy"), ("uỹ", "ũy"), ("uỵ", "ụy"),
]


def _tone_variants():
    out = []
    for src, dst in _TONE_PAIRS:
        for s, d in ((src, dst), (src.capitalize(), dst.capitalize()),
                     (src.upper(), dst.upper())):
            out.append((re.compile(re.escape(s) + "(?!" + _VN_CLASS + ")"), d))
    return out


_TONE_RE = _tone_variants()
_WS_RE = re.compile(r"\s+")
_HSPACE_RE = re.compile(r"[ \t\f\v]+")
_NL_RE = re.compile(r"\n{3,}")
_CTRL_RE = re.compile(r"[​-‏﻿­]")


def normalize_tone(text: str) -> str:
    """Map new-style tone placement (hoà) onto traditional style (hòa)."""
    for pat, repl in _TONE_RE:
        text = pat.sub(repl, text)
    return text


def normalize(text: str, lower: bool = True, tone: bool = True,
              keep_newlines: bool = False) -> str:
    """Canonical text normalisation. Idempotent.

    ``keep_newlines=True`` is REQUIRED when the text will later be split by
    ``src.chunking`` -- those regexes anchor on ``^``, so flattening newlines
    silently turns article-level chunking into a no-op. Use it for corpus text;
    the default is fine for queries.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = _CTRL_RE.sub("", text)
    text = text.replace(" ", " ")
    if tone:
        text = normalize_tone(text)
    if lower:
        text = text.lower()
    if keep_newlines:
        text = _HSPACE_RE.sub(" ", text.replace("\r\n", "\n").replace("\r", "\n"))
        text = _NL_RE.sub("\n\n", text)
        return "\n".join(ln.strip() for ln in text.split("\n")).strip()
    return _WS_RE.sub(" ", text).strip()


# --------------------------------------------------------------------------
# Word segmentation
# --------------------------------------------------------------------------

_SEGMENTER = None
_SEG_BACKEND = None


def _load_segmenter(backend: str):
    global _SEGMENTER, _SEG_BACKEND
    if _SEGMENTER is not None and _SEG_BACKEND == backend:
        return _SEGMENTER
    if backend == "pyvi":
        from pyvi import ViTokenizer  # type: ignore
        _SEGMENTER = ViTokenizer.tokenize
    elif backend == "underthesea":
        from underthesea import word_tokenize  # type: ignore
        _SEGMENTER = lambda s: word_tokenize(s, format="text")  # noqa: E731
    elif backend == "none":
        _SEGMENTER = lambda s: s  # noqa: E731
    else:
        raise ValueError(f"unknown segmenter backend: {backend!r}")
    _SEG_BACKEND = backend
    return _SEGMENTER


@lru_cache(maxsize=200_000)
def _segment_cached(text: str, backend: str) -> str:
    return _load_segmenter(backend)(text)


def segment(text: str, backend: str = "pyvi") -> str:
    """Word-segment Vietnamese text; multi-syllable words joined by '_'.

    Required for PhoBERT-backbone models. Cached, because segmenting a corpus
    twice is a common and entirely avoidable waste of an afternoon.
    """
    if not text:
        return ""
    return _segment_cached(text, backend)


# --------------------------------------------------------------------------
# BM25 tokenisation
# --------------------------------------------------------------------------

# Deliberately small. Legal phrasing leans on function words ("theo quy định
# của"), and BM25's IDF already discounts them. Default is OFF; turn it on only
# if the dev split says it helps.
STOPWORDS = {
    "và", "của", "có", "là", "được", "cho", "trong", "với", "các", "những",
    "này", "đó", "khi", "thì", "mà", "để", "từ", "đến", "về", "tại", "bởi",
    "hoặc", "nếu", "như", "một", "sẽ", "đã", "cũng", "vào", "ra", "nên",
}

_TOKEN_RE = re.compile(r"[" + re.escape(_VN_LETTERS) + r"0-9_/\.\-]+")


def tokenize(
    text: str,
    segmenter: str = "none",
    remove_stopwords: bool = False,
    min_len: int = 1,
) -> List[str]:
    """Normalise -> (optionally segment) -> split into BM25 terms.

    Keeps digits, '/', '.' and '-' inside tokens so that document identifiers
    such as ``100/2019/nđ-cp`` survive as a single high-IDF term.
    """
    t = normalize(text)
    if segmenter and segmenter != "none":
        t = segment(t, segmenter).lower()
    toks = _TOKEN_RE.findall(t)
    toks = [w.strip("./-") for w in toks]
    toks = [w for w in toks if len(w) >= min_len and w]
    if remove_stopwords:
        toks = [w for w in toks if w not in STOPWORDS]
    return toks


def batch_tokenize(texts: Iterable[str], **kw) -> List[List[str]]:
    return [tokenize(t, **kw) for t in texts]


# =============================================================================
# TODO(YOU/phase1): add legal-domain text handling here.
# -----------------------------------------------------------------------------
# WHY HERE: `normalize()` above is generic Vietnamese cleanup. It knows nothing
#   about legal writing. The corpus and the questions often say the SAME thing
#   differently, and every mismatch is a retrieval miss:
#       "NĐ-CP"  vs  "Nghị định"
#       "Đ.113"  vs  "Điều 113"
#       "BLLĐ"   vs  "Bộ luật Lao động"
#   Expanding those makes BM25 match where it currently misses.
#
# WHAT TO WRITE: take `text`, return `text` with substitutions applied. Add pairs
#   to ABBREVIATIONS below, or write your own logic. It is called for BOTH the
#   corpus and the questions, so a rule always applies to both sides.
#
# CAREFUL: only add a rule you are confident about. Expanding "khoản" -> "k."
#   the WRONG way round would break matches that currently work.
#
# HOW TO TEST IT:
#   1. python -c "import sys;sys.path.insert(0,'.');from src.normalize import \
#          expand_legal_abbreviations as e;print(e('Theo NĐ-CP 100/2019'))"
#   2. Re-run the BM25 baseline and compare recall in work/experiments/runs.csv.
#      If it does not go up, revert it -- and log that it did not help.
#
# PYTHON NOTE (from C++): a dict is std::unordered_map. `.items()` iterates
#   key/value pairs. `str.replace(a, b)` returns a NEW string (strings are
#   immutable), so you must assign the result back.
# =============================================================================
ABBREVIATIONS = {
    # "nđ-cp": "nghị định",     # <- uncomment / add your own, then measure
}


def expand_legal_abbreviations(text: str) -> str:
    """Expand legal shorthand so query and corpus use the same words."""
    for short, full in ABBREVIATIONS.items():
        text = text.replace(short, full)
    return text


def encoder_text(text: str, requires_segmentation: bool,
                 segmenter: str = "pyvi") -> str:
    """Prepare text for a neural encoder.

    ``requires_segmentation`` must mirror the model's backbone:
    PhoBERT -> True, BGE-M3/XLM-R -> False.
    """
    t = normalize(text, lower=False)
    return segment(t, segmenter) if requires_segmentation else t
