"""
text.py  --  Macedonian-aware tokenization shared by the eval harness and any
=============================================================================
lexical index built on top of it.

Deliberately dependency-free and deterministic: the same tokenizer must be used
by the BM25 baseline, the future lexical arm of hybrid retrieval, and by any
diagnostic that inspects term overlap -- otherwise "the retriever got worse"
can silently mean "the tokenizer changed".

Macedonian specifics
--------------------
* Cyrillic block U+0400-U+04FF covers the full MK alphabet (ѓ ѕ ј љ њ ќ џ).
* MK is morphologically rich and postfixes its definite article (куќа -> куќата
  -> куќите), so an unstemmed index loses a lot of recall. `stem()` strips a
  conservative, longest-first suffix list and never shortens a token below 4
  characters -- enough to merge article/plural forms without collapsing
  unrelated roots.
* Stopwords are pure function words only. Question words (колку, кои, како,
  каде, кога) are KEPT: they carry the user's intent (price / documents /
  procedure) and IDF already discounts them where they are genuinely common.
"""
from __future__ import annotations

import re
import unicodedata

# Latin is kept because the corpus mixes in URLs, form codes and abbreviations.
_TOKEN_RE = re.compile(r"[0-9A-Za-zЀ-ӿ]+")

STOPWORDS = frozenset("""
и на во со од за се да е дека а но или ли ќе го ја ги му им ни ме те
по до при што кој која кое кои чиј ова овој оваа овие тоа тој таа тие
не сум си сме сте се беше биле била било би
""".split())

# Longest first. Each entry is (suffix, min_len_of_token_to_strip).
# min_len guards against gutting short roots: "дом" must not become "до".
#
# Kept deliberately short. Rules like "-на"/"-ена" look like article endings but
# eat real roots (промена -> пром), and multi-consonant variants such as "-тите"
# are just <consonant> + "-ите", so matching them first mis-strips an extra
# character (документите -> докумен). Both were caught by selftest.py.
_SUFFIXES: tuple[tuple[str, int], ...] = (
    ("ите", 6), ("ови", 6), ("еви", 6), ("ата", 6), ("ото", 6), ("иот", 6),
    ("от", 5), ("та", 5), ("то", 5), ("те", 5), ("ов", 5), ("ев", 5), ("ја", 5),
    ("а", 5), ("и", 5), ("е", 5), ("о", 5),
)


def normalize(text: str) -> str:
    """NFKC-fold and lowercase. Applied identically to documents and queries."""
    return unicodedata.normalize("NFKC", text).lower()


def stem(token: str) -> str:
    """Strip one MK inflectional suffix (definite article / plural / gender).

    Single-pass and conservative on purpose: a second pass would merge
    "регистар"/"регистрација" style pairs that a legal-answer system should
    keep apart.
    """
    for suf, min_len in _SUFFIXES:
        if len(token) >= min_len and token.endswith(suf):
            return token[: -len(suf)]
    return token


def tokenize(text: str, *, do_stem: bool = True, drop_stopwords: bool = True) -> list[str]:
    """Normalize -> split -> (drop stopwords) -> (stem). Order matters: stopwords
    are matched on the surface form, before stemming mangles them."""
    out: list[str] = []
    for tok in _TOKEN_RE.findall(normalize(text)):
        if drop_stopwords and tok in STOPWORDS:
            continue
        out.append(stem(tok) if do_stem else tok)
    return out
