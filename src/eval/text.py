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


# DERIVATIONAL suffixes: they turn a verb into its action noun, or the other way
# round. The registry writes nouns ("бришење", "ликвидација", "поднесување");
# users write verbs ("избришам", "ликвидирам", "поднесам"). Without this the
# lexical arm contributes NOTHING to a question phrased as a verb -- measured
# live three times: ликвидирам/ликвидација, избришам/бришење, and a "Сакам да
# избришам залог" that retrieved zero blocks of the pledge-deletion service.
#
# These run BEFORE the inflectional pass, on the raw token. Order matters:
# "ликвидација" inflects to "ликвидаци" first, and no derivational rule would
# ever see the "ација" it needs.
#
# Longest first; min_len keeps a short root from being gutted.
#
# Measured, not assumed. Each family was A/B'd alone and in combination on the
# 544-case harness (lexical arm, ndcg@10, baseline 0.7568):
#
#     -ација / -ирање         +0.0014
#     -ување                  +0.0013
#     ^^ the two above, together:       0.7595  (+0.0027)   <-- kept
#     -ење / -ање             -0.0026
#     -ам                     -0.0024
#     -ирам                   +0.0000   (rejected -- see below)
#
# On the full stack (struct+alias+hybrid) the same pair is +0.0017 ndcg@10
# (0.8110 -> 0.8128), and variation_acc@1 +0.0063 (0.8994 -> 0.9057).
#
# -ење/-ање/-ам collapse variation_documents 0.849 -> 0.715 and terminology
# 0.664 -> 0.625: they are short and productive enough to erode the IDF of the
# registry's own discriminative nouns. So "бришам"/"бришење" still do NOT merge
# here; that pair belongs query-side in index/aliases.py, which can only ADD
# candidates via RRF and never reweights the index.
#
# -ирам was tried and rejected even though it looks like the natural partner of
# -ација. It is free on the harness but not free in the system: it folds the verb
# "регистрирам" (df=3, inert) onto "регистр" (df=187), which then out-votes the
# real subject in subject_anchors() -- "Како да регистрирам залог?" was attributed
# to Фондација instead of the pledge service. Caught by agent.selftest. The
# compensating fix (marking "регистр" generic) breaks the opposite case, where it
# is the whole subject: "Колку чини регистрација?".
_DERIVATIONAL: tuple[tuple[str, int], ...] = (
    ("ирање", 8), ("ување", 8), ("ација", 7),
)


def stem(token: str) -> str:
    """Strip one derivational and then one inflectional MK suffix.

    Still conservative: at most one suffix from each list, never below the
    minimum length, and no prefix handling -- "избришам" keeps its "из-", so it
    does not merge with "бришење". Verb prefixes in Macedonian change meaning
    often enough (јава / изјава) that stripping them would merge words a
    legal-answer system must keep apart.
    """
    for suf, min_len in _DERIVATIONAL:
        if len(token) >= min_len and token.endswith(suf):
            token = token[: -len(suf)]
            break
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
