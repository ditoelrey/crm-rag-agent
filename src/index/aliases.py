"""
aliases.py  --  bridging user vocabulary to registry vocabulary.
================================================================
Two live failures traced to the same cause: the words people use are not the
words the registry writes.

    "Како се регистрира залог врз машина?"  -> "машина" appears 0 times in the
                                               corpus; it says "подвижни предмети"
    "Колку чини регистрација?"              -> matched blocks about registering a
                                               USER ACCOUNT; the fee rows say
                                               "упис на основање"

Two sources, deliberately different in kind:

DERIVED   The registry defines its own abbreviations in glossary blocks --
          "Термин: Акционерско друштво (АД) | Објаснување: ...". Parsing those
          yields АД, ДОО, ПДОО, ТП, ЈТД, КД, КДА, СИЗ and more, for free, and
          the map updates itself when the corpus is rebuilt. Nothing to maintain
          and nothing invented.

CURATED   Genuine vocabulary gaps that no glossary covers. Every entry below is
          justified by counting both words in the corpus, because an alias is
          only worth adding if the target is what the text actually says. The
          map is deliberately small: guessed synonyms dilute a dense query
          without helping, so entries are added from observed failures, not
          from imagination.

Expansion is applied as a retriever wrapper, so it is an A/B switch rather than
a behaviour change baked into the index:

    python -m eval.cli run --retriever index.hybrid:build_aliased --tag hybrid_alias \\
           --baseline eval/reports/hybrid.json
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Sequence

from eval.corpus import Corpus
from eval.corpus import load as load_corpus
from eval.retriever import Hit

# user word (prefix, matched on a left word boundary) -> what the registry says.
# The counts are occurrences in embedding_text, measured 2026-07-28.
CURATED_ALIASES: dict[str, tuple[str, ...]] = {
    # "машина" is absent from the corpus entirely; pledges are over "подвижни
    # предмети". This single gap sank the залог case on all three retrievers.
    "машин": ("подвижни предмети",),            #    0 vs 430
    "возил": ("подвижни предмети",),            #    0 vs 430
    # A user registers a "фирма"; the registry registers a "субјект".
    "фирм": ("субјект", "трговско друштво"),    #  192 vs 1680 / 334
    # Registration is filed as an "упис на основање".
    "регистрациј": ("упис", "основање"),        #  360 vs 2235 / 830
    "регистрир": ("упис", "основање"),
    "отвор": ("основање", "упис"),              #   39 vs 830
    # Closing a company is a "бришење", never a "гасење".
    "гасењ": ("бришење", "ликвидација"),        #    0 vs 1081 / 727
    "затвор": ("бришење", "ликвидација"),       #   36 vs 1081
    # Fees live under "тарифа"/"надоместок".
    "цена": ("тарифа", "надоместок"),           #  153 vs 490 / 79
    "цени": ("тарифа", "надоместок"),
    "кошта": ("тарифа",),
}

# "Термин: <full name> (<ABBR>) | Објаснување: ..."
_GLOSSARY_RE = re.compile(
    r"^Термин:\s*(.+?)\s*\(\s*([А-ШЀ-ӿA-Z][А-ШЀ-ӿA-Z0-9]{1,7})\s*\)\s*\|")

# ПДОО is the corpus label for the variation whose tariff row reads
# "Упис на основање на ДОО / ДООЕЛ" -- so a user asking about a ДОО or ДООЕЛ is
# asking about the ПДОО variation, and would never recognise "ПДОО" in a list of
# options. This one cannot be derived; it comes from the tariff description.
FORM_EXTRA_SYNONYMS: dict[str, tuple[str, ...]] = {
    # ПДОО is "Поедноставено ДОО", a SIMPLIFIED fast-track incorporation, and it
    # is a different procedure from an ordinary ДОО. It used to carry ДОО/ДООЕЛ
    # as synonyms because the real "ДОО, ДООЕЛ" variation was missing from the
    # corpus -- 102 of 217 variations were never fetched -- so "доо" had nowhere
    # else to land. It has somewhere now, and sending an ordinary ДОО down the
    # simplified track would be a wrong answer, not a near miss.
    "ПДОО": ("поедноставено доо", "pdoo"),
    "ДОО, ДООЕЛ": ("друштво со ограничена одговорност", "doo", "dooel"),
    # Latin spellings: Macedonian keyboards are not universal and these
    # abbreviations get typed in Latin constantly.
    "АД": ("ad", "a.d.", "акционерско друштво"),
    "ТП": ("tp", "трговец поединец"),
}

# Not every variant is a legal form: some are the CHANNEL you use the service
# through. These labels are the registry's, and nobody types them -- a user says
# "преку интернет" or "на шалтер". Asked to choose between "Web сервис" and
# "Хартиено на шалтер", a live session answered "преку интернет", was asked
# again, answered "шалтер", and was asked a third time: the resolver only
# accepted the label verbatim, so the menu could not be got out of.
#
# The glossary cannot supply these -- it expands abbreviations, and these are
# not abbreviations of anything. Note "шалтер" appears here as a variant name
# while intent.py deliberately excludes it as a section cue; the two uses do not
# conflict, because this one only ever runs against a known list of variants.
CHANNEL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "Електронски": ("интернет", "преку интернет", "онлајн", "електронски",
                    "по електронски пат", "дигитално"),
    "Web сервис": ("интернет", "преку интернет", "онлајн", "веб сервис",
                   "web", "веб", "апликација"),
    "Хартиено на шалтер": ("шалтер", "на шалтер", "хартиено", "хартија",
                           "лично", "во филијала"),
    "Хартиено": ("хартиено", "хартија", "на хартија", "печатено"),
}

# Skopje is administratively ten municipalities, and the agent lists file each
# agent under one of them (plus a literal "СКОПЈЕ" bucket). 42-50% of all
# authorised agents sit inside this group, so "агенти во Скопје" has to reach
# every one -- looking up the literal "СКОПЈЕ" alone would return a third of
# them and call it the list.
MUNICIPALITY_GROUPS: dict[str, tuple[str, ...]] = {
    "СКОПЈЕ": ("СКОПЈЕ", "ЦЕНТАР", "КАРПОШ", "АЕРОДРОМ", "ЧАИР", "ГАЗИ БАБА",
               "КИСЕЛА ВОДА", "БУТЕЛ", "ШУТО ОРИЗАРИ", "ЃОРЧЕ ПЕТРОВ", "САРАЈ"),
}


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def abbreviations(corpus: Corpus) -> dict[str, str]:
    """ABBR -> full term, parsed from the corpus's own glossary blocks."""
    cached = getattr(corpus, "_abbreviations", None)
    if cached is not None:
        return cached
    out: dict[str, str] = {}
    for b in corpus:
        if b.type != "terminology":
            continue
        m = _GLOSSARY_RE.match(b.content)
        if m:
            out.setdefault(m.group(2), " ".join(m.group(1).split()))
    setattr(corpus, "_abbreviations", out)
    return out


def form_synonyms(label: str, corpus: Corpus) -> tuple[str, ...]:
    """Everything a user might type meaning this legal form.

    "АД" -> ("акционерско друштво",);  "ПДОО" -> ("ДОО", "ДООЕЛ", ...).
    Used both for matching a clarification reply and for showing options the
    user can actually recognise.
    """
    out: list[str] = []
    full = abbreviations(corpus).get(label.strip())
    if full:
        out.append(full)
    out.extend(FORM_EXTRA_SYNONYMS.get(label.strip(), ()))
    out.extend(CHANNEL_SYNONYMS.get(label.strip(), ()))
    seen, uniq = set(), []
    for s in out:
        if _norm(s) not in seen and _norm(s) != _norm(label):
            seen.add(_norm(s))
            uniq.append(s)
    return tuple(uniq)


def expand_query(query: str, corpus: Corpus | None = None, *,
                 max_terms: int = 6) -> tuple[str, list[str]]:
    """Append registry vocabulary for any user vocabulary found in the query.

    Returns (expanded_query, added_terms). Capped: a long tail of synonyms
    shifts a dense query's embedding away from the question that was actually
    asked, which costs more than the recall it buys.
    """
    q = _norm(query)
    added: list[str] = []

    def add(terms: Iterable[str]) -> None:
        for t in terms:
            if len(added) >= max_terms:
                return
            if _norm(t) not in q and t not in added:
                added.append(t)

    for prefix, targets in CURATED_ALIASES.items():
        if re.search(rf"(?<!\w){re.escape(prefix)}", q):
            add(targets)

    # The glossary abbreviations are deliberately NOT used for query expansion.
    # Measured: expanding them fired 13 times across the gold set and helped
    # nothing, while actively hurting -- "ТП" was expanded to "Трговец –
    # поединец" inside the variation label "Подружница на странско друштво и
    # странски ТП", injecting a different legal form into the query. They earn
    # their keep in form_synonyms(), matching and displaying legal forms, where
    # the string being matched is known to be a form name.

    if not added:
        return query, []
    return f"{query} {' '.join(added)}", added


class AliasExpandingRetriever:
    """Wraps any retriever, fusing results for the original and expanded query.

    Expansion ADDS a second query rather than replacing the first. Replacing it
    measured as a net loss: it lifted the underspecified questions it was built
    for (+0.050 ndcg@10) but regressed seven other families, because a query
    that already names its service precisely gets diluted by synonyms --
    "Која е цената за Потврда ... од фидуцијарен регистар" fell 0.875 -> 0.438
    once fee synonyms pulled in every other service's tariff rows.

    Fusing by rank (same RRF as the hybrid retriever, and for the same reason:
    the two result sets have no comparable score scale) keeps whatever the
    original query got right and lets the expansion contribute only what the
    original missed entirely.

    Cost: expansion fires on a minority of queries (46 of 512 in the gold set),
    and only those pay for a second search -- which for a dense arm means a
    second embedding call.
    """

    RRF_K = 60

    def __init__(self, inner, corpus: Corpus, *, max_terms: int = 6,
                 depth: int = 30):
        self.inner, self.corpus, self.max_terms = inner, corpus, max_terms
        self.depth = depth
        self.name = f"alias-rrf+{inner.name}"
        self.last_added: list[str] = []

    def _search(self, query: str, k: int, filters):
        try:
            return list(self.inner.search(query, k, filters=filters))
        except TypeError:                    # retriever without a filters kwarg
            return list(self.inner.search(query, k))

    def search(self, query: str, k: int, *, filters: dict[str, Any] | None = None):
        expanded, added = expand_query(query, self.corpus, max_terms=self.max_terms)
        self.last_added = added
        if not added:
            return self._search(query, k, filters)

        depth = max(k, self.depth)
        scores: dict[str, float] = {}
        best: dict[str, float] = {}
        for arm in (self._search(query, depth, filters),
                    self._search(expanded, depth, filters)):
            for rank, hit in enumerate(arm, 1):
                scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (self.RRF_K + rank)
                best[hit.chunk_id] = max(best.get(hit.chunk_id, 0.0), float(hit.score))
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [Hit(cid, score) for cid, score in ranked]

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if close:
            close()


def build_lexical_aliased(corpus: Corpus | None = None) -> AliasExpandingRetriever:
    """BM25 + aliases. Needs no API key, so the alias layer can be measured
    offline before spending anything on embeddings."""
    from eval.baselines import BM25Retriever
    c = corpus or load_corpus()
    return AliasExpandingRetriever(BM25Retriever(c), c)
