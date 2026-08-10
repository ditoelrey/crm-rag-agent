"""
structured.py  --  fetch the answer rows instead of hoping they rank.
=====================================================================
The measured failure this exists for:

    "Како да регистрирам залог?"  ->  hit@10 = 0.000

The retriever put service 2063 at the top of every slot (service_acc@1 = 1.000)
and returned not one `process` row. The agent then assembled a confident,
fully-cited four-step procedure out of the description, documents and forms
blocks -- silently dropping steps 3, 4 and 5 of the registry's six, including
what happens when an application is refused. Grounded, plausible, incomplete,
and undetectable by the person reading it.

Semantic search cannot fix that. `Број на чекор: 4 | ЦРРСМ врши обработка на
пријавата...` shares no vocabulary with "како да регистрирам залог"; there is no
word to bridge and so no alias can help. But once the SERVICE is known -- and
routing is now the reliable part of this system -- the rows are a lookup:
`corpus.of_type(2063, "process", 11187)` returns all six, in order, always.

So: resolve (service, variation) from what the retriever DID return, map the
question to a section type, and fetch that section directly.

Failure containment
-------------------
Structured rows are FUSED with the semantic results (RRF, as everywhere else in
this codebase), never substituted for them. A wrong intent guess or a wrong
service attribution then costs some ranking slots instead of replacing a correct
answer with a confident wrong one.

Three refusals to guess, each of which returns nothing rather than something:
  * no readable intent            -> nothing to fetch
  * no service attributable       -> nothing to fetch
  * a multi-variation service whose variation could not be pinned down ->
    nothing to fetch, because АД's fees are not Здружение's, and the agent's
    ambiguity detector will ask the user instead.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Sequence

from eval.corpus import Corpus
from eval.corpus import load as load_corpus
from eval.retriever import Hit

from .intent import detect_intent

# Sections are small: process maxes at 8 rows, tariffs 14, deadlines 7, forms 10,
# documentsLocations 3 -- so a whole section fits comfortably. Only `documents`
# runs long (max 33), and this cap is the only place truncation can bite.
MAX_ROWS_PER_SECTION = 15
RRF_K = 60


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


# Interrogatives and filler carry no topic, so they must never anchor a service.
_GENERIC_WORDS = ("како", "каде", "кога", "колку", "кој", "која", "кои", "што",
                  "дали", "може", "можам", "треба", "сакам", "имам", "ми")


def _anchor_terms(query: str) -> set[str]:
    from eval.text import stem, tokenize
    generic = {stem(w) for w in _GENERIC_WORDS}
    return {t for t in tokenize(query) if len(t) >= 4 and t not in generic}


def anchored_services(hits: Sequence[Hit], corpus: Corpus, query: str) -> set[int]:
    """Services the ORIGINAL question actually mentions.

    Alias expansion appends registry vocabulary to the query, which is what
    makes underspecified questions retrievable -- but the expanded terms can
    also pull in an unrelated service, and structured fetch then AMPLIFIES that
    by injecting the whole section.

    Observed live: "Како да регистрирам залог?" had "упис, основање" appended,
    the dense arm returned six `process` rows from service 2135 variant
    Фондација, attribution locked onto it, and the agent answered a pledge
    question with foundation-registration steps -- then asked which legal form.
    Anchoring attribution to the words the user actually typed ("залог") keeps
    the expansion helping recall without letting it choose the subject.
    """
    terms = _anchor_terms(query)
    if not terms:
        return set()
    from eval.text import tokenize

    out: set[int] = set()
    for h in hits:
        b = corpus.by_id.get(h.chunk_id)
        if b is None or b.id_service is None or b.id_service in out:
            continue
        if terms & set(tokenize(f"{b.service_name} {b.content}")):
            out.add(b.id_service)
    return out


@dataclass
class Attribution:
    id_service: int | None = None
    id_variation: int | None = None
    rank: int | None = None          # rank of the hit it came from


def attribute(hits: Sequence[Hit], corpus: Corpus,
              filters: dict[str, Any] | None = None,
              anchor_query: str | None = None) -> Attribution:
    """Which (service, variation) is this result set about?

    Only blocks with `id_variation > 0` are counted as evidence: shared
    terminology and FAQ text is boilerplate the portal replicates across
    services, so it identifies a topic but not a service. Same rule as the
    agent's ambiguity detector, and for the same reason.
    """
    if filters and filters.get("variation_scope"):
        vid = filters["variation_scope"]
        for rank, h in enumerate(hits, 1):
            b = corpus.by_id.get(h.chunk_id)
            if b is not None and b.id_variation == vid:
                return Attribution(b.id_service, vid, rank)

    # Corroboration, not a single hit. Attributing a variation from one result
    # is exactly the sibling-confusion this corpus punishes: a service can have
    # nine near-identical legal forms, and structured fetch AMPLIFIES a wrong
    # guess -- it injects that sibling's whole section, flooding the context
    # with rows that answer a question nobody asked. Measured: trusting a single
    # hit dropped variation_documents hit@10 from 0.846 to 0.615.
    anchored = anchored_services(hits, corpus, anchor_query) if anchor_query else set()

    votes: dict[tuple[int, int], list[int]] = {}
    for rank, h in enumerate(hits, 1):
        b = corpus.by_id.get(h.chunk_id)
        if b is not None and b.id_variation and not b.is_shared:
            if anchored and b.id_service not in anchored:
                continue          # the query never mentioned this service
            votes.setdefault((b.id_service, b.id_variation), []).append(rank)
    if votes:
        (sid, vid), ranks = max(votes.items(), key=lambda kv: (len(kv[1]), -min(kv[1])))
        solo = len(corpus.variations_of.get(sid, [])) <= 1
        if solo or len(ranks) >= 2:
            return Attribution(sid, vid, min(ranks))
        return Attribution(sid, None, min(ranks))

    for rank, h in enumerate(hits, 1):
        b = corpus.by_id.get(h.chunk_id)
        if b is not None and (not anchored or b.id_service in anchored):
            return Attribution(b.id_service, None, rank)
    return Attribution()


def resolve_variation(corpus: Corpus, id_service: int,
                      id_variation: int | None) -> int | None | bool:
    """The variation to fetch rows for. False means "refuse": the service has
    several legal forms and we cannot tell which one the user means."""
    if id_variation:
        return id_variation
    variations = corpus.variations_of.get(id_service, [])
    if len(variations) == 1:
        return variations[0]
    if len(variations) > 1:
        return False
    return None                      # service with no variations at all


def fetch_sections(corpus: Corpus, query: str, hits: Sequence[Hit], *,
                   filters: dict[str, Any] | None = None,
                   max_rows: int = MAX_ROWS_PER_SECTION,
                   intents: Sequence[str] | None = None) -> list[str]:
    """chunk_ids of the sections this question asks for, in the registry's own
    row order. Empty whenever anything is uncertain."""
    types = list(intents) if intents is not None else detect_intent(query, top_only=True)
    if not types:
        return []

    attr = attribute(hits, corpus, filters, anchor_query=query)
    if attr.id_service is None:
        return []

    vid = resolve_variation(corpus, attr.id_service, attr.id_variation)
    if vid is False:
        return []

    out: list[str] = []
    for type_ in types:
        rows = corpus.of_type(attr.id_service, type_, vid)
        if not rows:
            # Shared sections (terminology, legalBasis) carry no variation.
            rows = [b for b in corpus.of_type(attr.id_service, type_, None)
                    if b.is_shared]
        for b in rows[:max_rows]:
            if b.chunk_id not in out:
                out.append(b.chunk_id)
    return out


# --------------------------------------------------------------------------- #
# agent directory
# --------------------------------------------------------------------------- #
# A municipality name in a service question ("Колку чини регистрација во
# Гостивар?") is not a request for the agent directory. Both signals are
# required.
AGENT_CUES = ("агент", "адвокат", "полномошн")


def detect_municipalities(query: str, corpus: Corpus) -> list[str]:
    """Municipalities named in the query, expanded through MUNICIPALITY_GROUPS.

    Longest name first, so "ГАЗИ БАБА" is not shadowed by a shorter match, and
    matched on a left word boundary so "ЦЕНТАР" does not fire inside
    "во центарот на градот".
    """
    from .aliases import MUNICIPALITY_GROUPS

    q = _norm(query)
    known = set(corpus.by_municipality)
    found: list[str] = []

    for group, members in MUNICIPALITY_GROUPS.items():
        if re.search(rf"(?<!\w){re.escape(_norm(group))}", q):
            found += [m for m in members if m in known]

    for name in sorted(known, key=len, reverse=True):
        if name in found:
            continue
        if re.search(rf"(?<!\w){re.escape(_norm(name))}", q):
            found.append(name)
    return found


def fetch_directory(corpus: Corpus, query: str, *,
                    max_blocks: int = 6) -> list[str]:
    """Agent-directory blocks for the municipalities named in the question.

    Ordered by part number across every (list, municipality) pair, so a budget
    that cannot fit everything still returns the FIRST part of each list rather
    than nine parts of one. Both lists are kept distinct: they carry different
    scopes of authority (founding only vs founding, changes and deletions).
    """
    q = _norm(query)
    if not any(re.search(rf"(?<!\w){re.escape(cue)}", q) for cue in AGENT_CUES):
        return []
    names = detect_municipalities(query, corpus)
    if not names:
        return []

    blocks = [b for name in names for b in corpus.by_municipality.get(name, ())]
    blocks.sort(key=lambda b: (b.part or 1, b.list_key or "", b.municipality or ""))
    return [b.chunk_id for b in blocks[:max_blocks]]


class SectionFetchRetriever:
    """Wraps any retriever, fusing in the section rows the question asks for."""

    def __init__(self, inner, corpus: Corpus, *, depth: int = 30,
                 max_rows: int = MAX_ROWS_PER_SECTION, rrf_k: int = RRF_K,
                 w_semantic: float = 1.0, w_structured: float = 1.0):
        self.inner, self.corpus = inner, corpus
        self.depth, self.max_rows, self.rrf_k = depth, max_rows, rrf_k
        self.w_semantic, self.w_structured = w_semantic, w_structured
        self.name = f"struct+{inner.name}"
        self.last_sections: list[str] = []

    def _inner_search(self, query: str, k: int, filters) -> list[Hit]:
        try:
            return list(self.inner.search(query, k, filters=filters))
        except TypeError:
            return list(self.inner.search(query, k))

    def search(self, query: str, k: int, *,
               filters: dict[str, Any] | None = None) -> list[Hit]:
        semantic = self._inner_search(query, max(k, self.depth), filters)
        # Never let injection take every slot: a wrong service attribution then
        # floods the whole context with rows answering a question nobody asked.
        # Half the slots stay with the semantic arm, which is what recovers a
        # bad attribution. The agent over-fetches (k*4), so it still receives
        # complete sections -- the cap only binds at small k.
        budget = min(self.max_rows, max(3, k // 2 + 1))
        rows = fetch_directory(self.corpus, query, max_blocks=budget)
        rows += [c for c in fetch_sections(self.corpus, query, semantic,
                                           filters=filters, max_rows=budget)
                 if c not in rows]
        self.last_sections = rows
        if not rows:
            return semantic[:k]

        scores: dict[str, float] = {}
        for arm, weight in ((semantic, self.w_semantic),
                            ([Hit(c, 0.0) for c in rows], self.w_structured)):
            for rank, hit in enumerate(arm, 1):
                scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (self.rrf_k + rank)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [Hit(cid, score) for cid, score in ranked]

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if close:
            close()


# --------------------------------------------------------------------------- #
# factories
# --------------------------------------------------------------------------- #
def build(corpus: Corpus | None = None):
    """Structured fetch over alias-expanded hybrid -- the full stack."""
    from .aliases import AliasExpandingRetriever
    from .hybrid import HybridRetriever
    c = corpus or load_corpus()
    return SectionFetchRetriever(AliasExpandingRetriever(HybridRetriever(c), c), c)


def build_plain(corpus: Corpus | None = None):
    """Structured fetch over hybrid WITHOUT aliases -- isolates this layer."""
    from .hybrid import HybridRetriever
    c = corpus or load_corpus()
    return SectionFetchRetriever(HybridRetriever(c), c)


def build_lexical(corpus: Corpus | None = None):
    """Structured fetch over BM25 -- measurable offline, no API key."""
    from eval.baselines import BM25Retriever
    c = corpus or load_corpus()
    return SectionFetchRetriever(BM25Retriever(c), c)
