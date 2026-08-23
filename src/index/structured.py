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

from .intent import INTENT_CUES, detect_intent

# Sections are small: process maxes at 8 rows, tariffs 14, deadlines 7, forms 10,
# documentsLocations 3 -- so a whole section fits comfortably. Only `documents`
# runs long (max 33), and this cap is the only place truncation can bite.
MAX_ROWS_PER_SECTION = 15
RRF_K = 60


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


# Interrogatives and filler carry no topic, so they must never anchor a service.
#
# "регистр" was tried here and reverted: it is filler in "Како да регистрирам
# залог?" but the entire subject in "Колку чини регистрација?", where dropping it
# left no anchor at all and silenced the variation-ambiguity detector. A word
# cannot be generic in one query and the topic in the next, so the fix belongs in
# the stemmer (which no longer folds "регистрирам" onto it), not in this list.
_GENERIC_WORDS = ("како", "каде", "кога", "колку", "кој", "која", "кои", "што",
                  "дали", "може", "можам", "треба", "сакам", "имам", "ми",
                  "нешто", "друго", "тоа", "ова")

# A subject term is discriminating: it points at SOME services rather than all
# of them. Measured over the corpus, the split is clean at ~10% of blocks --
# услуга 98.6%, документ 25.1%, преку 17.7%, право 13.2%, чекор 11.9% are
# everywhere and name no subject, while залог 4.3%, заложно 3.1%, доверител
# 2.0%, изјава 1.9%, Гостивар 0.03% do.
_MAX_SUBJECT_DF = 0.10


def _anchor_terms(query: str) -> set[str]:
    from eval.text import stem, tokenize
    generic = {stem(w) for w in _GENERIC_WORDS}
    return {t for t in tokenize(query) if len(t) >= 4 and t not in generic}


def _document_frequency(corpus: Corpus) -> dict[str, int]:
    """How many blocks each stem appears in. Computed once per corpus."""
    cached = getattr(corpus, "_stem_df", None)
    if cached is not None:
        return cached
    from collections import Counter

    from eval.text import tokenize
    df: Counter[str] = Counter()
    for b in corpus:
        df.update(set(tokenize(b.embedding_text)))
    setattr(corpus, "_stem_df", df)
    return df


def subject_anchors(message: str, corpus: Corpus) -> set[str]:
    """Terms in the message that could pin a SUBJECT (a service or a place).

    Three filters, because no single one separates them. A subject term is:
      * not an interrogative or filler   (_anchor_terms)
      * not a SECTION word               -- документ / плаќ / рок name what you
        want to know, not what you want to know it about
      * discriminating in the corpus     -- present, but in under ~10% of blocks

    Used to decide whether a message may become the conversation topic. Live, a
    subject-free turn ("колку се плаќа и како се плаќа") became the topic and
    the next follow-up merged with nothing, landing on an unrelated service.
    """
    from eval.text import stem

    cue_stems = {stem(cue) for cues in INTENT_CUES.values() for cue in cues
                 if " " not in cue}
    df = _document_frequency(corpus)
    limit = _MAX_SUBJECT_DF * max(len(corpus), 1)

    out = set()
    for term in _anchor_terms(message):
        if any(term.startswith(c) or c.startswith(term) for c in cue_stems):
            continue
        if 0 < df.get(term, 0) <= limit:
            out.add(term)
    return out


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
        # TWO STAGES, and the order matters. Picking the strongest
        # (service, variation) pair in one step penalises a service for HAVING
        # variations: its evidence is divided among them while a single-variation
        # service keeps all of its own. Measured on
        # "Имам доказ ... на англиски јазик ... пред да го поднесам за бришење":
        # service 2135 had 9 hits with its best at rank 2, spread over nine legal
        # forms, so no pair of its exceeded the 7 hits that service 2103 -- one
        # variation, best rank 8, anchored only by the word "јазик" in a FAQ --
        # held in a single bucket. 2103 won, and structured fetch then PINNED its
        # rows, so a wrong attribution became the part of the context that could
        # not be trimmed. Deciding the service on its combined evidence first,
        # and the variation only among that service's own, removes the penalty.
        # Scored by RRF, not by counting. A raw count rewards a long tail of weak
        # hits: at depth 40 service 2103 reached 12 votes whose best was rank 8
        # and beat service 2135's 11 whose best was rank 2 -- the same wrong
        # answer arriving by a different route. RRF is what every other join in
        # this system uses to combine ranked evidence, and it discounts the tail
        # for the same reason here.
        by_service: dict[int, list[int]] = {}
        for (svc, _vid), ranks_ in votes.items():
            by_service.setdefault(svc, []).extend(ranks_)
        sid, service_ranks = max(
            by_service.items(),
            key=lambda kv: (sum(1.0 / (RRF_K + r) for r in kv[1]), -min(kv[1])))
        siblings = {vid_: r for (svc, vid_), r in votes.items() if svc == sid}
        vid, ranks = max(siblings.items(), key=lambda kv: (len(kv[1]), -min(kv[1])))
        solo = len(corpus.variations_of.get(sid, [])) <= 1
        if solo or len(ranks) >= 2:
            return Attribution(sid, vid, min(ranks))
        # Sibling evidence is too thin to name a form, but the service is not in
        # doubt -- report it with the service's own best rank.
        return Attribution(sid, None, min(service_ranks))

    for rank, h in enumerate(hits, 1):
        b = corpus.by_id.get(h.chunk_id)
        if b is not None and (not anchored or b.id_service in anchored):
            return Attribution(b.id_service, None, rank)
    return Attribution()


def resolve_variation(corpus: Corpus, id_service: int,
                      id_variation: int | None,
                      types: Sequence[str] = ()) -> int | None | bool:
    """The variation to fetch rows for. False means "refuse": the service has
    several legal forms and we cannot tell which one the user means.

    Refusing is only right when the choice CHANGES the rows. Service 2162 has one
    fee -- 295 МКД -- written once under "Хартиено на шалтер" and once under
    "Електронски"; when the corpus rebuild added that second variation, this
    refused to fetch either, no tariff row reached the context, and the agent
    said it had no information about a price it holds twice. The ambiguity
    detector had already stopped asking about that section for the same reason,
    so the two now agree: identical rows are not a choice.
    """
    if id_variation:
        return id_variation
    variations = corpus.variations_of.get(id_service, [])
    if len(variations) == 1:
        return variations[0]
    if len(variations) > 1:
        from agent.context import _differing_sections
        if types and not _differing_sections(corpus, id_service, list(types)):
            return variations[0]     # every variation says the same thing
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

    # A named legal form overrides whatever attribution guessed. Injected rows
    # are PINNED, so leaving this to attribution let the filter make things
    # strictly worse: scoping "Сакам да ликвидирам Здружение..." to Здружение
    # stripped the useful FAQ and terminology out of the semantic arms, while
    # structured fetch -- which never saw the filter -- flooded seven of the ten
    # slots with a DIFFERENT form's documents. The safety-critical false-premise
    # case regressed from 1.000 to 0.800 on five consecutive runs.
    if filters and filters.get("form_scope"):
        want = set(filters["form_scope"])
        named = [v for v in corpus.variations_of.get(attr.id_service, ())
                 if corpus.variation_name.get((attr.id_service, v)) in want]
        if not named:
            return []      # this service has no such form -- inject nothing
        attr = Attribution(attr.id_service, named[0], attr.rank)

    vid = resolve_variation(corpus, attr.id_service, attr.id_variation, types)
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


# A proper noun is rare by nature. "чамовск" is in 2 blocks of 6,069; "фирм" is
# in 192 and "агент" in 601. Anything this rare that the user typed is almost
# certainly the exact thing they are asking about.
ENTITY_MAX_DF = 8
# Two, because a person is a first name AND a surname. One rare token on its own
# is as likely to be an unusual topic word, and intersecting two makes a false
# match essentially impossible: a block must contain both.
ENTITY_MIN_TERMS = 2


def _rare_term_index(corpus: Corpus) -> dict[str, set[str]]:
    """term -> chunk_ids, for rare terms only. Built once; small by construction."""
    cached = getattr(corpus, "_rare_terms", None)
    if cached is not None:
        return cached
    from eval.text import tokenize
    postings: dict[str, set[str]] = {}
    for b in corpus:
        for t in set(tokenize(b.embedding_text)):
            if len(t) >= 4:
                postings.setdefault(t, set()).add(b.chunk_id)
    out = {t: ids for t, ids in postings.items() if len(ids) <= ENTITY_MAX_DF}
    setattr(corpus, "_rare_terms", out)
    return out


def fetch_entities(corpus: Corpus, query: str, *, max_blocks: int = 4) -> list[str]:
    """Blocks containing EVERY rare term the query names -- exact-entity search.

    The dense arm averages a question into one vector, so a name competes with
    the procedure around it and loses. Measured live: "Кој е телефонот на Моника
    Чамовска?" returned her directory page at rank 1, while "Дали можам да
    регистрирам фирма кај Моника Чамовска?" returned NOTHING from the directory
    -- the same name, drowned by "регистрирам фирма". A person's name is not a
    topic to be softly matched; it either appears in a block or it does not.
    """
    from eval.text import tokenize
    rare = _rare_term_index(corpus)
    generic = {stem_w for stem_w in _GENERIC_WORDS}
    terms = [t for t in dict.fromkeys(tokenize(query))
             if t in rare and t not in generic]
    if len(terms) < ENTITY_MIN_TERMS:
        return []
    # The BEST-covered blocks, not blocks covering every term. A verb can be rare
    # too -- "регистрирам" is in 3 blocks -- and demanding the full intersection
    # let one such word empty the result: {моник} ∩ {чамовск} ∩ {регистрирам} is
    # nothing, so the name it was meant to find was thrown away with it.
    from collections import Counter
    counts: Counter[str] = Counter()
    for t in terms:
        counts.update(rare[t])
    best = max(counts.values(), default=0)
    if best < ENTITY_MIN_TERMS:
        return []
    common = {cid for cid, n in counts.items() if n == best}
    # Registry order, so a multi-part directory list stays readable.
    order = {b.chunk_id: i for i, b in enumerate(corpus)}
    return sorted(common, key=lambda cid: order.get(cid, 0))[:max_blocks]


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
        rows += [c for c in fetch_entities(self.corpus, query) if c not in rows]
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
