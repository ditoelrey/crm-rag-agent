"""
agent.py  --  the RAG agent: hybrid retrieval -> XML context -> OpenAI -> answer.
=================================================================================
`CRMAgent.ask()` returns a structured `Answer`, not a string. That is deliberate:
the retrieval harness in src/eval already knows how to run something in batch,
aggregate per-slice metrics and gate on regressions, and answer quality will be
scored the same way. An agent that only returns prose cannot be evaluated.

Multi-turn clarification
------------------------
Asking "За каква правна форма станува збор?" is only useful if the answer to it
goes somewhere. When a turn was flagged ambiguous and the user replies, the next
turn:
  1. retrieves on the ORIGINAL question plus the reply (the reply alone -- "АД"
     -- retrieves nothing useful on its own), and
  2. if the reply names one of the candidate legal forms, restricts retrieval to
     that variation with a `variation_scope` filter, which also keeps the
     service-level shared blocks that apply to every form.
Both steps are deterministic; neither costs an extra model call.
"""
from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Sequence

from eval.corpus import Corpus
from eval.corpus import load as load_corpus
from eval.retriever import Hit

from .context import (Ambiguity, ContextDoc, _attributed_service, build_context,
                      _collapse_dotted, _names_absent_form,
                      clarification_text, named_forms,
                      validate_citations)
from .prompts import SYSTEM_PROMPT, context_message

# gpt-4o, chosen on a 20-query A/B against gpt-4o-mini. On the curated suite the
# gap is one point (0.9974 vs 0.9872) and 4o costs ~13.6x, which on its own would
# not justify it -- the case for it is made by behaviour the suite does not yet
# cover. Given documents for a FOREIGN sole trader when the question was about a
# domestic one, mini answered from them and 4o abstained; same on the tender
# package when the word "шалтер" pulled in generic counter documents. For a
# government-facing bot, correctly recognising that retrieved text does not apply
# is worth more than the cost multiple.
DEFAULT_MODEL = "gpt-4o"
DEFAULT_K = 10
DEFAULT_TEMPERATURE = 0.0
# The OpenAI SDK defaults to a 600-second timeout; combined with retries a single
# stuck request can block a batch run for over half an hour. Observed: an answer
# eval hung for ten minutes on one case with no way to tell it apart from slow
# generation. Long list answers here take ~20s at p95, so 120s is generous.
DEFAULT_TIMEOUT = 120.0
MAX_HISTORY_TURNS = 4
# How much of a past answer survives into the next turn's history.
#
# Observed live: turn 1 listed all 36 authorised agents in Гостивар. Six turns
# later, asked for "a few examples" of agents in СТРУМИЦА, the model produced
# "АДВОКАТ ТОМЕ ЃОРЃЕВИЌ, Ул. БОРЧЕ ЈОВАНОСКИ Бр.56 ГОСТИВАР" -- fusing a
# Струмица first name with a Гостивар surname and address. That person does not
# exist. It cited nothing, because there was nothing to cite: it was reading the
# 2,500-token list still sitting in its own message history.
#
# Past answers are kept as conversational memory, not as a data source. The
# retrieval layer re-fetches on every turn, so nothing that matters is lost --
# only the temptation to answer from stale text.
HISTORY_ANSWER_CHARS = 500
# Hits requested per document wanted. Dedup collapses byte-identical copies, and
# without over-fetching a k=10 turn can arrive with only 2 documents.
OVERFETCH = 4

# A follow-up is only treated as one on a strong signal: mis-classifying a real
# question as a follow-up glues an unrelated topic onto the query, while missing
# a follow-up merely costs the context we had. "Како да регистрирам залог?" is
# four tokens and must stay self-contained.
_DISCOURSE_OPENERS = frozenset("а ама и ок добро значи па аха".split())
_ANAPHORS = frozenset("тоа тој таа тие ова овој оваа овие истото истата истиот "
                      "него неа нив таму тогаш".split())

# Requests for MORE of what was just discussed. They carry no subject of their
# own, so they are follow-ups however long they are.
#
# Observed: "Дај ми неколку како пример" is five tokens with no discourse opener
# and no anaphor, so it was treated as a new question, retrieved on the bare
# phrase and found nothing -- and the agent then said it had no information
# about Струмица agents, one turn after listing forty-five of them. Safer than
# the fabrication it replaced, still a contradiction.
#
# Deliberately phrases, not bare verbs: "Дај ми го ЗП образецот" opens the same
# way but names its own subject and must stay self-contained.
_CONTINUATION = ("како пример", "за пример", "уште", "неколку", "повеќе",
                 "друг пример", "останати", "сите други", "на пример")

# USD per 1M tokens, as configured -- update if OpenAI's pricing changes.
PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


@dataclass
class Answer:
    query: str
    text: str
    docs: list[ContextDoc]
    citations: list[str]
    stale_citations: list[str]
    invalid_citations: list[str]
    ambiguity: Ambiguity | None
    retrieval_query: str
    filters: dict[str, Any] | None
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0

    @property
    def asked_for_clarification(self) -> bool:
        return self.ambiguity is not None

    @property
    def cited_docs(self) -> list[ContextDoc]:
        cited = set(self.citations)
        return [d for d in self.docs if d.chunk_id in cited]


# An answer that cites nothing must SAY it has nothing. These are the phrasings
# that count as saying so; anything else with no citation is replaced outright.
_ABSTAIN_MARKERS = ("нема информација", "немам информација", "не располагам",
                    "не постои информација", "не е наведено", "нема податоци",
                    "не можам да", "не се наведени")
HARD_ABSTENTION = ("Во документацијата со која располагам нема информација "
                   "за ова прашање.")


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _for_history(text: str) -> str:
    """Trim a past answer down to conversational memory (see HISTORY_ANSWER_CHARS)."""
    if len(text) <= HISTORY_ANSWER_CHARS:
        return text
    kept = text[:HISTORY_ANSWER_CHARS].rsplit("\n", 1)[0]
    return kept + ("\n[... остатокот од претходниот одговор не е достапен во "
                   "овој чекор; податоците мора повторно да се побараат ...]")


def _carries_subject(message: str, corpus: Corpus) -> bool:
    """Does this message name something to be about? See index.structured."""
    from index.structured import subject_anchors
    return bool(subject_anchors(message, corpus))


def is_followup(message: str) -> bool:
    """Is this a continuation of the previous question rather than a new one?"""
    from eval.text import tokenize
    toks = tokenize(message, do_stem=False, drop_stopwords=False)
    if not toks:
        return False
    lowered = message.casefold()
    return (len(toks) <= 3
            or toks[0] in _DISCOURSE_OPENERS
            or any(t in _ANAPHORS for t in toks)
            or any(phrase in lowered for phrase in _CONTINUATION))


def match_variation(reply: str, candidates: Sequence[tuple[int, str]],
                    corpus: Corpus | None = None) -> tuple[int, str] | None:
    """Which legal form did the user name?

    Matches the registry's own label AND the words a user would actually type:
    a live session showed "за АД" resolving correctly while "а колку е за
    акционерско друштво" did not, because the labels are abbreviations and
    people write them out. The synonyms come from the corpus glossary
    (index/aliases.py), so there is no hand-maintained list to drift.

    Short labels are matched on word boundaries -- a substring test fires on
    "адреса" or "склад" and would silently filter the search to the wrong form.
    """
    r = _norm(_collapse_dotted(reply))
    for vid, label in candidates:
        forms = [label]
        if corpus is not None:
            from index.aliases import form_synonyms
            forms += list(form_synonyms(label, corpus))
        for form in forms:
            f = _norm(form)
            if len(f) <= 5:
                if re.search(rf"(?<!\w){re.escape(f)}(?!\w)", r):
                    return vid, label
            elif f in r:
                return vid, label
            else:
                # Comma-listed labels name several forms at once ("ДОО, ДООЕЛ").
                # Each part is matched on a word boundary rather than by
                # substring: the old length guard required 4+ characters, so a
                # user replying "доо" -- three -- matched nothing and the menu
                # looped, while "дооел" resolved fine.
                for part in (p.strip() for p in f.split(",")):
                    if len(part) > 1 and re.search(rf"(?<!\w){re.escape(part)}(?!\w)", r):
                        return vid, label
    return None


class CRMAgent:
    def __init__(self, *, corpus: Corpus | None = None, retriever=None,
                 model: str = DEFAULT_MODEL, k: int = DEFAULT_K,
                 temperature: float = DEFAULT_TEMPERATURE,
                 dedup: bool = True, aliases: bool = True,
                 structured: bool = True, timeout: float = DEFAULT_TIMEOUT,
                 api_key: str | None = None, client=None):
        self.corpus = corpus or load_corpus()
        if retriever is None:
            from index.hybrid import HybridRetriever
            retriever = HybridRetriever(self.corpus)
            if aliases:
                from index.aliases import AliasExpandingRetriever
                retriever = AliasExpandingRetriever(retriever, self.corpus)
            if structured:
                from index.structured import SectionFetchRetriever
                retriever = SectionFetchRetriever(retriever, self.corpus)
        self.retriever = retriever
        self.model, self.k, self.temperature, self.dedup = model, k, temperature, dedup

        if client is None:
            import os

            from openai import OpenAI
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set.\n"
                    "  PowerShell (persistent): setx OPENAI_API_KEY '<your key>'")
            client = OpenAI(api_key=key, max_retries=3, timeout=timeout)
        self.client = client

        self.history: list[dict[str, str]] = []
        # Every document shown in this conversation, so a citation repeated from
        # an earlier turn is graded as stale rather than fabricated.
        self.seen_docs: set[str] = set()
        self._pending: tuple[str, Ambiguity] | None = None   # (question, ambiguity)
        self._topic: str | None = None          # last self-contained question
        self._last_service: int | None = None   # service the last turn was about
        self.total_cost_usd = 0.0

    # -- retrieval --------------------------------------------------------- #
    def _plan_retrieval(self, message: str) -> tuple[str, dict[str, Any] | None]:
        """Turn a conversational message into a self-contained retrieval query.

        Three cases, strongest first:
          1. a reply to a clarification we asked -> merge it with that question
             and, if it names a legal form, filter to that form;
          2. any other follow-up ("а колку е за акционерско друштво") -> merge
             with the running topic, and filter if it names a form of the
             service the last turn was about. Without this the follow-up
             retrieves on five words with no subject: observed live, it came
             back with two documents and lost the tariff row entirely;
          3. a self-contained question -> use as-is; it becomes the topic.
        """
        self._reply_named_absent = False
        if self._pending is not None:
            question, amb = self._pending
            matched = match_variation(message, amb.variations, self.corpus)
            query = f"{question} {message}".strip()
            if not matched:
                # They answered with a form this service does not offer. Showing
                # the same menu again cannot help -- it is the one reply that
                # carries no new information. Note "ТП" reaches here on purpose:
                # the corpus has no domestic sole-trader variation, and its only
                # ТП label is "Подружница на странско друштво и странски ТП", a
                # FOREIGN branch. Quietly resolving to that would route someone
                # opening a local shop into the wrong service entirely.
                self._reply_named_absent = _names_absent_form(
                    message, self.corpus, [lbl for _, lbl in amb.variations])
            if matched:
                # The SERVICE is already settled -- it is the one we asked about.
                # Scoping only the variation let the merged query go shopping:
                # asked which legal form for "вистински сопственик" (2183) and
                # answered "АД", the search took "АД" at face value and returned
                # the incorporation-of-AD service (2140), so the user got correct
                # instructions for a procedure they had not asked about. A reply
                # to our own menu narrows the established topic; it is not a new
                # question.
                return query, {"service_scope": amb.id_service,
                               "variation_scope": matched[0]}
            return query, None

        # NOTE: `_topic` is the PREVIOUS user message, not the last
        # self-contained one. A follow-up can change the subject -- "А во
        # Струмица?" is phrased as a continuation but moves to a new
        # municipality -- and anchoring on the last self-contained question
        # meant a third turn reached back past it. Observed: turn 3 asked for
        # examples of the Струмица agents just listed and retrieved Гостивар.
        # A message that names no subject is a continuation whatever its length.
        # "колку се плаќа и како се плаќа, преку кои" is eight tokens with no
        # discourse opener and no anaphor, so the shape-based test called it a
        # new question -- and it retrieved on a query that says only what the
        # user wants to know, never what about, landing on an unrelated service.
        # In a conversation, "how much is it" means "how much is THIS".
        if self._topic and (is_followup(message)
                            or not _carries_subject(message, self.corpus)):
            query = f"{self._topic} {message}".strip()
            forms = self._forms_of(self._last_service)
            matched = match_variation(message, forms, self.corpus) if forms else None
            return (query, {"variation_scope": matched[0]}) if matched else (query, None)

        # A self-contained question that NAMES its legal form scopes to it. Rows
        # belonging to a different form cannot answer it, and left in the pool
        # they win on semantic similarity -- "кои се потребните документи за да
        # регистрирам доо" filled all ten slots with Фондација documents. Rows
        # with no form of their own (descriptions, FAQs, shared terminology) are
        # kept: they apply to every form, and dropping them would leave nothing
        # to answer from.
        forms = named_forms(message, self.corpus)
        if forms:
            return message, {"form_scope": forms}

        return message, None

    def _forms_of(self, id_service: int | None) -> list[tuple[int, str]]:
        if id_service is None:
            return []
        return [(vid, self.corpus.variation_name.get((id_service, vid), ""))
                for vid in self.corpus.variations_of.get(id_service, ())
                if self.corpus.variation_name.get((id_service, vid))]

    def retrieve(self, query: str, filters: dict[str, Any] | None = None) -> list[Hit]:
        return list(self.retriever.search(query, self.k * OVERFETCH, filters=filters))

    # -- generation -------------------------------------------------------- #
    def _clarify(self, message: str, amb: Ambiguity, docs: list[ContextDoc],
                 retrieval_ms: float) -> Answer:
        """Ask which variant, deterministically. No model call, so nothing leaks."""
        text = clarification_text(amb, self.corpus)
        if getattr(self, "_reply_named_absent", False):
            text = ("Формата што ја наведовте не е меѓу формите достапни за оваа "
                    "услуга. Достапни се следниве:\n\n"
                    + text.split("\n\n", 1)[-1])
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": text})
        self._pending = ((self._pending[0] if self._pending else message), amb)
        self._last_service = amb.id_service
        self.seen_docs.update(d.chunk_id for d in docs)
        return Answer(query=message, text=text, docs=docs, citations=[],
                      stale_citations=[], invalid_citations=[], ambiguity=amb,
                      retrieval_query=message, filters=None, model=self.model,
                      prompt_tokens=0, completion_tokens=0, cost_usd=0.0,
                      retrieval_ms=retrieval_ms, generation_ms=0.0)

    def ask(self, message: str) -> Answer:
        retrieval_query, filters = self._plan_retrieval(message)

        t0 = time.perf_counter()
        hits = self.retrieve(retrieval_query, filters)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        # Rows structured fetch deliberately injected must survive the trim to
        # k distinct documents -- otherwise a section is fetched whole and then
        # cut apart by semantic competition (see build_documents).
        pinned = list(getattr(self.retriever, "last_sections", ()) or ())
        context_xml, docs, amb = build_context(hits, self.corpus, retrieval_query,
                                               dedup=self.dedup, limit=self.k,
                                               pin=pinned)
        # A clarification already resolved to one legal form is not ambiguous
        # any more, even if shared blocks still mention siblings.
        if filters and "variation_scope" in filters:
            amb = None

        # GATE 3, enforced rather than requested. When the choice is real the
        # clarification IS the reply, so we write it and never call the model --
        # there is no answer, no coverage hedge and no menu-underneath-an-answer,
        # because no model output exists to leak. Also saves a call.
        if amb is not None:
            return self._clarify(message, amb, docs, retrieval_ms)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += self.history[-MAX_HISTORY_TURNS * 2:]
        messages.append({"role": "system", "content": context_message(context_xml)})
        messages.append({"role": "user", "content": message})

        t1 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature)
        generation_ms = (time.perf_counter() - t1) * 1000

        text = (resp.choices[0].message.content or "").strip()
        valid, stale, invalid = validate_citations(text, docs, self.seen_docs)

        # HARD ABSTENTION. Every factual sentence must carry a citation, so an
        # answer with no valid one is either an abstention (fine) or ungrounded.
        # Observed live: asked which documents a ДОО needs, the context held ten
        # rows -- all Фондација -- and rather than say so the model fell back on
        # pre-trained knowledge and listed "Статут" and "Програма за дејствување",
        # which belong to other legal forms. Nothing in the answer was citable,
        # and the citation count said 0/10 while the text read as authoritative.
        # A prompt rule did not prevent this and cannot; the check is structural.
        if not valid and not any(m in _norm(text) for m in _ABSTAIN_MARKERS):
            text = HARD_ABSTENTION
            stale, invalid = [], []
        self.seen_docs.update(d.chunk_id for d in docs)

        usage = getattr(resp, "usage", None)
        p_tok = getattr(usage, "prompt_tokens", 0) or 0
        c_tok = getattr(usage, "completion_tokens", 0) or 0
        in_price, out_price = PRICING.get(self.model, (0.0, 0.0))
        cost = (p_tok * in_price + c_tok * out_price) / 1_000_000
        self.total_cost_usd += cost

        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": _for_history(text)})
        # Remember what we asked about, so the next message can be read as a reply.
        self._pending = ((self._pending[0] if self._pending else message), amb) if amb else None
        if self._pending is None and (self._topic is None
                                      or _carries_subject(message, self.corpus)):
            # Only a message that names a SUBJECT may become the topic. Live,
            # "колку се плаќа и како се плаќа" carried none -- it says what the
            # user wants to know, not what about -- yet it replaced the topic,
            # and the next follow-up merged with nothing and landed on an
            # unrelated service. A subject-free turn now leaves the topic alone,
            # so the thread survives a run of contextless questions.
            self._topic = message
        self._last_service = (amb.id_service if amb else
                              _attributed_service(docs) or self._last_service)

        return Answer(
            query=message, text=text, docs=docs, citations=valid,
            stale_citations=stale, invalid_citations=invalid, ambiguity=amb,
            retrieval_query=retrieval_query, filters=filters, model=self.model,
            prompt_tokens=p_tok, completion_tokens=c_tok, cost_usd=cost,
            retrieval_ms=retrieval_ms, generation_ms=generation_ms,
        )

    def reset(self) -> None:
        self.history.clear()
        self.seen_docs.clear()
        self._pending = None
        self._topic = None
        self._last_service = None

    def close(self) -> None:
        close = getattr(self.retriever, "close", None)
        if close:
            close()
