"""
context.py  --  turn retrieved hits into the XML context block the model reads.
===============================================================================
Two things happen here that are not just formatting, both driven by what the
retrieval eval measured on this corpus:

DEDUPLICATION
    16-20% of every top-10 is byte-identical text repeated under different
    services, and ~20% of queries waste four or more of their ten slots on the
    same sentence (measured across all three retriever reports). The portal
    replicates boilerplate: 88% of `process` rows, 93% of `terminology` and 94%
    of `documentsLocations` appear verbatim under more than one service.
    Collapsing duplicates recovers ~2 usable slots out of 10 for free.

    Provenance is preserved rather than discarded: the surviving copy is the one
    from the best-ranked service, and the services it also covers are listed in
    `also_in`. A citation has to point at the right service to be checkable.

AMBIGUITY DETECTION
    Variation disambiguation is too important to leave to the model noticing it.
    It is detected deterministically: if, after dedup, two or more variations of
    the same service contribute hits of the SAME section type, the answers
    genuinely differ (different documents, different fees), and if the user's
    question names none of those variations, the agent must ask instead of
    guessing.

    Testing "same section type, different content, after dedup" is what keeps
    this from firing on questions that only look ambiguous: all five variations
    of "Поднесување годишна сметка" share one identical e-submit access block,
    so dedup collapses them to one and no clarification is requested.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from eval.corpus import Block, Corpus
from eval.retriever import Hit

# Intent detection lives in index/intent.py: the retrieval layer needs it too
# (structured section fetch), and both must agree on what a question asks for.
from index.intent import INTENT_CUES, detect_intent   # noqa: F401  (re-export)


@dataclass
class ContextDoc:
    """One document as the model will see it."""
    chunk_id: str
    block: Block
    score: float
    rank: int
    also_in: list[str] = field(default_factory=list)   # other services with identical text


@dataclass
class Ambiguity:
    id_service: int
    service_name: str
    variations: list[tuple[int, str]]      # (id_variation, label)
    section_types: list[str]
    source: str = "evidence"               # "evidence" | "structure"

    @property
    def labels(self) -> list[str]:
        return [label for _, label in self.variations]


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def build_documents(hits: Sequence[Hit], corpus: Corpus, *, dedup: bool = True,
                    limit: int | None = None) -> list[ContextDoc]:
    """Rank-ordered documents, optionally collapsing byte-identical content.

    `limit` counts DISTINCT documents, which is why the caller over-fetches:
    live turns were arriving with 2 documents out of k=10 because eight were
    byte-identical copies. Asking for more hits and trimming to `limit` distinct
    ones costs nothing extra (the hybrid arms already search to depth 50, and
    the query is embedded once either way).
    """
    docs: list[ContextDoc] = []
    by_content: dict[str, ContextDoc] = {}

    for rank, hit in enumerate(hits, 1):
        block = corpus.by_id.get(hit.chunk_id)
        if block is None:                       # index newer than the corpus
            continue
        if dedup and block.content in by_content:
            kept = by_content[block.content]
            if block.service_name and block.service_name not in kept.also_in \
                    and block.service_name != kept.block.service_name:
                kept.also_in.append(block.service_name)
            continue
        doc = ContextDoc(chunk_id=block.chunk_id, block=block,
                         score=hit.score, rank=rank)
        by_content[block.content] = doc
        docs.append(doc)
        if limit is not None and len(docs) >= limit:
            break
    return docs


def _names_a_form(query: str, labels: Iterable[str]) -> bool:
    q = _norm(query)
    return any(_norm(label) in q for label in labels)


def _attributed_service(docs: Sequence[ContextDoc]) -> int | None:
    """Which service is the question actually about?

    Ranked by best position, but counting only blocks with `id_variation > 0`:
    those carry a service's own variation-specific content, while shared
    terminology and FAQ text is boilerplate the portal replicates across
    services and is near-worthless as evidence of *which* service is meant.

    On the observed failure ("Колку чини регистрација?") the top hit was a
    terminology block from service 2155, which has no variations at all, while
    the block that actually identified the subject -- a 2135 documents row --
    sat at rank 4. Ranking on shared blocks would have asked about the wrong
    service; this attribution picks 2135.
    """
    best: tuple[int, int] | None = None      # (rank, id_service)
    for d in docs:
        if d.block.id_variation and not d.block.is_shared:
            if best is None or d.rank < best[0]:
                best = (d.rank, d.block.id_service)
    return best[1] if best else None


def _differing_sections(corpus: Corpus, id_service: int,
                        types: Sequence[str]) -> list[str]:
    """Section types whose content genuinely differs between the service's
    variations. Asking the user to choose is only justified when the choice
    changes the answer -- all five variations of 'Поднесување годишна сметка'
    share one identical access block, and that must not trigger a question."""
    out = []
    for type_ in types:
        seen = {tuple(sorted(b.content for b in corpus.of_type(id_service, type_, vid)))
                for vid in corpus.variations_of.get(id_service, ())}
        seen.discard(())
        if len(seen) >= 2:
            out.append(type_)
    return out


def detect_ambiguity(docs: Sequence[ContextDoc], query: str,
                     corpus: Corpus | None = None) -> Ambiguity | None:
    """Does answering this require picking a legal form the user hasn't named?

    Two independent detectors, strongest first:

      evidence   two or more variations of one service are already in the
                 retrieved documents, contributing the same section type.
      structure  the corpus schema says the attributed service has sibling
                 variations whose relevant section differs -- even if retrieval
                 surfaced none of them. This is the one that matters: on a bare
                 question like "Колку чини регистрација?" retrieval returned no
                 tariff rows at all, so an evidence-only detector stays silent
                 exactly when the user most needs to be asked.
    """
    amb = _detect_from_evidence(docs, query, corpus)
    if amb is not None or corpus is None:
        return amb
    return _detect_from_structure(docs, query, corpus)


def _all_forms(corpus: Corpus | None, sid: int,
               fallback: dict[int, str]) -> dict[int, str]:
    """Every legal form the service has, not just the ones retrieval happened to
    return. Offering the user two options when the service has nine makes the
    other seven unreachable."""
    if corpus is None:
        return fallback
    labels = {vid: corpus.variation_name.get((sid, vid), "")
              for vid in corpus.variations_of.get(sid, ())}
    labels = {vid: name for vid, name in labels.items() if name}
    return labels or fallback


def _detect_from_evidence(docs: Sequence[ContextDoc], query: str,
                          corpus: Corpus | None = None) -> Ambiguity | None:
    # A question with no readable section intent is not a question about a
    # legal form. Observed live: "знаеш ли кој е маилот на регистарот?" pulled
    # blocks from several variations of one service and asked the user to pick
    # between АД, Здружение and Фондација -- to answer an email question. The
    # structure detector has always required an intent; this one did not.
    intents = set(detect_intent(query))
    if not intents:
        return None

    by_service: dict[int, list[ContextDoc]] = {}
    for d in docs:
        by_service.setdefault(d.block.id_service, []).append(d)

    # The service the retriever is most confident about: best (lowest) rank.
    for sid, group in sorted(by_service.items(),
                             key=lambda kv: min(d.rank for d in kv[1])):
        per_type: dict[str, dict[int, str]] = {}
        for d in group:
            b = d.block
            if b.is_shared or not b.id_variation or not b.variation_short_name:
                continue
            per_type.setdefault(b.type, {})[b.id_variation] = b.variation_short_name

        conflicting = {t: v for t, v in per_type.items() if len(v) >= 2}
        if not conflicting:
            continue

        # Two forms landing in the top-k is not proof their answers differ --
        # the same section can be replicated verbatim across variations. Verify
        # against the corpus, exactly as the structure detector does, so the two
        # paths cannot disagree about whether a choice is real.
        types = sorted(set(conflicting) & intents) or sorted(conflicting)
        if corpus is not None:
            types = _differing_sections(corpus, sid, types)
            if not types:
                continue

        variations: dict[int, str] = {}
        for v in conflicting.values():
            variations.update(v)
        variations = _all_forms(corpus, sid, variations)
        # The user already named one of them -> not ambiguous, just answer.
        if _names_a_form(query, variations.values()):
            return None
        return Ambiguity(id_service=sid,
                         service_name=group[0].block.service_name,
                         variations=sorted(variations.items(), key=lambda kv: kv[1]),
                         section_types=types, source="evidence")
    return None


def _detect_from_structure(docs: Sequence[ContextDoc], query: str,
                           corpus: Corpus) -> Ambiguity | None:
    sid = _attributed_service(docs)
    if sid is None:
        return None
    variation_ids = corpus.variations_of.get(sid, [])
    if len(variation_ids) < 2:
        return None

    intents = detect_intent(query)
    if not intents:
        return None                      # unclear intent -> do not guess
    differing = _differing_sections(corpus, sid, intents)
    if not differing:
        return None                      # the choice would not change the answer

    labels = {vid: corpus.variation_name.get((sid, vid), "") for vid in variation_ids}
    labels = {vid: name for vid, name in labels.items() if name}
    if len(labels) < 2 or _names_a_form(query, labels.values()):
        return None

    return Ambiguity(id_service=sid,
                     service_name=corpus.service_name.get(sid, ""),
                     variations=sorted(labels.items(), key=lambda kv: kv[1]),
                     section_types=sorted(differing), source="structure")


def _esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def render_documents(docs: Iterable[ContextDoc]) -> str:
    """The `<documents>` block. Attributes carry the metadata the model needs to
    disambiguate (service, legal form, section) without re-reading the text."""
    out = ["<documents>"]
    for d in docs:
        b = d.block
        attrs = [f'id="{_esc(b.chunk_id)}"',
                 f'service="{_esc(b.service_name)}"',
                 f'section="{_esc(b.type)}"']
        if b.variation_short_name:
            # NOT "legal_form": 31% of variation labels are delivery channels
            # ("Хартиено на шалтер", "Електронски", "Web сервис") or scopes
            # ("Упис на залог"), not legal forms. Calling them all legal forms
            # taught the model to ask "За каква правна форма?" about a paper-vs-
            # electronic choice, and about services that have only one variant.
            attrs.append(f'variant="{_esc(b.variation_short_name)}"')
        if b.is_shared:
            attrs.append('applies_to="all variants of this service"')
        if d.also_in:
            attrs.append(f'also_applies_to="{_esc("; ".join(d.also_in[:4]))}"')
        out.append(f"  <document {' '.join(attrs)}>")
        out.append("    " + _esc(b.content).replace("\n", "\n    "))
        out.append("  </document>")
    out.append("</documents>")
    return "\n".join(out)


def form_label(label: str, corpus: Corpus | None) -> str:
    """The registry's label plus the words a user would recognise.

    The corpus calls one variation "ПДОО"; its own tariff row describes it as
    "Упис на основање на ДОО / ДООЕЛ". Offering a bare "ПДОО" hides that option
    from everyone who came to register a ДОО -- which is most of them.
    """
    if corpus is None:
        return label
    from index.aliases import form_synonyms
    syn = form_synonyms(label, corpus)
    return f"{label} ({' / '.join(syn[:3])})" if syn else label


# List-shaped sections, where "here are the documents" implies a complete set.
# faq/terminology are lookups -- holding one of 81 FAQ rows is normal and says
# nothing about completeness, so they are not reported.
_LIST_SECTIONS = ("process", "documents", "tariffs", "deadlines", "forms",
                  "documentsLocations", "discounts", "instructions")


def section_coverage(docs: Sequence[ContextDoc],
                     corpus: Corpus) -> list[tuple[str, str | None, int, int]]:
    """(section, variant, present, total) for every list-shaped section in the
    context, so the model can tell a complete list from a fragment.

    Observed live: asked how to register a pledge, the agent answered the
    procedure correctly from all six fetched `process` rows, then volunteered
    "Потребни документи:" and listed the ONE `documents` row that happened to be
    retrieved -- out of six. Nothing in the context distinguishes "this is the
    whole section" from "this is a fragment", so the model cannot know, and a
    partial list under a complete-sounding heading is the same failure that made
    the four-step procedure look finished.
    """
    seen: dict[tuple[int, int | None, str], int] = {}
    label: dict[tuple[int, int | None, str], str | None] = {}
    for d in docs:
        b = d.block
        if b.type not in _LIST_SECTIONS:
            continue
        key = (b.id_service, None if b.is_shared else b.id_variation, b.type)
        seen[key] = seen.get(key, 0) + 1
        label[key] = b.variation_short_name
    out = []
    for (sid, vid, type_), present in seen.items():
        rows = corpus.of_type(sid, type_, vid)
        if vid is None:
            rows = [b for b in rows if b.is_shared]
        total = len(rows)
        if total > 1:
            out.append((type_, label[(sid, vid, type_)], present, total))
    return sorted(out)


def render_coverage(docs: Sequence[ContextDoc], corpus: Corpus | None) -> str:
    if corpus is None:
        return ""
    rows = section_coverage(docs, corpus)
    if not rows:
        return ""
    lines = []
    for type_, variant, present, total in rows:
        where = f" ({_esc(variant)})" if variant else ""
        state = ("COMPLETE" if present >= total
                 else f"PARTIAL -- {total - present} row(s) of this section are not here")
        lines.append(f"  {type_}{where}: {present} of {total} rows -- {state}")
    return ("\n<coverage>\n" + "\n".join(lines) + "\n"
            "A PARTIAL section must never be presented as a full list. Either "
            "omit it, or say explicitly that it is not the complete set.\n"
            "</coverage>")


def render_ambiguity(amb: Ambiguity | None, corpus: Corpus | None = None) -> str:
    if amb is None:
        return ""
    forms = "\n".join(f"  - {form_label(label, corpus)} (id_variation={vid})"
                      for vid, label in amb.variations)
    if amb.source == "structure":
        # The differing rows may not be in <documents> at all -- that is exactly
        # the case this detector exists for, so the model is told not to answer
        # from what it happens to have.
        lead = (
            f"The service \"{_esc(amb.service_name)}\" exists in "
            f"{len(amb.variations)} variants whose {', '.join(amb.section_types)} "
            f"are DIFFERENT. The documents above may contain none of them, or only "
            f"one form's — either way they are not a safe basis for an answer. "
            f"The available variants are:")
    else:
        lead = (
            f"The retrieved documents describe {len(amb.variations)} different "
            f"variants of the same service (\"{_esc(amb.service_name)}\"), and their "
            f"{', '.join(amb.section_types)} differ:")
    return (
        "\n<ambiguity>\n" + lead + f"\n{forms}\n"
        "The user's question does not say which one applies. Do NOT answer with "
        "one of them, do NOT average them, and do NOT present a range. Ask the "
        "user which legal form they mean, listing the options above in Macedonian.\n"
        "</ambiguity>"
    )


def build_context(hits: Sequence[Hit], corpus: Corpus, query: str, *,
                  dedup: bool = True, limit: int | None = None
                  ) -> tuple[str, list[ContextDoc], Ambiguity | None]:
    docs = build_documents(hits, corpus, dedup=dedup, limit=limit)
    amb = detect_ambiguity(docs, query, corpus)
    xml = (render_documents(docs) + render_coverage(docs, corpus)
           + render_ambiguity(amb, corpus))
    return xml, docs, amb


# Markdown links must be stripped BEFORE looking for citations: a model that
# writes [https://...](https://...) is formatting a link, not citing a source,
# and reading its label as a chunk_id reports a fabricated citation on a
# perfectly good answer. Crying wolf here is worse than missing a citation --
# the warning exists to be trusted.
_MD_LINK_RE = re.compile(r"\[[^\]]*\]\([^)]*\)")
_BRACKET_RE = re.compile(r"\[([^\[\]\s]+)\]")


def _is_citation_shaped(token: str) -> bool:
    """chunk_ids look like srv_2162_v11050_tariffs_1: no spaces, underscores,
    not a URL. Anything else in brackets is prose, not a claimed source."""
    if token.lower().startswith(("http://", "https://", "www.")):
        return False
    return "_" in token


def extract_citations(answer: str) -> list[str]:
    return [t for t in _BRACKET_RE.findall(_MD_LINK_RE.sub(" ", answer))
            if _is_citation_shaped(t)]


def validate_citations(answer: str, docs: Sequence[ContextDoc],
                       seen: Iterable[str] = ()) -> tuple[list[str], list[str], list[str]]:
    """Split citations into (valid, stale, fabricated).

    `seen` is every document shown earlier in the same conversation. The
    distinction is not pedantry: in a live session the model answered a repeated
    question by re-citing a tariff row from two turns earlier, which the current
    retrieval had not returned. That claim is weaker than one grounded in the
    current context -- but the user was shown that document, and calling it
    fabricated is the kind of false alarm that teaches people to ignore the
    warning. Only an id that was never retrieved at all is fabricated.
    """
    allowed = {d.chunk_id for d in docs}
    earlier = set(seen) - allowed
    valid, stale, bogus = [], [], []
    for c in extract_citations(answer):
        (valid if c in allowed else stale if c in earlier else bogus).append(c)
    return valid, stale, bogus
