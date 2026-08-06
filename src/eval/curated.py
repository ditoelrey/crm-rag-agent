"""
curated.py  --  hand-authored evaluation cases in real user phrasing.
=====================================================================
The synthetic families measure retrieval mechanics; this file is the only place
that measures whether the system answers the questions people actually ask. Every
entry below was written as a question first, then its gold was located in the
corpus and read to confirm it genuinely answers it (verified 2026-07-28).

Gold is expressed as a *rule* (service + variation + section type, or explicit
chunk ids) rather than a frozen id list, so a corpus rebuild that renumbers rows
re-resolves instead of silently rotting. `expand()` raises if a rule matches
nothing -- an unanswerable case must fail loudly, not score 0 forever.

Adding cases
------------
Append a Spec. Keep the query in the user's words (typos and all, if that is
what they type), point `service`/`variation` at the answer, and put the reason
in `notes`. 15 honest cases beat 500 generated ones; grow this file whenever a
real question is seen to fail.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .corpus import Corpus
from .goldset import EvalCase, _support_grades


@dataclass
class Spec:
    query: str
    intent: str                       # names the family: curated_<intent>
    notes: str
    service: int | None = None
    variation: int | None = None
    types: tuple[str, ...] = ()       # gold = all rows of these types
    chunk_ids: tuple[str, ...] = ()   # ...or these exact blocks
    also_services: tuple[int, ...] = ()   # additional acceptable services (grade 1)
    # Additional primary gold as (id_service, id_variation|None, types) -- for
    # questions a user could legitimately mean about more than one service.
    plus: tuple[tuple[int, int | None, tuple[str, ...]], ...] = ()
    difficulty: str = "hard"
    pin_provenance: bool = False      # see _COPY_TOLERANT_TYPES
    behavior: str | None = None       # "answer" | "clarify" | "abstain"
    # Agent-directory gold, as (municipality, list|None) selectors rather than
    # pinned chunk_ids -- part counts shift whenever the registry republishes a
    # list, and gold that rots looks exactly like a retriever bug.
    directory: tuple[tuple[str, str | None], ...] = ()


# Types whose text is portal boilerplate, copied verbatim under many services.
# When gold of one of these types is pinned, every byte-identical copy is graded
# primary too: the question ("Што значи ПКД?") names no service, so any copy
# answers it. Set `pin_provenance=True` on a Spec where the asking service
# genuinely matters and only its own copy should count.
_COPY_TOLERANT_TYPES = frozenset({"terminology", "faq", "legalBasis"})


# --------------------------------------------------------------------------- #
# The cases.
# --------------------------------------------------------------------------- #
SPECS: list[Spec] = [
    Spec(query="Сакам сам да пријавам основање на здружение, кои документи ми се потребни?",
         intent="documents", service=2135, variation=11116, types=("documents",),
         notes="Здружение is one of 9 sibling variations of 2135; the document "
               "list differs per variation, so a sibling hit is a wrong answer."),

    Spec(query="Колку чини упис на основање на фондација?",
         intent="tariffs", service=2135, variation=11117, types=("tariffs",),
         notes="2452 МКД, variation-specific tariff row."),

    Spec(query="За колку време се одобрува пријава за упис на основање на АД?",
         intent="deadlines", service=2135, variation=11113, types=("deadlines",),
         notes="4 часа for approval; the variation has 4 different deadline rows."),

    Spec(query="Колку чини потврда за тековна состојба на фирма?",
         intent="tariffs", service=2162, variation=11050, types=("tariffs",),
         notes="295 МКД. Competing service 2185 has a near-identical 299 МКД "
               "tariff for a different certificate -- a real confusion pair."),

    Spec(query="Каде ја подигам потврдата за тековна состојба?",
         intent="access", service=2162, variation=11050,
         types=("documentsLocations", "access"),
         notes="Counter pickup at ЦРРСМ offices."),

    Spec(query="Дали годишна сметка може да се поднесе преку интернет и на кој линк?",
         intent="access", service=2111, types=("access",),
         notes="Underspecified on purpose: the user did not say which subject "
               "type, and all five variations share the same e-submit portal, "
               "so any variation's access block is a correct answer."),

    Spec(query="Кој е крајниот рок за поднесување годишна сметка за банка?",
         intent="deadlines", service=2111, variation=11176, types=("deadlines",),
         notes="15.3. legal deadline / 31.12. final; variation = банки."),

    Spec(query="Во кој рок морам да пријавам промена на вистински сопственик?",
         intent="deadlines", service=2183, variation=11122,
         chunk_ids=("srv_2183_v11122_deadlines_2",),
         notes="15 days from the change. Pinpoint gold: the sibling deadline "
               "rows in the same variation are about other obligations."),

    Spec(query="Што значи ПКД?",
         intent="terminology", service=2075, chunk_ids=("srv_2075_shared_terminology_2",),
         notes="Abbreviation lookup. The definition is byte-identical under 16 "
               "sanctions-register services and the question names none of "
               "them, so every copy counts as correct."),

    Spec(query="Колку чини тендерско досие со економско-финансиска состојба?",
         intent="tariffs", service=2059, variation=10947, types=("tariffs",),
         notes="8710 МКД. 2051 is the cheaper package WITHOUT the financial "
               "part (5140 МКД) -- picking it is a wrong, expensive answer."),

    Spec(query="Како се регистрира залог врз машина?",
         intent="process", service=2063, variation=11187, types=("process",),
         notes="6-step notary-led procedure; 2113/2115 (change/deletion of a "
               "pledge) are the near-miss services."),

    Spec(query="Ми треба потврда дека фирмата нема забрана за учество во јавни набавки",
         intent="service_routing", service=2075,
         types=("description", "process"),
         notes="Routing case: the user describes the need, never the service "
               "name, and 8 sibling services differ only in which sanction they "
               "certify."),

    Spec(query="Може ли да извадам потврда без регистрационен агент?",
         intent="faq", service=2185, chunk_ids=("srv_2185_shared_faq_1",),
         also_services=(2186,),
         notes="This FAQ is duplicated verbatim under 14 services; all copies "
               "count. 2186's remaining blocks are graded acceptable context."),

    Spec(query="Колку чини упис на промена кај приватна установа?",
         intent="tariffs", service=2140, variation=11173, types=("tariffs",),
         notes="1603 МКД + 200 МКД per additional change in the same filing; "
               "both rows are needed for a complete answer."),

    Spec(query="Кои документи се потребни за упис на промена кај АД преку регистрационен агент?",
         intent="documents", service=2141, variation=11206, types=("documents",),
         notes="2140 (self-filing) vs 2141 (through an agent) are near-identical "
               "services with the same АД variation label -- service-level "
               "confusion, not variation-level."),

    # ----------------------------------------------------------------- #
    # Underspecified questions.
    #
    # Added after watching the live agent fail on them. Every case above names
    # its service; real users do not, and the gap is where the system broke:
    # "Колку чини регистрација?" retrieved zero tariff rows, because the word
    # "регистрација" matches blocks about registering a USER ACCOUNT in the
    # e-system, while the actual fee rows say "упис на основање" -- the word
    # appears in 2 of 315 tariff rows corpus-wide.
    #
    # Retrieval gold is every legal form's rows across BOTH registration
    # services: with the question as asked, any of them is a legitimate
    # retrieval, and the agent's job is to ask which one rather than to pick.
    # These are the cases that will show whether the vocabulary work lands.
    # ----------------------------------------------------------------- #
    Spec(query="Колку чини регистрација?",
         intent="underspecified", service=2135, types=("tariffs",),
         plus=((2136, None, ("tariffs",)),), behavior="clarify",
         notes="Ambiguous on two axes: which service (2135 self-filing vs 2136 "
               "via a registration agent) and which legal form. The fees really "
               "do diverge -- АД and ПДОО are 0 МКД, the other seven forms are "
               "2452 МКД -- so a guess is wrong by the entire amount."),

    Spec(query="Колку чини основање на фирма?",
         intent="underspecified", service=2135, types=("tariffs",),
         plus=((2136, None, ("tariffs",)),), behavior="clarify",
         notes="Same question in the vocabulary a user actually reaches for "
               "('фирма', 'основање'). Pairs with the 'регистрација' phrasing "
               "to separate the vocabulary gap from the ambiguity handling."),

    Spec(query="Кои документи ми требаат за да регистрирам фирма?",
         intent="underspecified", service=2135, types=("documents",),
         plus=((2136, None, ("documents",)),), behavior="clarify",
         notes="Document lists differ across all 9 forms (9 distinct sets), so "
               "this must be clarified, not answered."),

    Spec(query="Колку време трае регистрација на фирма?",
         intent="underspecified", service=2135, types=("deadlines",),
         plus=((2136, None, ("deadlines",)),), behavior="clarify",
         notes="Deadlines differ across forms (5 distinct sets across 9)."),

    Spec(query="Како да регистрирам залог?",
         intent="underspecified", service=2063, variation=11187,
         types=("process",), behavior="answer",
         notes="Single-variation service: NOT ambiguous, so the agent must "
               "answer rather than ask. Isolates the vocabulary gap from the "
               "disambiguation logic -- the sibling case adds the noun 'машина' "
               "(the corpus says 'подвижни предмети') and fails on all three "
               "retrievers."),
    # ----------------------------------------------------------------- #
    # Authorised-agent directory (scope == "directory").
    #
    # A different shape of question: not "what does this service require" but
    # "who near me can file it". The answer is a list, so completeness is the
    # whole game -- returning ten of Прилеп's 75 agents is a wrong answer that
    # reads like a right one. Gold is every block for that municipality, and
    # structured.fetch_directory serves them deterministically.
    # ----------------------------------------------------------------- #
    Spec(query="Кои се овластените регистрациони агенти во Гостивар?",
         intent="agents", directory=(("ГОСТИВАР", None),), behavior="answer",
         notes="Both lists, one part each (33 advocates + 36 ДОО/ТП). They stay "
               "separate on purpose: the two carry different scopes of "
               "authority and merging them would blur a real legal distinction."),

    Spec(query="Дај ми список на адвокати регистрациони агенти во Прилеп",
         intent="agents", directory=(("ПРИЛЕП", "advocates"),), behavior="answer",
         notes="64 advocates across 2 parts -- no answer can list them all, so "
               "what matters is that both parts are reachable and the total is "
               "stated rather than silently truncated."),

    Spec(query="Кој може да ми регистрира ДОО во Битола?",
         intent="agents", directory=(("БИТОЛА", "doo_tp"),), behavior="answer",
         notes="Deliberately phrased WITHOUT 'агент' or 'адвокат', so the "
               "deterministic lookup does not fire and this rests on semantic "
               "retrieval alone. Tracks a known gap instead of hiding it."),
]


def expand(c: Corpus) -> list[EvalCase]:
    """Resolve every Spec against the corpus into a validated EvalCase."""
    cases: list[EvalCase] = []
    for i, s in enumerate(SPECS, 1):
        primary: list[str] = []
        if s.chunk_ids:
            for cid in s.chunk_ids:
                if cid not in c.by_id:
                    raise ValueError(f"curated case {i} ({s.query[:40]!r}): "
                                     f"gold chunk_id {cid!r} is not in the corpus")
                primary.append(cid)
        targets: list[tuple[int, int | None, tuple[str, ...]]] = []
        if s.types:
            if s.service is None:
                raise ValueError(f"curated case {i}: `types` needs a `service`")
            targets.append((s.service, s.variation, s.types))
        targets += list(s.plus)
        for sid, vid, types in targets:
            for type_ in types:
                found = c.of_type(sid, type_, id_variation=vid)
                if not found:
                    raise ValueError(
                        f"curated case {i} ({s.query[:40]!r}): no {type_!r} blocks "
                        f"for service {sid} variation {vid}")
                primary += [b.chunk_id for b in found]
        for municipality, list_key in s.directory:
            blocks = [b for b in c.by_municipality.get(municipality, ())
                      if list_key is None or b.list_key == list_key]
            if not blocks:
                raise ValueError(
                    f"curated case {i} ({s.query[:40]!r}): no agent-directory "
                    f"blocks for {municipality!r} list={list_key!r}")
            primary += [b.chunk_id for b in blocks]

        if not primary:
            raise ValueError(f"curated case {i}: no gold resolved")

        if not s.pin_provenance:
            for cid in list(primary):
                if c.by_id[cid].type in _COPY_TOLERANT_TYPES:
                    primary += c.identical_to(cid)

        gold = {cid: 2 for cid in primary}
        if s.service is not None:
            gold.update(_support_grades(c, s.service, s.variation, gold))
        for sid in s.also_services:
            for b in c.by_service.get(sid, ()):
                gold.setdefault(b.chunk_id, 1)

        cases.append(EvalCase(
            case_id=f"curated::{i:02d}::{s.intent}",
            query=s.query,
            family=f"curated_{s.intent}",
            source="curated",
            gold=gold,
            expect_service=s.service,
            expect_variation=s.variation,
            expect_types=list(s.types) or _types_of(c, primary),
            difficulty=s.difficulty,
            notes=s.notes,
            expect_behavior=s.behavior,
        ))
    return cases


def _types_of(c: Corpus, chunk_ids: list[str]) -> list[str]:
    return sorted({c.by_id[cid].type for cid in chunk_ids})
