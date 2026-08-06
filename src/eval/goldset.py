"""
goldset.py  --  evaluation cases: synthetic (generated) + curated (hand-written).
=================================================================================
Two tiers, on purpose:

  SYNTHETIC  Derived deterministically from the corpus itself. Free, hundreds of
             cases, covers every service. It measures retrieval *mechanics* and
             is the regression signal -- NOT a measure of real-world quality:
             several families are lexically leaky (an FAQ question is quoted
             verbatim inside its own answer block), so absolute scores read high.
             Leaky families are flagged `difficulty="easy"` and can be excluded
             with --exclude-leaky.

  CURATED    Real user phrasing, hand-labelled gold (cases/curated.jsonl). Small
             and slow to grow, but it is the only honest quality signal. Every
             gold chunk_id is checked against the corpus at load time: a typo'd
             or stale id aborts the run rather than quietly scoring 0.

Both tiers produce the same `EvalCase`, and the generated set is frozen to disk
(`cli.py gen`) so a run is reproducible even after the corpus is rebuilt.

Relevance grading
-----------------
  2 primary    the block(s) that answer the query
  1 acceptable same service/variation supporting context (shared terminology,
               legal basis, the access block) -- rewarded by nDCG, ignored by
               recall/MRR so it can never mask a miss.

The variation families are the ones that matter most for this corpus: a base
service like "Самостојна пријава за упис на основање" has 11 sibling variations
(АД, ДОО, Здружение, Фондација...) whose documents and tariffs genuinely differ.
Returning the right service but the wrong variation is a *wrong legal answer*,
so those cases also drive the `sibling_confusion` diagnostic in harness.py.
"""
from __future__ import annotations

import json
import os
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

from .corpus import Block, Corpus

CASES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases")

# Query templates per section type. `{svc}` is the service label (service name,
# plus " (<variation>)" for variation-targeted cases).
_INTENT_TEMPLATES: dict[str, tuple[str, ...]] = {
    "tariffs": ("Колку чини {svc}?", "Која е цената за {svc}?",
                "Каков надоместок се плаќа за {svc}?"),
    "documents": ("Кои документи се потребни за {svc}?",
                  "Што треба да поднесам за {svc}?"),
    "deadlines": ("Кој е рокот за {svc}?", "Колку време трае {svc}?"),
    "process": ("Каква е постапката за {svc}?", "Како се поднесува {svc}?"),
    "forms": ("Кој образец е потребен за {svc}?",),
    "access": ("Дали {svc} може да се заврши онлајн?",
               "Како можам да платам за {svc}?"),
    "documentsLocations": ("Каде и како го подигнувам документот за {svc}?",),
}

# Service names in this registry run to 300+ characters ("Потврда дали е
# изречена прекршочна санкција забрана за вршење професија..."). Interpolating
# one into a template yields a query no human would type, so template families
# skip them; the affected services are still covered by the faq/terminology
# families, which use their own text.
_MAX_NAME_FOR_TEMPLATE = 90

_FAQ_RE = re.compile(r"^Прашање:\s*(.+?)\nОдговор:", re.S)
_TERM_RE = re.compile(r"^Термин:\s*(.+?)\s*\|\s*Објаснување:", re.S)

LEAKY_FAMILIES = frozenset({"faq", "terminology"})


@dataclass
class EvalCase:
    """One query plus its labelled gold set."""
    case_id: str
    query: str
    family: str
    source: str                          # "synthetic" | "curated"
    gold: dict[str, int]                 # chunk_id -> grade
    expect_service: int | None = None
    expect_variation: int | None = None   # set only for variation-targeted cases
    expect_types: list[str] = field(default_factory=list)
    difficulty: str = "medium"           # easy = lexically leaky, hard = curated
    notes: str = ""
    # What the *agent* should do, as opposed to what retrieval should find.
    # "answer" | "clarify" | "abstain". Retrieval metrics ignore it; the answer
    # eval will not. A `clarify` case still carries retrieval gold: the agent
    # can only ask an informed question if the competing rows were retrievable.
    expect_behavior: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "EvalCase":
        known = {f for f in EvalCase.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"case {d.get('case_id')!r}: unknown field(s) {sorted(unknown)}")
        return EvalCase(**d)


# --------------------------------------------------------------------------- #
# Grading helpers
# --------------------------------------------------------------------------- #
def _support_grades(c: Corpus, id_service: int, id_variation: int | None,
                    primary: Iterable[str]) -> dict[str, int]:
    """Grade-1 context: the target service's shared blocks and, when a variation
    is targeted, that variation's other blocks. Sibling variations get nothing --
    they are the confusion we are trying to measure."""
    prim = set(primary)
    out: dict[str, int] = {}
    for b in c.by_service.get(id_service, ()):
        if b.chunk_id in prim:
            continue
        if b.is_shared:
            out[b.chunk_id] = 1
        elif id_variation is not None and b.id_variation == id_variation:
            out[b.chunk_id] = 1
        elif id_variation is None:
            out[b.chunk_id] = 1
    return out


def _round_robin(by_key: dict[Any, list[Any]], cap: int, rng: random.Random) -> list[Any]:
    """Stratified sample: take one item per key per pass, so a service with 200
    document rows cannot swamp a family with 60 slots."""
    keys = sorted(by_key)
    rng.shuffle(keys)
    pools = {k: list(by_key[k]) for k in keys}
    for k in keys:
        rng.shuffle(pools[k])
    picked: list[Any] = []
    while len(picked) < cap:
        progressed = False
        for k in keys:
            if pools[k]:
                picked.append(pools[k].pop())
                progressed = True
                if len(picked) >= cap:
                    break
        if not progressed:
            break
    return picked


# --------------------------------------------------------------------------- #
# Synthetic families
# --------------------------------------------------------------------------- #
def _gen_faq(c: Corpus, cap: int, rng: random.Random) -> list[EvalCase]:
    """query = the FAQ question; gold = its answer block.

    The portal copies boilerplate FAQs across services byte-for-byte (the parser
    keeps them per-service on purpose, for provenance). Any copy is a correct
    retrieval for a bare question, so every copy is graded 2 -- otherwise the
    harness would penalise a retriever for picking a legitimately identical block.
    """
    by_question: dict[str, list[Block]] = defaultdict(list)
    for b in c:
        if b.type != "faq":
            continue
        m = _FAQ_RE.match(b.content)
        if m:
            by_question[" ".join(m.group(1).split())].append(b)

    by_service: dict[int, list[tuple[str, list[Block]]]] = defaultdict(list)
    for q, blocks in by_question.items():
        if len(q) >= 12:
            by_service[blocks[0].id_service].append((q, blocks))

    cases = []
    for q, blocks in _round_robin(by_service, cap, rng):
        target = blocks[0]
        gold = {b.chunk_id: 2 for b in blocks}
        gold.update(_support_grades(c, target.id_service, None, gold))
        cases.append(EvalCase(
            case_id=f"faq::{target.chunk_id}",
            query=q, family="faq", source="synthetic", gold=gold,
            expect_service=target.id_service, expect_types=["faq"],
            difficulty="easy",
            notes=(f"{len(blocks)} identical copies across services; any counts"
                   if len(blocks) > 1 else "verbatim question, lexically leaky"),
        ))
    return cases


def _gen_terminology(c: Corpus, cap: int, rng: random.Random) -> list[EvalCase]:
    """query = "Што значи <термин>?"; gold = every block defining that term."""
    by_term: dict[str, list[Block]] = defaultdict(list)
    for b in c:
        if b.type != "terminology":
            continue
        m = _TERM_RE.match(b.content)
        if m:
            by_term[" ".join(m.group(1).split())].append(b)

    by_service: dict[int, list[tuple[str, list[Block]]]] = defaultdict(list)
    for term, blocks in by_term.items():
        if 3 <= len(term) <= 60:
            by_service[blocks[0].id_service].append((term, blocks))

    cases = []
    for term, blocks in _round_robin(by_service, cap, rng):
        target = blocks[0]
        gold = {b.chunk_id: 2 for b in blocks}
        cases.append(EvalCase(
            case_id=f"term::{target.chunk_id}",
            query=f"Што значи {term}?", family="terminology", source="synthetic",
            gold=gold, expect_service=None, expect_types=["terminology"],
            difficulty="easy",
            notes=f"term defined in {len(blocks)} block(s)",
        ))
    return cases


def _gen_service_lookup(c: Corpus, cap: int, rng: random.Random) -> list[EvalCase]:
    """query = the service name; gold = that service's description block.

    Measures service *routing* -- with 112 services whose names share long
    prefixes ("Потврда дали е изречена..."), this is the family most likely to
    expose a retriever that matches on boilerplate instead of the discriminator.
    """
    by_service: dict[int, list[Block]] = defaultdict(list)
    for b in c:
        if b.type == "description":
            by_service[b.id_service].append(b)

    cases = []
    for blocks in _round_robin({k: [v] for k, v in by_service.items()}, cap, rng):
        sid = blocks[0].id_service
        gold = {b.chunk_id: 2 for b in blocks}
        gold.update(_support_grades(c, sid, None, gold))
        cases.append(EvalCase(
            case_id=f"svc::{sid}",
            query=c.service_name[sid], family="service_lookup", source="synthetic",
            gold=gold, expect_service=sid, expect_types=["description"],
            difficulty="medium", notes="query is the service name verbatim",
        ))
    return cases


def _gen_intent(c: Corpus, cap: int, rng: random.Random) -> list[EvalCase]:
    """Natural-language intent -> section type, for single-variation services.

    The query never quotes the block text, only the service name, so this family
    actually tests whether "колку чини" reaches a `tariffs` row.
    """
    cands: dict[int, list[tuple[str, str, list[Block]]]] = defaultdict(list)
    for sid, name in c.service_name.items():
        if len(name) > _MAX_NAME_FOR_TEMPLATE:
            continue
        if len(c.variations_of.get(sid, [])) > 1:
            continue     # multi-variation services are covered by _gen_variation
        for type_, templates in _INTENT_TEMPLATES.items():
            blocks = c.of_type(sid, type_)
            if blocks:
                cands[sid].append((type_, rng.choice(templates), blocks))

    cases = []
    for type_, template, blocks in _round_robin(cands, cap, rng):
        sid = blocks[0].id_service
        gold = {b.chunk_id: 2 for b in blocks}
        gold.update(_support_grades(c, sid, None, gold))
        cases.append(EvalCase(
            case_id=f"intent::{sid}::{type_}",
            query=template.format(svc=c.service_name[sid]),
            family=f"intent_{type_}", source="synthetic", gold=gold,
            expect_service=sid, expect_types=[type_], difficulty="medium",
            notes=f"{len(blocks)} gold rows of type {type_}",
        ))
    return cases


def _gen_variation(c: Corpus, cap: int, rng: random.Random) -> list[EvalCase]:
    """The disambiguation family: same service, different variation, different
    answer. Gold is one variation's rows; its siblings are explicitly NOT gold,
    which is what makes `sibling_confusion` measurable."""
    cands: dict[int, list[tuple[int, str, str, list[Block]]]] = defaultdict(list)
    for sid, vids in c.variations_of.items():
        if len(vids) < 2:
            continue
        name = c.service_name[sid]
        short_name = name if len(name) <= _MAX_NAME_FOR_TEMPLATE else name[:_MAX_NAME_FOR_TEMPLATE].rsplit(" ", 1)[0]
        for vid in vids:
            label = c.variation_name.get((sid, vid))
            if not label:
                continue
            for type_, templates in _INTENT_TEMPLATES.items():
                blocks = c.of_type(sid, type_, id_variation=vid)
                if blocks:
                    q = rng.choice(templates).format(svc=f"{short_name} ({label})")
                    cands[sid].append((vid, type_, q, blocks))

    cases = []
    for vid, type_, query, blocks in _round_robin(cands, cap, rng):
        sid = blocks[0].id_service
        gold = {b.chunk_id: 2 for b in blocks}
        gold.update(_support_grades(c, sid, vid, gold))
        cases.append(EvalCase(
            case_id=f"var::{sid}::{vid}::{type_}",
            query=query, family=f"variation_{type_}", source="synthetic",
            gold=gold, expect_service=sid, expect_variation=vid,
            expect_types=[type_], difficulty="hard",
            notes=(f"{len(c.variations_of[sid])} sibling variations; "
                   f"target={c.variation_name.get((sid, vid))!r}"),
        ))
    return cases


_GENERATORS = {
    "faq": _gen_faq,
    "terminology": _gen_terminology,
    "service_lookup": _gen_service_lookup,
    "intent": _gen_intent,
    "variation": _gen_variation,
}

DEFAULT_CAPS = {"faq": 80, "terminology": 60, "service_lookup": 112,
                "intent": 120, "variation": 140}


def generate(c: Corpus, *, seed: int = 20260728,
             caps: dict[str, int] | None = None) -> list[EvalCase]:
    """Build the synthetic gold set. Deterministic for a given (corpus, seed)."""
    caps = {**DEFAULT_CAPS, **(caps or {})}
    cases: list[EvalCase] = []
    for name, gen in _GENERATORS.items():
        cases.extend(gen(c, caps.get(name, 50), random.Random(f"{seed}:{name}")))
    cases.sort(key=lambda x: x.case_id)
    return cases


# --------------------------------------------------------------------------- #
# Persistence + validation
# --------------------------------------------------------------------------- #
def write_cases(cases: Sequence[EvalCase], path: str) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(case.to_json() + "\n")
    return len(cases)


def read_cases(path: str) -> list[EvalCase]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                out.append(EvalCase.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                raise ValueError(f"{path}:{lineno}: bad case ({e})") from e
    return out


def validate_cases(cases: Sequence[EvalCase], c: Corpus) -> None:
    """Fail fast on gold that cannot be satisfied.

    A stale chunk_id scores 0 forever and looks exactly like a retriever bug --
    the single most expensive failure mode an eval harness has. Same fail-fast
    posture as validate_corpus.py.
    """
    problems: list[str] = []
    seen: set[str] = set()
    for case in cases:
        if case.case_id in seen:
            problems.append(f"{case.case_id}: duplicate case_id")
        seen.add(case.case_id)
        if not case.query.strip():
            problems.append(f"{case.case_id}: empty query")
        if not any(g >= 2 for g in case.gold.values()):
            problems.append(f"{case.case_id}: no grade-2 (primary) gold item")
        for cid, grade in case.gold.items():
            if cid not in c.by_id:
                problems.append(f"{case.case_id}: gold chunk_id {cid!r} not in corpus")
            elif grade not in (1, 2):
                problems.append(f"{case.case_id}: gold {cid!r} has grade {grade} (want 1 or 2)")
        if case.expect_service is not None and case.expect_service not in c.by_service:
            problems.append(f"{case.case_id}: expect_service {case.expect_service} not in corpus")
    if problems:
        raise ValueError("gold set is invalid:\n  " + "\n  ".join(problems[:40]) +
                         (f"\n  ... and {len(problems) - 40} more" if len(problems) > 40 else ""))
