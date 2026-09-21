"""
announcement_search.py  --  finding a деловоден број, so Form 3 can be used.
============================================================================
Form 3 fetches one decision by its 14-digit деловоден број. Users do not have
that number; they have a company. This tool closes the gap:

    "решенијата за ЕМБС 7405855"
        -> search_announcements(embs="7405855")   -> 30120260014967
        -> get_registration_decision("30120260014967")

Two tool rounds, which is what MAX_TOOL_ROUNDS already allows.

WHAT THIS DOES NOT DO: INVENT A NUMBER
--------------------------------------
The whole point of a resolver is that the agent stops needing to guess. A
search that returns a plausible 14-digit number it did not actually find is
worse than no search at all -- Form 3 would fetch it, the fetch would either
miss or land on some other company's filing, and the answer would look right
either way. So this tool only ever returns filings it can name a source for,
and "found nothing" is a first-class result.

OFFLINE COVERAGE, AND SAYING SO
-------------------------------
The live endpoint is known (see SEARCH_URL and build_payload, both verified
against a real capture) but is not called: the index is built from the
decisions actually saved in assets/, so the tool can only find filings the
agent can also open, which is the right coverage for an offline agent.

That makes the empty result dangerous in a specific way: "no filings for this
company" is true of OUR ARCHIVE and says nothing about the registry, which
certainly holds more. Every empty result therefore states its own scope, and
the wording is checked in the gates. An agent that turns "I have none saved"
into "the company has none" is worse than one that says nothing.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

from datetime import date as date_type

from pydantic import BaseModel, Field, field_validator, model_validator

from eval.corpus import Block

from ..context import ContextDoc
from .matching import as_date, digits, norm_text, same_embs
from .registration_decision import (ASSETS, PATTERN, DELOVODEN_DIGITS,
                                    LocalDecisionSource, RegistrationDecision,
                                    verify_decision)
from .schema import strict_schema

# --- THE LIVE CALL, recorded and verified but not made --------------------- #
# POST, JSON body, no reCAPTCHA header in the capture (unlike the detail render
# this search feeds). Kept here so the transport is documented next to the code
# that would use it; nothing in this module calls it.
SEARCH_URL = "https://www.crm.com.mk/CRMPublicPortalApi/api/freeservice/announcementoss"

# From the Angular form's <select>. INCOMPLETE -- the DOM listing ends in "etc.",
# so a code absent here is unknown rather than invalid. The model is given
# Macedonian words, not codes; this is the translation the live payload needs.
DOC_TYPE_CODES = {
    100: "основање", 130: "промена", 160: "бришење",
    200: "стечај", 201: "ликвидација",
}
# --------------------------------------------------------------------------- #


class SearchArgs(BaseModel):
    """At least one criterion. A search with none would return the archive."""
    embs: str = Field(description=(
        "ЕМБС of the entity, 7-8 digits, exactly as the user gave it. "
        "Empty string if the user did not give one."))
    # No example name or date here, for the reason EntitySizeArgs gives: an
    # example a model can copy is an example it can search for, and a search
    # nobody asked for returns some other company's filings.
    name: str = Field(description=(
        "Part of the entity's name, as the user wrote it. Empty string if the "
        "user did not name an entity."))
    doc_type: str = Field(description=(
        "Вид на упис as Macedonian text -- основање, промена, бришење, стечај "
        "or ликвидација. Empty string if the user did not restrict the kind of "
        "decision."))
    date: str = Field(description=(
        "Датум на упис in the form dd.mm.yyyy, as the user gave it. Empty "
        "string if the user did not give a date."))

    @field_validator("embs")
    @classmethod
    def _embs_shape(cls, v: str) -> str:
        v = v.strip()
        if v and not re.fullmatch(r"\d{7,8}", v):
            raise ValueError("ЕМБС мора да биде 7 или 8 цифри")
        return v

    @field_validator("date")
    @classmethod
    def _date_shape(cls, v: str) -> str:
        v = v.strip()
        if v and not re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{4}", v):
            raise ValueError("Датумот мора да биде во облик дд.мм.гггг")
        return v

    @model_validator(mode="after")
    def _at_least_one(self) -> SearchArgs:
        if not any((self.embs, self.name, self.doc_type, self.date)):
            raise ValueError(
                "Наведи барем еден критериум: ЕМБС, назив, вид на упис или датум")
        return self

    def describe(self) -> str:
        """The criteria, for the citable text and for the model to echo back."""
        parts = [f"ЕМБС {self.embs}" if self.embs else "",
                 f"назив содржи „{self.name}“" if self.name else "",
                 f"вид на упис „{self.doc_type}“" if self.doc_type else "",
                 f"датум {self.date}" if self.date else ""]
        return ", ".join(p for p in parts if p)


def _utc_offset_hours(day: date_type) -> int:
    """Skopje's offset from UTC on `day`: EU DST, +2 in summer, +1 in winter.

    Computed rather than looked up because this machine has no tz database, and
    a hardcoded +2 would be wrong for five months of the year. EU rule: from the
    last Sunday in March to the last Sunday in October.
    """
    def last_sunday(year: int, month: int) -> date_type:
        last = date_type(year, month, 31) if month == 3 else date_type(year, month, 31)
        return last - timedelta(days=(last.weekday() + 1) % 7)

    start, end = last_sunday(day.year, 3), last_sunday(day.year, 10)
    return 2 if start <= day < end else 1


def date_ticks(ddmmyyyy: str) -> int:
    """.NET ticks for a date, the way the portal's own client sends them.

    NOT the naive conversion. The capture searched 10.09.2026 and sent
    639245880000000000, which is 2026-09-09 22:00 UTC -- local midnight in
    Skopje, expressed in UTC. A naive midnight is two hours later in September
    and would ask the registry about the wrong day.
    """
    day = datetime.strptime(as_date(ddmmyyyy), "%d.%m.%Y")
    utc = day - timedelta(hours=_utc_offset_hours(day.date()))
    return int((utc - datetime(1, 1, 1)).total_seconds()) * 10_000_000


def build_payload(args: SearchArgs) -> dict:
    """The JSON body the portal's own search sends. Verified against a capture.

    Unused -- see the module docstring -- but written and tested, so that
    implementing the live call is wiring rather than guesswork.
    """
    code = next((c for c, label in DOC_TYPE_CODES.items()
                 if norm_text(label) == norm_text(args.doc_type)), None)
    return {
        "LEID": int(args.embs) if args.embs else None,
        "partOfName": args.name or None,
        "docTypeID": code,
        "announcementDateTicks": date_ticks(args.date) if args.date else None,
        "ut": False,
    }


class AnnouncementHit(BaseModel):
    """One filing found. `deloveden_broj` is what Form 3 takes."""
    deloveden_broj: str
    embs: str
    full_name: str
    entry_type: str
    dates: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    criteria: str
    hits: list[AnnouncementHit]
    searched: int                 # filings in the offline archive
    fetched_at: str
    source: str = "offline:assets"
    available: bool = True
    found: bool = False

    def as_context_doc(self, rank: int = 0) -> ContextDoc:
        key = re.sub(r"\W+", "-", self.criteria)[:60].strip("-").lower()
        chunk_id = f"tool:announcement_search:{key}"
        if self.hits:
            lines = [f"Пронајдени објави за: {self.criteria} "
                     f"({len(self.hits)} од {self.searched} зачувани решенија)"]
            for h in self.hits:
                lines.append(
                    f"Деловоден број {h.deloveden_broj} -- {h.entry_type}; "
                    f"ЕМБС {h.embs}; {h.full_name}"
                    + (f"; датум {', '.join(h.dates)}" if h.dates else ""))
        else:
            # SCOPE, not a verdict. This text is what the model relays, so the
            # limit has to be in the sentence itself -- "нема зачувано" (none
            # saved), never "нема решенија" (none exist).
            lines = [
                f"Нема пронајдено објави за: {self.criteria}.",
                f"Пребарувањето е извршено само низ офлајн архивата на агентот "
                f"({self.searched} зачувани решенија), не низ Централниот "
                f"регистар. Регистарот може да има објави што овде ги нема -- "
                f"проверете на порталот, во делот Објави на уписи за субјекти."]
        text = "\n".join(lines)
        block = Block(
            chunk_id=chunk_id,
            context=(f"Пребарување на објави на уписи ({self.criteria}), "
                     f"извршено {self.fetched_at}"),
            content=text,
            embedding_text=text,
            scope="service",
            type="live_lookup",
            id_service=None,
            service_name="Објави на уписи за субјекти",
        )
        return ContextDoc(chunk_id=chunk_id, block=block, score=1.0, rank=rank)


def dates_of(decision: RegistrationDecision) -> list[str]:
    """Every date the decision prints, normalised. Order preserved, deduped."""
    seen: dict[str, None] = {}
    for section in decision.sections:
        for row in section.rows:
            iso = as_date(row.value)
            if iso:
                seen.setdefault(iso, None)
    return list(seen)


def as_hit(decision: RegistrationDecision) -> AnnouncementHit:
    return AnnouncementHit(
        deloveden_broj=decision.deloveden_broj, embs=decision.embs,
        full_name=decision.full_name, entry_type=decision.entry_type,
        dates=dates_of(decision))


def matches(decision: RegistrationDecision, args: SearchArgs) -> bool:
    """Every criterion the user gave must hold. AND, not OR: a search for
    'ЛОРА' AND '17.09.2026' that returns everything named ЛОРА is a wrong
    answer wearing the shape of a right one."""
    if args.embs and not same_embs(decision.embs, args.embs):
        return False
    if args.name and norm_text(args.name) not in norm_text(decision.full_name):
        return False
    if args.doc_type and norm_text(args.doc_type) not in norm_text(decision.entry_type):
        return False
    if args.date and as_date(args.date) not in dates_of(decision):
        return False
    return True


class AnnouncementIndex(Protocol):
    """Where filings are searched. The seam a live search endpoint plugs into.

    `all_decisions()` returns everything searchable; filtering is done here so
    every implementation agrees on what a criterion MEANS. A live source that
    filters server-side would implement `search()` instead -- add it then, with
    the endpoint in hand, rather than guessing the query shape now.
    """
    name: str

    def all_decisions(self) -> list[RegistrationDecision]: ...


class LocalDecisionIndex:
    """The decisions saved in assets/. Searchable because they are readable.

    Coverage is exactly the set Form 3 can open, which is the property that
    matters: this tool never hands the agent a number it cannot then fetch.
    """
    name = "offline:assets"

    def __init__(self, directory: Path | str = ASSETS, *, client=None):
        self.directory = Path(directory)
        self.source = LocalDecisionSource(self.directory, client=client)
        self.rejected: dict[str, str] = {}

    def numbers(self) -> list[str]:
        """Filing numbers on disk, read from the filenames -- no vision call."""
        prefix, suffix = PATTERN.split("{number}")
        found = []
        for path in sorted(self.directory.glob(prefix + "*" + suffix)):
            number = path.name[len(prefix):len(path.name) - len(suffix)]
            if re.fullmatch(rf"\d{{{DELOVODEN_DIGITS}}}", number):
                found.append(number)
        return found

    def all_decisions(self) -> list[RegistrationDecision]:
        """Every saved decision we can identify and that passes its checks.

        A file whose printed number still disagrees with its filename after the
        re-read is SKIPPED, not offered with the number from either side: we do
        not know which filing it is, and a search result is a promise that
        get_registration_decision can then open it. Skipped files are kept in
        `rejected` so the operator can be told which ones, rather than quietly
        having a document that never appears in any search.

        Each load is cached per file version, so a search followed by a fetch of
        one of its hits costs one vision call, not two.
        """
        self.rejected: dict[str, str] = {}
        out = []
        for number in self.numbers():
            decision = self.source.load(number)
            if decision is None:
                continue
            if digits(decision.deloveden_broj) != number:
                self.rejected[number] = (
                    f"сликата покажува деловоден број "
                    f"{decision.deloveden_broj!r}")
                continue
            problems = verify_decision(decision)
            if problems:
                self.rejected[number] = "; ".join(problems)
                continue
            out.append(decision)
        return out


_DEFAULT_INDEX: AnnouncementIndex | None = None


def default_index() -> AnnouncementIndex:
    global _DEFAULT_INDEX
    if _DEFAULT_INDEX is None:
        _DEFAULT_INDEX = LocalDecisionIndex()
    return _DEFAULT_INDEX


def set_default_index(index: AnnouncementIndex | None) -> None:
    """Swap the search backend. None restores the offline default."""
    global _DEFAULT_INDEX
    _DEFAULT_INDEX = index


def search_announcements(embs: str = "", name: str = "", doc_type: str = "",
                         date: str = "", *,
                         index: AnnouncementIndex | None = None) -> SearchResult:
    """Find filings matching the criteria. Never invents one; may find none."""
    args = SearchArgs(embs=embs, name=name, doc_type=doc_type, date=date)
    src = index if index is not None else default_index()
    decisions = src.all_decisions()
    hits = [as_hit(d) for d in decisions if matches(d, args)]
    return SearchResult(
        criteria=args.describe(), hits=hits, searched=len(decisions),
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source=src.name, found=bool(hits))


TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "search_announcements",
        "description": (
            "Пребарува објави на уписи (решенија) на Централниот регистар и го "
            "враќа ДЕЛОВОДНИОТ БРОЈ на секоја пронајдена објава. Користи го кога "
            "корисникот бара решение или објава за субјект, а НЕ дал деловоден "
            "број -- на пр. по ЕМБС, по назив на субјектот, по вид на упис или "
            "по датум. Потоа, за содржината на конкретно решение, повикај "
            "get_registration_decision со добиениот деловоден број. Пребарувањето "
            "оди низ офлајн архивата на агентот, не низ целиот регистар: ако не "
            "врати ништо, кажи дека во архивата нема такви објави и упати на "
            "порталот -- НЕ тврди дека субјектот нема решенија и НЕ измислувај "
            "деловоден број."),
        "parameters": strict_schema(SearchArgs),
        "strict": True,
    },
}
