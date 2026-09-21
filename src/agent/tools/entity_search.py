"""
entity_search.py  --  finding an ЕМБС, so Form 2 can be used.
=============================================================
Form 2 fetches a profile by ЕМБС. Users have a company name:

    "основните податоци за ЛОРА КОМПАНИ"
        -> search_entity_profile(name="ЛОРА КОМПАНИ")  -> 7696876
        -> get_entity_profile("7696876")

Same shape as announcement_search, and the same rule: the resolver exists so the
agent stops guessing, and a resolver that returns a plausible ЕМБС it did not
find would be worse than none -- Form 2 would fetch it and return a real profile
for the wrong company, which reads exactly like the right answer.

WHERE THE NAMES COME FROM
-------------------------
Two sources, because the agent already holds two kinds of document that name an
entity: the saved profile images (Form 2) and the saved decisions (Form 3). A
decision prints ЕМБС and Целосен назив, so it identifies a company perfectly
well even when no profile for it has been saved -- which is exactly the case for
ТОПОЛЧАН АГРАР.

That makes `has_profile` load-bearing rather than decorative. An entity known
only from a decision can be named and cited, but get_entity_profile will report
it unavailable, and the model is told so up front instead of discovering it in a
second round.

COST, AND WHY THIS IS NOT THE FINAL SHAPE
-----------------------------------------
Names live inside the images, so building the index means reading every saved
document once: a vision call per file, cached afterwards per file version. Fine
for the handful saved today, wrong at a hundred. The fix when it matters is a
sidecar index written at save time, which is a change to this class and nothing
else -- the tool, the seam and the agent do not know how the index is built.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field, field_validator, model_validator

from eval.corpus import Block

from ..context import ContextDoc
from .announcement_search import LocalDecisionIndex
from .entity_profile import ASSETS, LocalImageSource
from .matching import norm_text, same_embs
from .schema import strict_schema

PROFILE_GLOB = "basic_profile_*.png"


class EntitySearchArgs(BaseModel):
    name: str = Field(description=(
        "Part of the entity's name, as the user wrote it. Empty string if the "
        "user did not name one."))
    embs: str = Field(description=(
        "ЕМБС, 7-8 digits, if the user happens to have given one. Empty string "
        "otherwise -- if the user gave an ЕМБС and wants the profile, call "
        "get_entity_profile directly instead of searching."))

    @field_validator("embs")
    @classmethod
    def _embs_shape(cls, v: str) -> str:
        v = v.strip()
        if v and not re.fullmatch(r"\d{7,8}", v):
            raise ValueError("ЕМБС мора да биде 7 или 8 цифри")
        return v

    @model_validator(mode="after")
    def _at_least_one(self) -> EntitySearchArgs:
        if not (self.name or self.embs):
            raise ValueError("Наведи назив на субјектот или ЕМБС")
        return self

    def describe(self) -> str:
        parts = [f"назив содржи „{self.name}“" if self.name else "",
                 f"ЕМБС {self.embs}" if self.embs else ""]
        return ", ".join(p for p in parts if p)


class EntityHit(BaseModel):
    embs: str
    full_name: str
    has_profile: bool          # can get_entity_profile actually serve this one
    seen_in: list[str] = Field(default_factory=list)   # профил / решение


class EntitySearchResult(BaseModel):
    criteria: str
    hits: list[EntityHit]
    searched: int
    fetched_at: str
    source: str = "offline:assets"
    available: bool = True
    found: bool = False

    def as_context_doc(self, rank: int = 0) -> ContextDoc:
        key = re.sub(r"\W+", "-", self.criteria)[:60].strip("-").lower()
        chunk_id = f"tool:entity_search:{key}"
        if self.hits:
            lines = [f"Пронајдени субјекти за: {self.criteria} "
                     f"({len(self.hits)} од {self.searched} зачувани)"]
            for h in self.hits:
                lines.append(
                    f"ЕМБС {h.embs} -- {h.full_name}"
                    + ("" if h.has_profile
                       else " (основниот профил не е зачуван офлајн)"))
        else:
            # The same scope warning announcement_search carries, for the same
            # reason: this searched OUR archive. "Not found here" is not
            # "not registered", and only one of those two is something we know.
            lines = [
                f"Нема пронајден субјект за: {self.criteria}.",
                f"Пребарувањето е извршено само низ офлајн архивата на агентот "
                f"({self.searched} зачувани субјекти), не низ Централниот "
                f"регистар. Субјектот може да постои во регистарот -- "
                f"проверете на порталот на Централниот регистар."]
        text = "\n".join(lines)
        block = Block(
            chunk_id=chunk_id,
            context=(f"Пребарување на субјекти ({self.criteria}), извршено "
                     f"{self.fetched_at}"),
            content=text,
            embedding_text=text,
            scope="service",
            type="live_lookup",
            id_service=None,
            service_name="Основен профил на регистриран субјект",
        )
        return ContextDoc(chunk_id=chunk_id, block=block, score=1.0, rank=rank)


def matches(hit: EntityHit, args: EntitySearchArgs) -> bool:
    if args.embs and not same_embs(hit.embs, args.embs):
        return False
    if args.name and norm_text(args.name) not in norm_text(hit.full_name):
        return False
    return True


class EntityIndex(Protocol):
    """Where entities are searched. The seam a live name search plugs into."""
    name: str

    def entities(self) -> list[EntityHit]: ...


class LocalEntityIndex:
    """Entities named by the documents in assets/ -- profiles and decisions."""
    name = "offline:assets"

    def __init__(self, directory: Path | str = ASSETS, *, client=None):
        self.directory = Path(directory)
        self.profiles = LocalImageSource(self.directory, client=client)
        self.decisions = LocalDecisionIndex(self.directory, client=client)
        self.rejected: dict[str, str] = {}

    def profile_numbers(self) -> list[str]:
        """ЕМБС values from filenames -- no vision call to list them."""
        prefix, suffix = "basic_profile_", ".png"
        out = []
        for path in sorted(self.directory.glob(PROFILE_GLOB)):
            embs = path.name[len(prefix):len(path.name) - len(suffix)]
            if re.fullmatch(r"\d{7,8}", embs):
                out.append(embs)
        return out

    def entities(self) -> list[EntityHit]:
        by_embs: dict[str, EntityHit] = {}

        def record(embs: str, full_name: str, seen: str, has_profile: bool):
            key = embs.lstrip("0") or embs
            existing = by_embs.get(key)
            if existing is None:
                by_embs[key] = EntityHit(embs=embs, full_name=full_name,
                                         has_profile=has_profile, seen_in=[seen])
                return
            if seen not in existing.seen_in:
                existing.seen_in.append(seen)
            existing.has_profile = existing.has_profile or has_profile
            # A profile's own name wins: it is the registry's heading for the
            # entity, while a decision quotes it inside a filing.
            if seen == "профил":
                existing.full_name = full_name

        self.rejected: dict[str, str] = {}
        for embs in self.profile_numbers():
            profile = self.profiles.load(embs)
            if profile is None:
                continue
            # Identity has to be corroborated by the filename, for the reason
            # LocalDecisionIndex.all_decisions gives: a hit is a promise that
            # get_entity_profile can then open it under this ЕМБС.
            if not same_embs(profile.embs, embs):
                self.rejected[embs] = (
                    f"сликата покажува ЕМБС {profile.embs!r}")
                continue
            record(profile.embs or embs, profile.full_name, "профил", True)
        for decision in self.decisions.all_decisions():
            record(decision.embs, decision.full_name, "решение",
                   has_profile=False)
        return list(by_embs.values())


_DEFAULT_INDEX: EntityIndex | None = None


def default_index() -> EntityIndex:
    global _DEFAULT_INDEX
    if _DEFAULT_INDEX is None:
        _DEFAULT_INDEX = LocalEntityIndex()
    return _DEFAULT_INDEX


def set_default_index(index: EntityIndex | None) -> None:
    global _DEFAULT_INDEX
    _DEFAULT_INDEX = index


def search_entity_profile(name: str = "", embs: str = "", *,
                          index: EntityIndex | None = None) -> EntitySearchResult:
    """Find entities by name. Never invents an ЕМБС; may find none."""
    args = EntitySearchArgs(name=name, embs=embs)
    src = index if index is not None else default_index()
    known = src.entities()
    hits = [h for h in known if matches(h, args)]
    return EntitySearchResult(
        criteria=args.describe(), hits=hits, searched=len(known),
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source=src.name, found=bool(hits))


TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "search_entity_profile",
        "description": (
            "Го наоѓа ЕМБС на субјект според неговиот назив. Користи го кога "
            "корисникот бара податоци за субјект по ИМЕ, а не дал ЕМБС -- потоа "
            "повикај get_entity_profile со добиениот ЕМБС. Ако корисникот веќе "
            "дал ЕМБС, не пребарувај, туку повикај го директно потребниот алат. "
            "Ако резултатот покажува дека основниот профил не е зачуван офлајн, "
            "кажи го тоа наместо да го бараш профилот. Пребарувањето оди низ "
            "офлајн архивата на агентот, не низ целиот регистар: ако не врати "
            "ништо, кажи дека во архивата нема таков субјект и упати на "
            "порталот -- НЕ тврди дека субјектот не постои и НЕ измислувај ЕМБС."),
        "parameters": strict_schema(EntitySearchArgs),
        "strict": True,
    },
}
