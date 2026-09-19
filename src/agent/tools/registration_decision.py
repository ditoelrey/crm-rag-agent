"""
registration_decision.py  --  Form 3: Објави на уписи за субјекти (Решение).
============================================================================
A published registration decision, looked up by its деловоден број.

THE IDENTIFIER IS NOT THE ЕМБС
------------------------------
Forms 1 and 2 are keyed by the entity. This one is keyed by the FILING: the
endpoint is `announcement/oss/{деловоден број}`, a 14-digit number, and one
entity accumulates many decisions over its life. So the tool takes the
деловоден број and nothing else, and an ЕМБС handed to it fails validation
before anything runs -- the model is then told why and asks the user for the
number. It must never invent one: a fabricated filing number either misses (a
wasted turn) or, worse, hits some other company's decision.

THE SCHEMA IS NOT A TEMPLATE
----------------------------
The profile (Form 2) is one fixed 11-field table. A Решение is not: its tables
depend on the "Вид на упис". The one sample we have -- a decision fixing an
entity's main revenue code and size -- carries Деловодник, Основни податоци and
Дејности; a founding decision will carry founders, managers and capital, a
change decision the changed fields. Freezing this sample's rows into fields
would silently drop most of every other kind of decision.

So there are two layers:

    fixed fields   deloveden_broj, embs, entry_type, decision_date, full_name
                   -- what the self-check and the shape checks need
    sections       EVERY table in the image, verbatim, as (title, rows)
                   -- what gets cited, whatever kind of decision it is

The identifiers appear in both, and the two reads must agree. That is a free
consistency check: the model has to copy the same number twice, and a misread
digit in one of them fails the document instead of reaching the user.

TABLES ONLY -- THE PREAMBLE IS NOT READ
---------------------------------------
Above РЕШЕНИЕ sits a paragraph of small-font prose: the registrar's name, the
laws the decision rests on, the date. The first extraction read every table
cell perfectly and made three errors in that one paragraph, checked against the
image at native resolution:

    Регистраторот  -> Регистратор      (inflection dropped)
    Голејшки       -> Голешки          (a real official's surname, misspelt)
    ЗЕШС           -> ЗЕЈС             (the wrong law)

and no check noticed, because prose has no structure to check. The rule this
project applies to vision is that a read must be falsifiable or it does not
ship, so the paragraph is skipped. Nothing a user needs is lost: the dates live
in the Деловодник table (Прием / Одобрување на пријавата), where they read
cleanly, and the legal basis of each service is in the corpus.

SELF-CHECK
----------
As in Form 2: the деловоден број printed in the image must be the one asked
for, or the result is REFUSED. The failure it catches is right-looking data from
the wrong filing.

ACQUISITION
-----------
Same portal, same screenshot service, same reCAPTCHA v3 header as Form 2, so the
same rule: this module does not acquire a token. It reads PNGs an operator saved
from the portal (assets/decision_<деловоден број>.png) and reports a clean
"not available" for anything else. DecisionSource is the seam an official feed
would plug into.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field, field_validator

from eval.corpus import Block

from ..context import ContextDoc
from .entity_size import KNOWN_SIZES
from .schema import strict_schema
from .vision import ImageCache, extract_structured

ASSETS = Path(__file__).parent / "assets"
PATTERN = "decision_{number}.png"

# From the capture. Not called -- see ACQUISITION -- but recorded so the live
# shape is documented next to the code that would use it.
ANNOUNCEMENT_URL = ("https://www.crm.com.mk/CRMPublicPortalApi/api/freeservice/"
                    "announcement/oss/{number}")
PAGE_URL = ("https://www.crm.com.mk/mk/otvoreni-podatotsi/"
            "objavi-na-upisi-za-subjekti")

# Both numbers seen so far (the capture's 30120260014978 and the saved image's
# 30120260014967) are 14 digits: "301" + year + a 7-digit sequence. Two samples
# are enough to validate against, not enough to parse the structure out of.
DELOVODEN_DIGITS = 14


class DecisionArgs(BaseModel):
    # No example number, deliberately. The obvious one is the saved filing, and
    # a model that copies an example instead of asking the user would then get
    # a real decision about a real company -- one nobody asked about.
    deloveden_broj: str = Field(description=(
        "Деловоден број на решението exactly as the user gave it -- 14 "
        "digits. This is NOT the ЕМБС (which has 7-8 digits)."))

    @field_validator("deloveden_broj")
    @classmethod
    def _fourteen_digits(cls, v: str) -> str:
        v = v.strip()
        if not re.fullmatch(rf"\d{{{DELOVODEN_DIGITS}}}", v):
            raise ValueError(
                f"Деловодниот број мора да има {DELOVODEN_DIGITS} цифри. "
                f"ЕМБС не е деловоден број -- побарај го бројот од решението.")
        return v


class DecisionRow(BaseModel):
    """One row of one table."""
    label: str = Field(description="Left cell, verbatim, e.g. 'Вид на упис'")
    value: str = Field(description="Right cell, verbatim, e.g. '17 септ. 2026'")


class DecisionSection(BaseModel):
    """One titled table."""
    title: str = Field(description="The table's heading, e.g. 'Деловодник'")
    rows: list[DecisionRow]


class RegistrationDecision(BaseModel):
    """A Решение as printed. See the module docstring for the two layers."""
    deloveden_broj: str = Field(description="Деловоден број, digits only")
    entry_type: str = Field(description="Вид на упис, verbatim")
    embs: str = Field(description="ЕМБС of the subject, digits only")
    full_name: str = Field(description=(
        "Целосен назив of the subject, verbatim; a name that wraps onto two "
        "lines is one string"))
    sections: list[DecisionSection] = Field(description=(
        "EVERY table in the image, top to bottom -- including the ones the "
        "fields above were read from -- each row copied verbatim. Tables "
        "only: the paragraph above РЕШЕНИЕ is not a table"))


class DecisionResult(BaseModel):
    decision: RegistrationDecision
    fetched_at: str
    source: str = "vision:file"
    available: bool = True
    requested: str = ""          # the деловоден број as asked -- identity

    def as_context_doc(self, rank: int = 0) -> ContextDoc:
        """Citable. Rendered from `sections`, which is the complete record."""
        d = self.decision
        number = self.requested or d.deloveden_broj
        chunk_id = f"tool:registration_decision:{number}"
        # Labelled lines grouped under their table titles, not JSON: this text
        # is what numeric_groundedness and value_citation check answers
        # against, so every number the user can be told must appear verbatim.
        lines = [f"Решение, деловоден број {d.deloveden_broj}"]
        for section in d.sections:
            lines.append(f"[{section.title}]")
            lines += [f"{row.label}: {row.value}" for row in section.rows]
        text = "\n".join(filter(None, lines))
        block = Block(
            chunk_id=chunk_id,
            context=(f"Објавено решение за упис од Централен регистар "
                     f"(деловоден број {number}, ЕМБС {d.embs}), прочитано "
                     f"{self.fetched_at}"),
            content=text,
            embedding_text=text,
            scope="service",
            # Not a corpus section type: must stay invisible to the coverage
            # block and the ambiguity detectors. Same reasoning as Form 1.
            type="live_lookup",
            id_service=None,
            service_name="Објави на уписи за субјекти",
        )
        return ContextDoc(chunk_id=chunk_id, block=block, score=1.0, rank=rank)


class DecisionUnavailable(BaseModel):
    """No saved image for this filing -- an ANSWER, not an error.

    Same shape and the same reasons as entity_profile.ProfileUnavailable: the
    tool loop makes every result citable, and the abstention guard deletes an
    uncited answer, so a miss has to be a result or an honest "not held" is
    replaced by the generic abstention.
    """
    deloveden_broj: str
    available: bool = False
    reason: str
    message: str

    def as_context_doc(self, rank: int = 0) -> ContextDoc:
        chunk_id = f"tool:registration_decision:{self.deloveden_broj}:unavailable"
        text = f"Деловоден број {self.deloveden_broj}: {self.message}"
        block = Block(
            chunk_id=chunk_id,
            context=("Обид за читање на објавено решение за упис "
                     f"(деловоден број {self.deloveden_broj})"),
            content=text,
            embedding_text=text,
            scope="service",
            type="live_lookup",
            id_service=None,
            service_name="Објави на уписи за субјекти",
        )
        return ContextDoc(chunk_id=chunk_id, block=block, score=1.0, rank=rank)


class DecisionMismatch(RuntimeError):
    """The image is not the filing we asked about. Raised, never returned."""


_EXTRACT_PROMPT = (
    "Read this registration decision (РЕШЕНИЕ) from the Macedonian Central "
    "Registry. Copy everything exactly as printed: do not translate, reformat, "
    "normalise or expand anything -- keep the Cyrillic text, codes, numbers "
    "and dates verbatim, so a date printed '17 септ. 2026' stays '17 септ. "
    "2026'. A label or value that wraps onto a second line is still one row. "
    "Put EVERY table in `sections`, top to bottom, and skip the introductory "
    "paragraph above the word РЕШЕНИЕ -- it is not a table. If the image is "
    "cut off, "
    "stop at the last row you can read in full; never infer a value that is "
    "not visible.")


def extract_decision(image_bytes: bytes, *, client=None) -> RegistrationDecision:
    """Read the decision out of the PNG. The one non-deterministic step."""
    return extract_structured(image_bytes, RegistrationDecision, _EXTRACT_PROMPT,
                              name="registration_decision", client=client)


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text or "")


def _row(decision: RegistrationDecision, label: str) -> str | None:
    """The value of the first row whose label starts with `label`, if any."""
    for section in decision.sections:
        for row in section.rows:
            if row.label.strip().lower().startswith(label.lower()):
                return row.value
    return None


def verify_decision(decision: RegistrationDecision) -> list[str]:
    """Shape and consistency checks. Returns the problems; empty means sane.

    Nothing here proves the read right. Each check catches a realistic misread:
    a dropped digit, a reformatted date, a size class the registry does not
    use, or the two copies of an identifier disagreeing with each other.
    """
    problems: list[str] = []
    if len(_digits(decision.deloveden_broj)) != DELOVODEN_DIGITS:
        problems.append(f"деловоден број {decision.deloveden_broj!r} does not "
                        f"have {DELOVODEN_DIGITS} digits")
    if not re.fullmatch(r"\d{7,8}", _digits(decision.embs)):
        problems.append(f"ЕМБС {decision.embs!r} is not 7-8 digits")
    if not decision.sections or not any(s.rows for s in decision.sections):
        problems.append("no tables were read")

    # The same identifier, read twice. Disagreement means one read is wrong
    # and there is no way to tell which, so the document is not trusted.
    for label, field in (("Деловоден број", decision.deloveden_broj),
                         ("ЕМБС", decision.embs)):
        printed = _row(decision, label)
        if printed is not None and (_digits(printed).lstrip("0")
                                    != _digits(field).lstrip("0")):
            problems.append(f"{label} read as {field!r} and as {printed!r}")

    size = _row(decision, "Големина")
    if size is not None and size.strip().lower() not in KNOWN_SIZES:
        problems.append(f"големина {size!r} is not a registry size class")
    return problems


class DecisionSource(Protocol):
    """Where a RegistrationDecision comes from -- fields, not bytes, for the
    reason entity_profile.ProfileSource gives. None means "not held"; raising
    means the source is broken."""
    name: str

    def load(self, deloveden_broj: str) -> RegistrationDecision | None: ...


class LocalDecisionSource:
    """Decisions from PNGs an operator saved out of the portal. No network."""
    name = "vision:file"

    def __init__(self, directory: Path | str = ASSETS, *, client=None,
                 pattern: str = PATTERN):
        self.directory = Path(directory)
        self.client = client
        self.pattern = pattern
        self._cache = ImageCache()

    def path_for(self, deloveden_broj: str) -> Path:
        # No padding variants, unlike the ЕМБС: every деловоден број seen is a
        # full 14 digits and the validator refuses anything shorter.
        return self.directory / self.pattern.format(number=deloveden_broj)

    def load(self, deloveden_broj: str) -> RegistrationDecision | None:
        path = self.path_for(deloveden_broj)
        if not path.is_file():
            return None
        return self._cache.get(
            path, lambda data: extract_decision(data, client=self.client))


_DEFAULT_SOURCE: DecisionSource | None = None


def default_source() -> DecisionSource:
    """A singleton, so the extraction cache actually hits across calls."""
    global _DEFAULT_SOURCE
    if _DEFAULT_SOURCE is None:
        _DEFAULT_SOURCE = LocalDecisionSource()
    return _DEFAULT_SOURCE


def set_default_source(source: DecisionSource | None) -> None:
    """Swap the transport. None restores the offline default."""
    global _DEFAULT_SOURCE
    _DEFAULT_SOURCE = source


def fetch_decision(deloveden_broj: str, *, source: DecisionSource | None = None,
                   ) -> DecisionResult | DecisionUnavailable:
    """Look up one published decision. Verified fields, or a clean miss.

    A miss comes back as a value; a document that fails its checks RAISES,
    because it means the saved file or the source is wrong, and _run_tool's rule
    is that a broken integration fails loudly instead of being narrated around.
    """
    args = DecisionArgs(deloveden_broj=deloveden_broj)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    src = source if source is not None else default_source()

    decision = src.load(args.deloveden_broj)
    if decision is None:
        return DecisionUnavailable(
            deloveden_broj=args.deloveden_broj, reason="no_local_image",
            message=("Решението со овој деловоден број не е достапно офлајн. "
                     "Објавите на уписи од Централниот регистар се добиваат "
                     "како слика од порталот, а зачувана слика за овој "
                     "деловоден број нема. Решението може да се прегледа "
                     "рачно на порталот на Централниот регистар, во делот "
                     "Објави на уписи за субјекти."))

    if _digits(decision.deloveden_broj) != args.deloveden_broj:
        raise DecisionMismatch(
            f"asked for деловоден број {args.deloveden_broj}, {src.name} "
            f"returned {decision.deloveden_broj!r} -- refusing to return "
            f"another filing's decision")
    problems = verify_decision(decision)
    if problems:
        raise ValueError("decision failed checks: " + "; ".join(problems))
    return DecisionResult(decision=decision, fetched_at=fetched_at,
                          source=src.name, requested=args.deloveden_broj)


TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "get_registration_decision",
        "description": (
            "Го враќа објавеното решение за упис од Централниот регистар според "
            "неговиот ДЕЛОВОДЕН БРОЈ (14 цифри): вид на "
            "упис, датум, субјект (ЕМБС и назив) и сите податоци од решението. "
            "Користи го САМО кога корисникот дал деловоден број. ЕМБС (7-8 "
            "цифри) НЕ е деловоден број: ако корисникот дал само ЕМБС, побарај "
            "го деловодниот број од решението и НИКОГАШ не измислувај деловоден "
            "број. За големина или основен профил на субјект користи "
            "check_entity_size или get_entity_profile. Ако одговорот има "
            "available=false, решението не е достапно -- пренеси ја пораката на "
            "корисникот и НЕ го повикувај алатот повторно за истиот број."),
        "parameters": strict_schema(DecisionArgs),
        "strict": True,
    },
}
