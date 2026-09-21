"""
entity_profile.py  --  Form 2: Основен профил на регистриран субјект.
=====================================================================
The registry renders this profile as a PNG rather than as text. Reading it needs
vision, which puts a non-deterministic step in the middle of a pipeline built on
the opposite principle -- so the design question is not "can gpt-4o read the
table" (it can) but "how do we know it read it correctly".

MAKING A PROBABILISTIC READ FALSIFIABLE
---------------------------------------
The image contains the ЕМБС we searched for. So the extraction carries its own
check: if the ЕМБС in the picture is not the one we asked about, something is
wrong -- wrong entity fetched, a stale cached image, or a misread -- and the
result is REJECTED rather than returned with a caveat. That converts "the model
probably read it right" into a claim that can fail loudly, and it catches the
failure that actually costs a user something: right-looking data for the wrong
company.

Three further shape checks are free and catch a misread digit: ЕДБ is 13 digits,
the founding date is a real date, and the size class is one the registry uses.

Scope: LOOKUP BY ЕМБС ONLY
--------------------------
The live form also searches by name, which returns a set rather than a row and
has no self-check available -- there is nothing to compare the picture against.
Name search is therefore a separate tool and a separate decision, not a second
argument bolted onto this one.

HOW THE IMAGE IS PRODUCED, AND WHY WE DO NOT FETCH IT OURSELVES
---------------------------------------------------------------
Form 2 is not a WebForm like Form 1. The portal is an Angular SPA and the
request is a server-side SCREENSHOT service, which the DevTools capture makes
unambiguous:

    POST /CRMPublicPortalApi/api/freeservice/basicProfile/{embs}?sci={viewport}
    body: 120 KB of the page's own HTML, base64'd -- the UNFILLED template

The body carries eleven placeholders ([^$.LEID^], [^$.LeSize^], ...) and zero
entity values. The client ships the template, the server substitutes the fields
for the ЕМБС in the path and renders the result to PNG at the declared viewport.
There is consequently no JSON endpoint to prefer: the registry does not expose
this data as text anywhere in the flow.

The same capture carries a `recaptcha:` header -- reCAPTCHA v3, site key
6LcLUNAZAAAAAJ08HQkGbOwh5F2RP5LCpxwQycdS, loaded invisibly by the page. That is
a bot-detection control, so THIS MODULE DOES NOT ACQUIRE ONE. `PortalSession`
has no default and no environment fallback: an operator who legitimately holds
session material can pass it, and if nobody does, the tool does not run. The
supported path meanwhile is `profile_image_from_file()` -- a human opens the
portal, saves the PNG, and everything downstream is identical.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import httpx
from pydantic import BaseModel, Field

from eval.corpus import Block

from ..context import ContextDoc
from .entity_size import KNOWN_SIZES, EntitySizeArgs
from .schema import strict_schema
from .vision import (MAX_IMAGE_BYTES, VISION_MODEL, ImageCache,  # noqa: F401
                     extract_structured, read_png)

PROFILE_URL = "https://www.crm.com.mk/CRMPublicPortalApi/api/freeservice/basicProfile/{embs}"
PAGE_URL = ("https://www.crm.com.mk/mk/otvoreni-podatotsi/"
            "osnoven-profil-na-registriran-subjekt")

# Not "fixtures/": the template is a runtime asset the live path needs, and a
# sibling fixtures.py already owns that name for recorded tool responses.
ASSETS = Path(__file__).parent / "assets"
TEMPLATE_PATH = ASSETS / "basic_profile_template.html"

# Layout inputs, not identity: `sci` decides the PNG's dimensions and nothing
# else. Taken from the capture, whose render came back 976x730.
SCI = {"screenSizeW": 1536, "screenSizeH": 864, "vpSizeW": 1048, "vpSizeH": 730,
       "scrollOffsetX": 0, "scrollOffsetY": 49, "docWidth": 1033,
       "docHeight": 1482, "isMobile": False, "deviceScaleFactor": 1}


class ProfileArgs(BaseModel):
    # No example number -- see EntitySizeArgs. The old one was ЛОРА's real ЕМБС.
    embs: str = Field(description=(
        "ЕМБС (Единствен матичен број на субјектот) -- the 7- or 8-digit "
        "registration number, digits only, exactly as the user gave it. Never "
        "add or remove a leading zero."))


class EntityProfile(BaseModel):
    """One row of the registry's basic profile.

    Every field is a plain string: the picture is the source of truth and a row
    it does not show is "" rather than a guess. Dates stay in the registry's own
    dd.mm.yyyy so nothing is silently reformatted between the image and the user.

    Field ORDER follows the image top to bottom, which is also the order the
    template declares. A vision model filling a schema out of reading order has
    to hold the table in mind while it jumps around it; matching the layout
    costs nothing and removes the opportunity.
    """
    full_name: str = Field(description=(
        "Целосен назив -- the bold heading above the table, not a row in it, "
        "e.g. 'Друштво за производство, трговија и услуги ЛОРА КОМПАНИ 2023 "
        "ДОО Скопје'"))
    embs: str = Field(description="ЕМБС, digits only")
    edb: str = Field(description="ЕДБ (даночен број), digits only, usually 13")
    short_name: str = Field(description="Скратен назив, verbatim")
    founded: str = Field(description="Датум на основање as printed, e.g. 19.09.2023")
    legal_form: str = Field(description="Правна форма as printed, e.g. "
                                        "'05.3 - друштво со ограничена одговорност'")
    legal_status: str = Field(description="Правен статус, e.g. Активен")
    address: str = Field(description="Адреса, verbatim including municipality")
    additional_info: str = Field(description=(
        "Дополнителни податоци, if the table shows such a row. Most profiles "
        "do not have one -- return \"\" then, never a guess"))
    activity: str = Field(description="Претежна дејност as printed, with its code")
    size: str = Field(description="Големина: микро, мал, среден or голем")


class ProfileResult(BaseModel):
    profile: EntityProfile
    fetched_at: str
    source: str = "vision"       # how the fields were read, for the transcript
    available: bool = True       # the one field both result shapes share
    # The ЕМБС as the USER gave it. Identity, not data -- see as_context_doc.
    requested_embs: str = ""

    def as_context_doc(self, rank: int = 0) -> ContextDoc:
        """Citable, like every other source the answer layer can use."""
        p = self.profile
        # Keyed on what was ASKED, not on what the image printed. The registry
        # prints 7696876 for the entity the user knows as 07696876, so keying
        # on the extracted value hands the model an id it cannot predict from
        # its own question -- it cites the padded form, validate_citations does
        # not match it, and the guard deletes a correct answer. The ЕМБС the
        # user supplied is the one identifier both sides can agree on.
        chunk_id = f"tool:entity_profile:{self.requested_embs or p.embs}"
        # Rendered as labelled lines, not JSON: this text is what
        # numeric_groundedness and value_citation check the answer against, so
        # every number the user might be told has to appear here verbatim.
        # The ЕМБС as ASKED goes in the content as well as the one the image
        # printed, when they differ. Any sane answer repeats the user's own
        # spelling ("субјектот со ЕМБС 07696876 ..."), and numeric_groundedness
        # checks every number in the answer against the cited text -- measured:
        # the only ungrounded number in an otherwise perfect profile answer was
        # the user's own ЕМБС. Form 1 hit the same thing; see EntitySize.
        asked = self.requested_embs
        text = "\n".join(filter(None, [
            f"Побаран ЕМБС: {asked}" if asked and asked != p.embs else "",
            f"Целосен назив: {p.full_name}" if p.full_name else "",
            f"ЕМБС: {p.embs}",
            f"ЕДБ: {p.edb}" if p.edb else "",
            f"Скратен назив: {p.short_name}" if p.short_name else "",
            f"Датум на основање: {p.founded}" if p.founded else "",
            f"Правна форма: {p.legal_form}" if p.legal_form else "",
            f"Правен статус: {p.legal_status}" if p.legal_status else "",
            f"Адреса: {p.address}" if p.address else "",
            f"Дополнителни податоци: {p.additional_info}" if p.additional_info else "",
            f"Претежна дејност: {p.activity}" if p.activity else "",
            f"Големина: {p.size}" if p.size else "",
        ]))
        block = Block(
            chunk_id=chunk_id,
            context=(f"Основен профил на регистриран субјект (ЕМБС {p.embs}), "
                     f"прочитан од Централен регистар {self.fetched_at}"),
            content=text,
            embedding_text=text,
            scope="service",
            type="live_lookup",
            id_service=None,
            service_name="Основен профил на регистриран субјект",
        )
        return ContextDoc(chunk_id=chunk_id, block=block, score=1.0, rank=rank)


class ProfileUnavailable(BaseModel):
    """A profile we have no way to read right now -- an ANSWER, not an error.

    The tool loop turns whatever a tool returns into a citable document and the
    hard-abstention guard deletes any answer that cites nothing. So "I cannot
    get this one" has to arrive as a result with the same shape as a success,
    or the agent's own safety net eats a perfectly honest reply and substitutes
    the generic abstention.

    It carries `available: False` so the model branches on a field rather than
    on the absence of one, and the Macedonian `message` is what it relays.
    """
    embs: str
    available: bool = False
    reason: str                  # machine-readable: no_local_image, no_transport
    message: str                 # what the user should be told, in Macedonian

    def as_context_doc(self, rank: int = 0) -> ContextDoc:
        """Citable too. The unavailability is itself a sourced statement.

        Content is worded so it cannot be mistaken for profile data if it is
        ever read back out of a transcript: no labelled rows, no numbers beyond
        the ЕМБС that was asked about.
        """
        chunk_id = f"tool:entity_profile:{self.embs}:unavailable"
        text = f"ЕМБС {self.embs}: {self.message}"
        block = Block(
            chunk_id=chunk_id,
            context=("Обид за читање на основен профил на регистриран субјект "
                     f"(ЕМБС {self.embs})"),
            content=text,
            embedding_text=text,
            scope="service",
            type="live_lookup",
            id_service=None,
            service_name="Основен профил на регистриран субјект",
        )
        return ContextDoc(chunk_id=chunk_id, block=block, score=1.0, rank=rank)


class ProfileMismatch(RuntimeError):
    """The picture is not of the entity we asked about.

    Raised rather than returned: a profile for the wrong company is the one
    outcome here that is worse than no profile at all, because every field in it
    looks perfectly plausible.
    """


_EXTRACT_PROMPT = (
    "Read this table from the Macedonian Central Registry and return its fields "
    "exactly as printed. Do not translate, reformat or normalise anything -- "
    "copy the Cyrillic text, codes and dates verbatim. If a row is absent from "
    "the image, return an empty string for it; never infer a value.")


def extract_profile(image_bytes: bytes, *, client=None,
                    model: str = VISION_MODEL) -> EntityProfile:
    """Read the profile table out of the PNG. The one non-deterministic step."""
    return extract_structured(image_bytes, EntityProfile, _EXTRACT_PROMPT,
                              name="entity_profile", client=client, model=model)


def verify_profile(profile: EntityProfile, expect_embs: str) -> list[str]:
    """Cheap shape checks. Returns the problems; empty means it looks sane.

    None of these prove the read is right -- nothing can, short of a second
    source -- but each one catches a misread digit, which is the realistic
    failure. A wrong ЕМБС is fatal and handled by the caller; the rest are
    reported so a suspicious field can be shown as suspicious.
    """
    problems: list[str] = []
    if _digits(profile.edb) and len(_digits(profile.edb)) != 13:
        problems.append(f"ЕДБ has {len(_digits(profile.edb))} digits, expected 13")
    if profile.founded and not re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{4}",
                                            profile.founded.strip()):
        problems.append(f"датум на основање {profile.founded!r} is not dd.mm.yyyy")
    if profile.size and profile.size.strip().lower() not in KNOWN_SIZES:
        problems.append(f"големина {profile.size!r} is not a registry size class")
    return problems


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text or "")


def _same_embs(a: str, b: str) -> bool:
    """Is this the same entity, allowing for zero padding?

    The registry pads inconsistently: the profile image for ЛОРА КОМПАНИ prints
    ЕМБС 7696876 while the size form accepts 07696876 for the same company. A
    strict digit comparison therefore rejects a CORRECT profile -- the
    self-check firing on the one entity we had been testing with, and reporting
    it as "another entity's data".

    Padding is stripped only for the comparison. Neither the request nor the
    extracted value is rewritten: the identifier the user gave is the identifier
    they get back.
    """
    return _digits(a).lstrip("0") == _digits(b).lstrip("0")


@dataclass(frozen=True)
class PortalSession:
    """Session material for one authorised call, supplied by the caller.

    Deliberately inert: no default, no constructor that goes and gets one, no
    environment lookup. The endpoint is gated by reCAPTCHA v3 and this project
    does not defeat bot detection -- so the only way a token gets in here is if
    a human who is entitled to it passes it, and the absence of one is a normal
    outcome rather than a problem to route around.
    """
    recaptcha: str
    cookies: dict[str, str]


class PortalRefused(RuntimeError):
    """The registry did not return an image.

    Kept distinct from "no such entity" for the reason parse_entity_size()
    draws the same line: a refused token and an unknown ЕМБС must never reach
    the user as the same sentence.
    """


def _load_template() -> str:
    """The page HTML the server fills in and screenshots.

    Replayed verbatim from the capture rather than rebuilt -- it is 124 KB of
    someone else's Angular output, and the placeholders the server substitutes
    on ([^$.LEID^] and friends) are the only part of it we actually understand.
    """
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"missing profile template at {TEMPLATE_PATH}")
    return TEMPLATE_PATH.read_text(encoding="utf-8")


def fetch_profile_image(embs: str, session: PortalSession, *,
                        template: str | None = None,
                        client: httpx.Client | None = None) -> bytes:
    """POST the page template, get back the rendered PNG.

    A POST despite reading like a fetch: the request BODY is the HTML the server
    renders, the ЕМБС is a path parameter, and `sci` only sizes the output.
    """
    from .crm_http import HEADERS, TIMEOUT_S

    body = _load_template() if template is None else template
    own = client is None
    client = client or httpx.Client(timeout=TIMEOUT_S, follow_redirects=False)
    try:
        resp = client.post(
            PROFILE_URL.format(embs=embs),
            params={"sci": json.dumps(SCI, separators=(",", ":"))},
            content=body.encode("utf-8"),
            cookies=session.cookies,
            headers={
                "Accept": "application/json, text/plain, */*",
                "Content-Type": "text/plain",
                "Origin": "https://www.crm.com.mk",
                "Referer": f"{PAGE_URL}?embs={embs}",
                "recaptcha": session.recaptcha,
                "User-Agent": HEADERS["User-Agent"],
            },
        )
        # A stale token does not come back as 401 -- it comes back as JSON, or
        # as the WAF's HTML page, with a perfectly ordinary status code. The
        # content type is what separates "the registry said no" from an answer.
        ctype = resp.headers.get("content-type", "")
        if not ctype.startswith("image/"):
            raise PortalRefused(
                f"expected an image, got {ctype!r} (HTTP {resp.status_code}). "
                f"The reCAPTCHA token is most likely stale -- v3 tokens last "
                f"about two minutes. First 300 bytes: {resp.content[:300]!r}")
        return resp.content
    finally:
        if own:
            client.close()


def profile_image_from_file(path: str | Path) -> bytes:
    """Read a profile PNG an operator saved from the portal.

    The supported acquisition path today, and not a lesser one: extraction, the
    ЕМБС self-check and citation behave identically whether the bytes arrived
    over the wire or off the disk, so nothing downstream knows which happened.
    """
    return read_png(path)


class ProfileSource(Protocol):
    """Where an EntityProfile comes from. The seam the transport swaps at.

    Note the return type: FIELDS, not bytes. That is the whole point. An
    official JSON feed would have no image to hand back and no vision step to
    run, so a byte-shaped seam would force the future transport to fake one.
    Sources own whatever they need to produce the fields -- a file and a vision
    model, a session and a renderer, an API key and a parser -- and everything
    downstream of `load()` is identical for all of them.

    Returning None means "this source has nothing for that ЕМБС", which is an
    ordinary outcome. Raising means the source itself is broken.
    """
    name: str

    def load(self, embs: str) -> EntityProfile | None: ...


class LocalImageSource:
    """Profiles from PNGs an operator saved out of the portal.

    The working transport while the open-data request is pending. Nothing in
    it touches the network, so it cannot be tricked into an unauthorised call
    by a malformed ЕМБС or a missing file.
    """
    name = "vision:file"

    def __init__(self, directory: Path | str = ASSETS, *, client=None,
                 pattern: str = "basic_profile_{embs}.png"):
        self.directory = Path(directory)
        self.client = client
        self.pattern = pattern
        self._cache = ImageCache()      # see vision.ImageCache

    def candidates(self, embs: str) -> list[Path]:
        """Filenames that could hold this entity, padding included.

        The registry prints 7696876 and accepts 07696876 for the same company,
        so an operator saving from the portal and a user typing from a document
        will disagree about the leading zero roughly half the time. Resolving
        both beats making the human guess which spelling the cache wants.
        """
        bare = _digits(embs).lstrip("0")
        forms = dict.fromkeys([embs, bare, bare.zfill(8), bare.zfill(7)])
        return [self.directory / self.pattern.format(embs=f) for f in forms]

    def load(self, embs: str) -> EntityProfile | None:
        path = next((p for p in self.candidates(embs) if p.is_file()), None)
        if path is None:
            return None
        # A lambda, not a bound reference: extract_profile is looked up when
        # the cache misses, so tests that patch the module attribute still
        # reach the path they are testing.
        read = lambda data: extract_profile(data, client=self.client)  # noqa: E731
        profile = self._cache.get(path, read)
        # One re-read when the image disagrees with the filename about which
        # entity this is. See LocalDecisionSource.load: vision transposes a
        # digit occasionally, and a cached bad read would look permanent.
        if not _same_embs(profile.embs, embs):
            profile = self._cache.refresh(path, read)
        return profile


class PortalImageSource:
    """Profiles from a live render, for a caller who holds session material.

    Dormant: constructing one requires a PortalSession, and this project does
    not produce those (see the module docstring). It exists so the live path
    stays exercised by the type system rather than rotting in a comment, and so
    an official API source can be written against a seam that already has two
    implementations instead of one.
    """
    name = "vision:live"

    def __init__(self, session: PortalSession, *, client=None, http=None):
        self.session = session
        self.client = client
        self.http = http

    def load(self, embs: str) -> EntityProfile | None:
        image = fetch_profile_image(embs, self.session, client=self.http)
        return extract_profile(image, client=self.client)


_DEFAULT_SOURCE: ProfileSource | None = None


def default_source() -> ProfileSource:
    """The source the registered tool uses.

    Held as a singleton rather than built per call, because LocalImageSource
    owns the extraction cache -- a fresh instance each time is a cache that
    never hits, which is a vision call and three seconds per repeated question.
    """
    global _DEFAULT_SOURCE
    if _DEFAULT_SOURCE is None:
        _DEFAULT_SOURCE = LocalImageSource()
    return _DEFAULT_SOURCE


def set_default_source(source: ProfileSource | None) -> None:
    """Swap the transport. THE migration point.

    When the open-data feed arrives, an OpenDataSource implementing `load()`
    goes in here and nothing downstream changes: not fetch_profile, not the
    validation, not as_context_doc, not the citation ids, not the tool spec.
    Passing None restores the offline default.
    """
    global _DEFAULT_SOURCE
    _DEFAULT_SOURCE = source


def fetch_profile(embs: str, *, source: ProfileSource | None = None,
                  image_path: str | Path | None = None,
                  client=None) -> ProfileResult | ProfileUnavailable:
    """Look up one entity's basic profile. Verified fields, or a clean miss.

    Returns ProfileUnavailable rather than raising when the source simply has
    nothing: the caller is a tool loop reporting to a user, and "we do not hold
    that one offline" is an answer. It still RAISES when the data it did get
    fails its checks -- see below.
    """
    args = ProfileArgs(embs=EntitySizeArgs(embs=embs).embs)   # same ЕМБС guard
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if image_path is not None:
        src: ProfileSource = LocalImageSource(
            Path(image_path).parent, client=client,
            pattern=Path(image_path).name)
    else:
        src = source if source is not None else default_source()

    profile = src.load(args.embs)
    if profile is None:
        return ProfileUnavailable(
            embs=args.embs, reason="no_local_image",
            message=("Основниот профил за овој субјект не е достапен офлајн. "
                     "Податоците од Централниот регистар за оваа услуга се "
                     "добиваат како слика од порталот, а зачувана слика за "
                     "овој ЕМБС нема. Профилот може да се преземе рачно од "
                     "порталот на Централниот регистар."))

    # Source-INDEPENDENT validation. A JSON feed can transpose a digit as
    # easily as a vision model can misread one, so these run wherever the
    # fields came from rather than being bolted onto the vision path.
    #
    # Both raise. A wrong-entity profile or a malformed field means the source
    # is broken -- the wrong PNG filed under this ЕМБС, or an API contract that
    # moved -- and _run_tool's own rule is that a dead integration raises
    # instead of letting the model narrate around it. Only "nothing here"
    # comes back as a value.
    if not _same_embs(profile.embs, args.embs):
        raise ProfileMismatch(
            f"asked for ЕМБС {args.embs}, {src.name} returned "
            f"{profile.embs!r} -- refusing to return another entity's data")

    problems = verify_profile(profile, args.embs)
    if problems:
        raise ValueError("profile failed shape checks: " + "; ".join(problems))
    return ProfileResult(profile=profile, fetched_at=fetched_at,
                         source=src.name, requested_embs=args.embs)


TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "get_entity_profile",
        "description": (
            "Го враќа основниот профил на конкретен регистриран субјект од "
            "Централниот регистар по ЕМБС: ЕДБ, скратен назив, датум на "
            "основање, правна форма, правен статус, адреса, претежна дејност и "
            "големина. Користи го кога корисникот бара податоци за ОДРЕДЕН "
            "субјект и го дал неговиот ЕМБС. Ако дал само НАЗИВ, прво повикај "
            "search_entity_profile за да го добиеш ЕМБС, па потоа овој алат; "
            "НЕ измислувај ЕМБС. НЕ го користи за општи прашања за "
            "постапки, документи, рокови или тарифи -- тие се одговараат од "
            "документацијата. Ако е потребна САМО големината, користи "
            "check_entity_size, кој е побрз и поевтин. Ако одговорот има "
            "available=false, профилот за тој ЕМБС не е достапен -- пренеси ја "
            "пораката на корисникот и НЕ го повикувај алатот повторно за истиот "
            "ЕМБС."),
        "parameters": strict_schema(ProfileArgs),
        "strict": True,
    },
}
