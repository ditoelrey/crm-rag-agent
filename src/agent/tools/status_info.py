"""
status_info.py  --  Form 4: Статус на предмет, by деловоден број.
=================================================================
What became of one filing: when it was published, what kind of entry it was,
and whether it was decided.

WHY THIS IS OFFLINE TOO
-----------------------
The capture for this endpoint carried no reCAPTCHA header, which read as an
unprotected route. It is not. One request, made once to find out:

    GET /CRMPublicPortalApi/api/freeservice/statusInfo/30120260014978?idL=1
    -> HTTP 412  {"message":"Recaptcha token missing"}

So the same rule as Forms 2 and 3 applies: this module does not obtain a token.
`fetch_status_live` exists, takes a PortalSession the caller must already hold,
and is never called from the registered tool. What runs is the offline source.

NO IMAGE, NO VISION
-------------------
Unlike Forms 2 and 3 this view is text -- either the JSON the API returns or the
`infobox` the page renders from it -- so the offline source reads operator-saved
files directly and there is no probabilistic step anywhere in this form. That is
worth stating plainly: the status the agent reports is exactly the bytes someone
saved, not a model's reading of a picture.

THE IDENTIFIER, AGAIN
---------------------
Keyed by деловоден број, like Form 3, so the same routing applies: a user with
only an ЕМБС or a name goes through search_announcements first, and a user with
neither is asked rather than guessed at.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field, field_validator

from eval.corpus import Block

from ..context import ContextDoc
from .entity_profile import PortalSession
from .registration_decision import ASSETS, DELOVODEN_DIGITS
from .schema import strict_schema

# --- THE LIVE CALL, recorded and refused ----------------------------------- #
STATUS_URL = ("https://www.crm.com.mk/CRMPublicPortalApi/api/freeservice/"
              "statusInfo/{document_id}")
PAGE_URL = "https://www.crm.com.mk/mk/otvoreni-podatotsi/"
LANGUAGE_ID = 1          # idL=1 -- Macedonian, per the capture
# --------------------------------------------------------------------------- #

STATUS_PATTERN = "status_{document_id}.json"


class StatusArgs(BaseModel):
    document_id: str = Field(description=(
        "Деловоден број на предметот exactly as the user gave it -- 14 digits. "
        "This is NOT the ЕМБС (7-8 digits)."))

    @field_validator("document_id")
    @classmethod
    def _fourteen_digits(cls, v: str) -> str:
        v = v.strip()
        if not re.fullmatch(rf"\d{{{DELOVODEN_DIGITS}}}", v):
            raise ValueError(
                f"Деловодниот број мора да има {DELOVODEN_DIGITS} цифри. "
                f"ЕМБС не е деловоден број -- прво пребарај ја објавата.")
        return v


class StatusInfo(BaseModel):
    """The four fields the portal's infobox shows, verbatim."""
    entity_title: str = Field(description="Назив на субјектот, as printed")
    publication_date: str = Field(description="Датум на објава, as printed")
    document_description: str = Field(description="Опис на документ, as printed")
    status: str = Field(description="Статус, as printed")


class StatusResult(BaseModel):
    document_id: str
    info: StatusInfo
    fetched_at: str
    source: str = "offline:assets"
    available: bool = True

    def as_context_doc(self, rank: int = 0) -> ContextDoc:
        chunk_id = f"tool:status_info:{self.document_id}"
        i = self.info
        text = "\n".join(filter(None, [
            f"Деловоден број: {self.document_id}",
            f"Субјект: {i.entity_title}" if i.entity_title else "",
            f"Датум на објава: {i.publication_date}" if i.publication_date else "",
            f"Опис на документ: {i.document_description}"
            if i.document_description else "",
            f"Статус: {i.status}" if i.status else "",
        ]))
        block = Block(
            chunk_id=chunk_id,
            context=(f"Статус на предмет (деловоден број {self.document_id}), "
                     f"прочитан {self.fetched_at}"),
            content=text,
            embedding_text=text,
            scope="service",
            type="live_lookup",
            id_service=None,
            service_name="Статус на предмет",
        )
        return ContextDoc(chunk_id=chunk_id, block=block, score=1.0, rank=rank)


class StatusUnavailable(BaseModel):
    """No saved status for this filing. An answer, not an error -- see
    entity_profile.ProfileUnavailable for why this has to be a value."""
    document_id: str
    available: bool = False
    reason: str
    message: str

    def as_context_doc(self, rank: int = 0) -> ContextDoc:
        chunk_id = f"tool:status_info:{self.document_id}:unavailable"
        text = f"Деловоден број {self.document_id}: {self.message}"
        block = Block(
            chunk_id=chunk_id,
            context=(f"Обид за читање статус на предмет "
                     f"(деловоден број {self.document_id})"),
            content=text,
            embedding_text=text,
            scope="service",
            type="live_lookup",
            id_service=None,
            service_name="Статус на предмет",
        )
        return ContextDoc(chunk_id=chunk_id, block=block, score=1.0, rank=rank)


class StatusRefused(RuntimeError):
    """The registry declined the request -- distinct from "no such filing"."""


_LABELS = {
    "датум на објава": "publication_date",
    "опис на документ": "document_description",
    "статус": "status",
}


def parse_infobox(html: str) -> StatusInfo:
    """Read the rendered infobox. Deterministic -- no model involved.

    The page puts each field in a <p> as "Label: <b>value</b>", with the
    entity's name in the <h2> above them. Labels are matched, not positions, so
    a reordered or extended box still reads correctly and an unknown label is
    ignored rather than shifting every field by one.
    """
    from bs4 import BeautifulSoup

    box = BeautifulSoup(html, "html.parser")
    root = box.find(class_="infobox") or box
    heading = root.find("h2")
    values = {"entity_title": heading.get_text(" ", strip=True) if heading else ""}
    for para in root.find_all("p"):
        bold = para.find("b")
        if bold is None:
            continue
        label = para.get_text(" ", strip=True).split(":")[0]
        field = _LABELS.get(" ".join(label.casefold().split()))
        if field:
            values[field] = " ".join(bold.get_text(" ", strip=True).split())
    return StatusInfo(**{"publication_date": "", "document_description": "",
                         "status": "", **values})


def parse_status(payload: str) -> StatusInfo:
    """Read whatever the saved file holds: the API's JSON or the page's HTML."""
    text = payload.strip()
    if text.startswith("{"):
        data = json.loads(text)
        # Accept both our own field names and the portal's, so a file saved
        # straight from the API needs no hand-editing.
        return StatusInfo(
            entity_title=str(data.get("entity_title")
                             or data.get("title") or ""),
            publication_date=str(data.get("publication_date")
                                 or data.get("publishDate") or ""),
            document_description=str(data.get("document_description")
                                     or data.get("documentDescription") or ""),
            status=str(data.get("status") or ""))
    return parse_infobox(text)


class StatusSource(Protocol):
    """Where a status comes from. None means "not held"."""
    name: str

    def load(self, document_id: str) -> StatusInfo | None: ...


class LocalStatusSource:
    """Statuses an operator saved next to the images, as JSON or saved HTML."""
    name = "offline:assets"

    def __init__(self, directory: Path | str = ASSETS):
        self.directory = Path(directory)

    def candidates(self, document_id: str) -> list[Path]:
        stem = STATUS_PATTERN.format(document_id=document_id).removesuffix(".json")
        return [self.directory / f"{stem}.json", self.directory / f"{stem}.html"]

    def load(self, document_id: str) -> StatusInfo | None:
        path = next((p for p in self.candidates(document_id) if p.is_file()), None)
        if path is None:
            return None
        return parse_status(path.read_text(encoding="utf-8"))


_DEFAULT_SOURCE: StatusSource | None = None


def default_source() -> StatusSource:
    global _DEFAULT_SOURCE
    if _DEFAULT_SOURCE is None:
        _DEFAULT_SOURCE = LocalStatusSource()
    return _DEFAULT_SOURCE


def set_default_source(source: StatusSource | None) -> None:
    global _DEFAULT_SOURCE
    _DEFAULT_SOURCE = source


def fetch_status_live(document_id: str, session: PortalSession, *,
                      client=None) -> StatusInfo:
    """The live GET. Requires session material the caller already holds.

    Never called by the registered tool: the endpoint answers 412 "Recaptcha
    token missing" without one, and obtaining that token is out of scope. Kept
    so the transport is written down, and so it fails loudly rather than
    silently returning the refusal as data.
    """
    import httpx

    from .crm_http import HEADERS, TIMEOUT_S

    own = client is None
    client = client or httpx.Client(timeout=TIMEOUT_S)
    try:
        resp = client.get(
            STATUS_URL.format(document_id=document_id),
            params={"idL": LANGUAGE_ID},
            cookies=session.cookies,
            headers={"Accept": "application/json, text/plain, */*",
                     "Origin": "https://www.crm.com.mk",
                     "Referer": PAGE_URL,
                     "recaptcha": session.recaptcha,
                     "User-Agent": HEADERS["User-Agent"]})
        if resp.status_code == 412:
            raise StatusRefused(
                f"the registry refused the request: {resp.text[:200]}")
        resp.raise_for_status()
        return parse_status(resp.text)
    finally:
        if own:
            client.close()


def fetch_status(document_id: str, *,
                 source: StatusSource | None = None) -> StatusResult | StatusUnavailable:
    """Look up one filing's status. Verified fields, or a clean miss."""
    args = StatusArgs(document_id=document_id)
    src = source if source is not None else default_source()
    info = src.load(args.document_id)
    if info is None:
        return StatusUnavailable(
            document_id=args.document_id, reason="no_local_status",
            message=("Статусот на овој предмет не е достапен офлајн. Статусите "
                     "од Централниот регистар се читаат од порталот, а зачуван "
                     "статус за овој деловоден број нема. Проверете на "
                     "порталот на Централниот регистар."))
    return StatusResult(
        document_id=args.document_id, info=info, source=src.name,
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))


TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "get_status_info",
        "description": (
            "Го враќа статусот на конкретен предмет (пријава) во Централниот "
            "регистар по ДЕЛОВОДЕН БРОЈ (14 цифри): назив на субјектот, датум "
            "на објава, опис на документот и статус, на пр. „Одлучен - "
            "одобрен“. Користи го САМО со деловоден број. Ако корисникот дал "
            "ЕМБС или назив, прво повикај search_announcements за да го добиеш "
            "деловодниот број. Ако корисникот не дал ниту број, ниту ЕМБС, ниту "
            "назив, побарај го деловодниот број од него -- НИКОГАШ не "
            "измислувај број. Ако одговорот има available=false, статусот не е "
            "достапен: пренеси ја пораката и не повикувај повторно."),
        "parameters": strict_schema(StatusArgs),
        "strict": True,
    },
}
