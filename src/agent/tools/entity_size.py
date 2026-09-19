"""
entity_size.py  --  Form 1: ПРОВЕРИ ГОЛЕМИНА НА СУБЈЕКТ (live lookup).
=====================================================================
The first live tool. Three layers, and the model only reaches the top one:

    fetch_entity_size()   pure I/O + deterministic parsing, no LLM
    EntitySize            a validated result object
    TOOL_SPEC             the only thing gpt-4o ever sees

Raw HTML never reaches the model. It chooses WHICH tool and WITH WHAT argument;
everything after that is ordinary Python, which is the same division this
codebase already draws between the retrieval layer and the answer layer.

Citations
---------
`EntitySize.as_context_doc()` is not a convenience. validate_citations() decides
what may be cited from the chunk_ids present in the context, and the hard
abstention guard REPLACES any answer that carries no valid citation. A tool
result that is not a ContextDoc is therefore unusable however correct it is --
the agent's own safety net deletes it. Giving the result a synthetic id keeps
every existing check working unmodified, and the `tool:` prefix keeps live data
visibly distinct from a corpus block wherever a chunk_id is displayed.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator

from eval.corpus import Block

from ..context import ContextDoc
from .crm_http import post_webform
from .schema import strict_schema

URL = "https://e-submit.crm.com.mk/AAOL/pCheckLeSize.aspx"
FIELD_EMBS = "ctl00$cphMain$ucCheckLeSize$TbxLEID"
FIELD_SUBMIT = "ctl00$cphMain$ucCheckLeSize$BtnSearch"
SUBMIT_VALUE = "ПРОВЕРИ"
RESULT_SPAN_ID = "ctl00_cphMain_ucCheckLeSize_LblResult"

# The registry's four size classes, longest first so "мал" cannot match inside
# a hypothetical longer token before the right one is tried.
#
# Used to VALIDATE, never to coerce. An unrecognised message is returned
# verbatim with size=None: a wrong size is a reportable error for the user,
# an unknown one is merely unhelpful, and the two must not be traded.
KNOWN_SIZES = ("среден", "микро", "голем", "мал")


class EntitySizeArgs(BaseModel):
    """What the model must supply, validated BEFORE any network call.

    The guard matters: without it an invented or malformed ЕМБС reaches a
    government host, and the portal answers a malformed id the same way it
    answers an unknown one.

    No example number in the description. The one that used to be here was a
    real company's ЕМБС, and a model that copies an example instead of asking
    the user would look up that company for someone who never mentioned it.
    """
    embs: str = Field(description=(
        "ЕМБС (Единствен матичен број на субјектот) -- the 7- or 8-digit "
        "registration number of the legal entity, digits only, exactly as the "
        "user gave it. Never add or remove a leading zero."))

    @field_validator("embs")
    @classmethod
    def _seven_or_eight_digits(cls, v: str) -> str:
        v = v.strip()
        if not re.fullmatch(r"\d{7,8}", v):
            raise ValueError("ЕМБС мора да биде 7 или 8 цифри")
        return v          # never zero-stripped: that would change the identifier


class EntitySize(BaseModel):
    embs: str
    size: str | None            # микро | мал | среден | голем, else None
    message: str                # the registry's own sentence, whitespace-collapsed
    fetched_at: str             # UTC ISO-8601; live data is a point-in-time claim
    found: bool

    def as_context_doc(self, rank: int = 0) -> ContextDoc:
        """Make the live result citable by the answer layer. See module docstring."""
        chunk_id = f"tool:entity_size:{self.embs}"
        # The ЕМБС belongs in the CONTENT, not only in the context line. Any
        # sane answer repeats the number it was asked about, and
        # numeric_groundedness checks every multi-digit number in the answer
        # against the text of the cited blocks -- so leaving it out of the
        # content marks a correctly-quoted ЕМБС as a fabrication.
        text = (f"ЕМБС {self.embs}: {self.message}" if self.message
                else f"Нема резултат за ЕМБС {self.embs}.")
        block = Block(
            chunk_id=chunk_id,
            # The timestamp rides in `context` so the model can date the claim.
            # A corpus row is stable; this one was true at the moment we asked.
            context=(f"Проверка во живо на Централен регистар "
                     f"(ЕМБС {self.embs}), извршена {self.fetched_at}"),
            content=text,
            embedding_text=text,
            scope="service",
            # Deliberately not one of the corpus section types: it must not be
            # picked up by the coverage block or the ambiguity detectors, which
            # reason about sections of a service. id_service/id_variation stay
            # None for the same reason.
            type="live_lookup",
            id_service=None,
            service_name="Провери големина на субјект",
        )
        return ContextDoc(chunk_id=chunk_id, block=block, score=1.0, rank=rank)


def parse_entity_size(page, embs: str, fetched_at: str) -> EntitySize:
    """Pull the result out of a rendered page. Split out so it can be tested
    against a saved fixture without touching the network."""
    span = page.find(id=RESULT_SPAN_ID)
    if span is None:
        # NOT reported as "not found". A rejected handshake or moved markup and
        # a genuinely unknown ЕМБС must never look the same to the caller --
        # one is our bug, the other is an answer.
        raise RuntimeError(
            f"result span {RESULT_SPAN_ID!r} missing -- the WebForms POST was "
            f"rejected or the page changed; re-check the field names in "
            f"{__name__}")

    message = " ".join(span.get_text(" ", strip=True).split())
    size = next((s for s in KNOWN_SIZES
                 if re.search(rf"(?<!\w){s}(?!\w)", message.lower())), None)
    return EntitySize(embs=embs, size=size, message=message,
                      fetched_at=fetched_at, found=size is not None)


def fetch_entity_size(embs: str) -> EntitySize:
    """Look up one entity's size class. Deterministic: no model involved."""
    args = EntitySizeArgs(embs=embs)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    page = post_webform(URL, {FIELD_EMBS: args.embs, FIELD_SUBMIT: SUBMIT_VALUE})
    return parse_entity_size(page, args.embs, fetched_at)


# What gpt-4o sees. The description is restrictive on purpose: the corpus ALSO
# explains what entity size means, and the failure to guard against is the model
# reaching for a live lookup to answer "што значи големина на субјект", which is
# a documentation question with no ЕМБС in it.
TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "check_entity_size",
        "description": (
            "Проверува ја големината на конкретен регистриран субјект (микро, "
            "мал, среден или голем) во живо од Централниот регистар, по ЕМБС. "
            "Користи го САМО кога корисникот прашува за големината на ОДРЕДЕН "
            "субјект и го дал неговиот ЕМБС. НЕ го користи за општи прашања за "
            "тоа што значи големина, за постапки, документи, рокови или тарифи "
            "-- тие се одговараат од документацијата."),
        "parameters": strict_schema(EntitySizeArgs),
        "strict": True,
    },
}
