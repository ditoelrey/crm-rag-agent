"""
matching.py  --  comparing what a user typed with what the registry printed.
============================================================================
Both search tools have to decide whether a typed criterion matches a value read
off a rendered page, and they must decide it the same way: a name found by
search_entity_profile and the same name missed by search_announcements would be
a bug nobody could reproduce.

Three mismatches recur, all of them the registry's own doing:

  * ЕМБС padding -- the size form takes 07696876, the profile image prints
    7696876, and they are one company;
  * dates -- a decision prints "17 септ. 2026" where the user types 17.09.2026;
  * case and spacing in names, which wrap across lines in the rendered tables.
"""
from __future__ import annotations

import re

MONTHS = ("јануари", "февруари", "март", "април", "мај", "јуни",
          "јули", "август", "септември", "октомври", "ноември", "декември")


def digits(text: str) -> str:
    return re.sub(r"\D", "", text or "")


def norm_text(text: str) -> str:
    """Casefolded, whitespace-collapsed -- for substring matching on names."""
    return " ".join((text or "").casefold().split())


def same_embs(a: str, b: str) -> bool:
    """One company, whatever the padding. Padding is ignored for the COMPARISON
    only; neither value is rewritten."""
    return digits(a).lstrip("0") == digits(b).lstrip("0")


def as_date(value: str) -> str | None:
    """dd.mm.yyyy for a printed date, or None if the text is not one.

    The month is matched by prefix in both directions, so the registry's
    "септ.", a bare "сеп" and a full "септември" all resolve to 09.
    """
    value = norm_text(value)
    numeric = re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.(\d{4})\.?", value)
    if numeric:
        day, month, year = (int(x) for x in numeric.groups())
        return f"{day:02d}.{month:02d}.{year}"
    worded = re.fullmatch(r"(\d{1,2})\s+([^\s\d]+)\.?\s+(\d{4})", value)
    if not worded:
        return None
    day, month_text, year = worded.groups()
    stem = month_text.rstrip(".")
    for index, name in enumerate(MONTHS, start=1):
        if name.startswith(stem) or stem.startswith(name):
            return f"{int(day):02d}.{index:02d}.{year}"
    return None
