"""
intent.py  --  which section of a service is the question about?
================================================================
Shared by the retrieval layer (structured section fetch) and the agent
(variation-ambiguity detection), so both agree on what a question is asking for.

Cues are matched with a LEFT word boundary only, so a prefix still catches
Macedonian inflection (рок / рокот / рокови) while "чин" cannot fire inside
"начин" -- a real bug the selftest caught.

An empty result means "unclear", and every caller treats that as a reason to do
nothing rather than to guess: injecting the wrong section into a legal answer is
worse than injecting none.
"""
from __future__ import annotations

import re
import unicodedata

INTENT_CUES: dict[str, tuple[str, ...]] = {
    "documentsLocations": ("подигн", "преземањ", "презем", "подига"),
    "forms": ("образец", "обрасц", "формулар"),
    # "колку се плаќа" is a compound cue on purpose. Macedonian splits payment
    # across two stems -- плати/платам and плаќа/плаќање -- and the SECTION
    # depends on the interrogative, not the verb: "колку се плаќа" wants the
    # tariff, "како се плаќа" wants the payment methods in `access`. Live, a
    # question that was entirely about payment matched neither cue list ("плаќа"
    # falls between "плати" and "плаќањ") and routed to `process` on the
    # strength of "како" alone.
    "tariffs": ("чин", "цена", "цени", "тариф", "надомест", "кошта", "плати",
                "чинат", "пари", "колку се плаќа", "колку плаќ"),
    "deadlines": ("рок", "колку време", "трае", "траење"),
    # "поднесе" was here to catch "Што треба да поднесам за X?" and fired on the
    # ordinary passive "да се поднесе" instead -- it turned "Дали годишна сметка
    # може да се поднесе преку интернет?" into a documents question and made the
    # agent ask which legal form, for a service whose five variants share one
    # identical link. "што треба" already covers the intended phrasing.
    "documents": ("документ", "потребн", "прилож", "доказ", "што треба"),
    # "шалтер" is deliberately absent. It names a delivery CHANNEL and almost
    # always appears as a modifier ("преку шалтер", "на шалтер"), so it beat the
    # subject word in "кои се ЧЕКОРИТЕ ... преку шалтер" and routed a procedure
    # question to `access`. Pickup questions still reach documentsLocations via
    # "подигн"; genuine access questions via онлајн / интернет / електронск.
    "access": ("онлајн", "интернет", "електронск", "плаќ", "плат"),
    # "пријав" is deliberately absent: it is a NOUN in dozens of service names
    # ("Самостојна пријава за упис на основање"), so it fired on questions that
    # merely named a service and injected its procedure over the description.
    "process": ("како", "постапк", "чекор", "начин", "кога", "каде"),
}

# Dict order above IS the precedence, most specific first. Questions routinely
# trip several cues at once -- "Каде и како го подигнувам документот за X?" hits
# documents ("документ"), process ("како") and access ("каде") -- and injecting
# all three floods the answer with the wrong sections. The vague interrogatives
# (како / каде / кога) sit last on purpose: they appear in almost every
# question and are the weakest evidence of what is actually being asked.



def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def detect_intent(query: str, *, top_only: bool = False) -> list[str]:
    """Section type(s) the question is about; empty when unclear.

    `top_only` keeps just the most specific match -- what structured fetch uses,
    so a question cannot pull in three sections at once. Ambiguity detection
    wants the full list: any section that differs between legal forms is a
    reason to ask.
    """
    q = _norm(query)
    found = [type_ for type_, cues in INTENT_CUES.items()
             if any(re.search(rf"(?<!\w){re.escape(cue)}", q) for cue in cues)]
    return found[:1] if top_only else found
