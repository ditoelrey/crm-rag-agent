"""
fixtures.py  --  recorded tool responses, so the gates never need the portal.
=============================================================================
Every eval and selftest run would otherwise POST to a government host: slow,
impolite, and dependent on that host being up. A gate that can fail because
someone else's server is down is a gate people learn to ignore, and a gate
people ignore is worse than no gate.

These are real responses, recorded once and asserted against the same parser the
live path uses -- so a corpus of fixtures cannot drift into fiction while the
parser changes underneath it.

Pass `--live-tools` to the answer eval to hit the real portal instead.
"""
from __future__ import annotations

from .announcement_search import search_announcements
from .entity_search import EntityHit, search_entity_profile
from .status_info import StatusInfo, fetch_status
from .entity_profile import (EntityProfile, ProfileResult, ProfileUnavailable,
                             fetch_profile)
from .entity_size import EntitySize
from .registration_decision import (DecisionResult, DecisionRow, DecisionSection,
                                    DecisionUnavailable, RegistrationDecision,
                                    fetch_decision)

# Recorded 2026-09-18 from https://e-submit.crm.com.mk/AAOL/pCheckLeSize.aspx
# ЛОРА КОМПАНИ 2023 ДОО Скопје -- the entity used in the Form 1 bring-up.
RECORDED = {
    "07696876": EntitySize(
        embs="07696876", size="мал",
        message="Големината на правниот субјект е: мал",
        fetched_at="2026-09-18T00:00:14+00:00", found=True),
    "7405855": EntitySize(
        embs="7405855", size="микро",
        message="Големината на правниот субјект е: микро",
        fetched_at="2026-09-18T00:00:14+00:00", found=True),
}

_UNKNOWN = "Не е пронајден субјект со внесениот ЕМБС"


def fixture_entity_size(embs: str) -> EntitySize:
    """Replay a recorded lookup. Unrecorded ids answer as the portal does for an
    entity it does not hold -- not as an error, so the not-found path is
    exercised too."""
    from .entity_size import EntitySizeArgs

    embs = EntitySizeArgs(embs=embs).embs      # same validation as the live path
    if embs in RECORDED:
        return RECORDED[embs]
    return EntitySize(embs=embs, size=None, message=_UNKNOWN,
                      fetched_at="2026-09-18T00:00:14+00:00", found=False)


# Form 2, recorded from agent/tools/assets/basic_profile_7696876.png -- the
# verbatim gpt-4o read, checked field by field against the picture. Kept here so
# the gates exercise the profile path without spending a vision call per run,
# and so a drift in the extraction prompt shows up as a diff against a read a
# human actually verified rather than against whatever the model says today.
RECORDED_PROFILES = {
    "7696876": EntityProfile(
        full_name="Друштво за производство, трговија и услуги ЛОРА КОМПАНИ "
                  "2023 ДОО Скопје",
        embs="7696876", edb="4058023546097",
        short_name="ЛОРА КОМПАНИ 2023 ДОО Скопје", founded="19.09.2023",
        legal_form="05.3 - друштво со ограничена одговорност",
        legal_status="Активен",
        address="СВ.НАУМ ОХРИДСКИ бр.35А СКОПЈЕ - КИСЕЛА ВОДА, КИСЕЛА ВОДА",
        additional_info="",
        activity="56.400 - Дејности за посредување при подготовка и "
                 "послужување храна и пијалаци",
        size="мал"),
}


class _FixtureSource:
    """A ProfileSource backed by the recordings above. No file, no model.

    Implementing the same seam as LocalImageSource rather than monkeypatching
    fetch_profile keeps the fixtures honest: the ЕМБС self-check and the shape
    checks run over recorded fields exactly as they do over live ones, so a
    recording that could not have survived validation cannot sit in this file
    pretending otherwise.
    """
    name = "fixture"

    def load(self, embs: str):
        bare = embs.lstrip("0")
        return RECORDED_PROFILES.get(bare)


def fixture_profile(embs: str) -> ProfileResult | ProfileUnavailable:
    """Replay a recorded profile. Unrecorded ids come back unavailable, which
    is what the live offline source does for an entity nobody has saved."""
    return fetch_profile(embs, source=_FixtureSource())


# Form 3, recorded from agent/tools/assets/decision_30120260014967.png -- two
# independent gpt-4o reads, identical, checked row by row against the image.
# Tables only: the preamble paragraph is deliberately not read (see
# registration_decision's module docstring for the three misreads it produced).
#
# Note the entity: ЕМБС 7405855 is the same company Form 1 records as микро, and
# this decision independently prints микро. Two tools, one fact, no shared code
# path between them.
def _rows(*pairs: tuple[str, str]) -> list[DecisionRow]:
    return [DecisionRow(label=label, value=value) for label, value in pairs]


_TOPOLCHAN = ("Друштво за производство, трговија и услуги ТОПОЛЧАН АГРАР ДООЕЛ "
              "с.Тополчани Прилеп - во ликвидација")
_ENTRY = "Документ за определување на главна приходна шифра и големина"

RECORDED_DECISIONS = {
    "30120260014967": RegistrationDecision(
        deloveden_broj="30120260014967", entry_type=_ENTRY, embs="7405855",
        full_name=_TOPOLCHAN,
        sections=[
            DecisionSection(title="Деловодник", rows=_rows(
                ("Прием на пријавата", "17 септ. 2026"),
                ("Вид на упис", _ENTRY),
                ("Деловоден број", "30120260014967"),
                ("Начин на доставување", "По службена должност"),
                ("Одобрување на пријавата", "17 септ. 2026"))),
            DecisionSection(title="Основни податоци за субјектот на упис",
                            rows=_rows(
                ("ЕМБС", "7405855"),
                ("Целосен назив", _TOPOLCHAN),
                ("Големина на субјектот", "микро"))),
            DecisionSection(title="Дејности", rows=_rows(
                ("Приоритетна дејност / Главна приходна шифра",
                 "01.250 - Одгледување јагодесто, јатчесто и друго овошје"))),
        ]),
}


class _FixtureDecisionSource:
    """A DecisionSource backed by the recording above -- same seam, so the
    self-check and consistency checks run over it exactly as over a live read."""
    name = "fixture"

    def load(self, deloveden_broj: str):
        return RECORDED_DECISIONS.get(deloveden_broj)


def fixture_decision(deloveden_broj: str) -> DecisionResult | DecisionUnavailable:
    return fetch_decision(deloveden_broj, source=_FixtureDecisionSource())


# Form 4. Operator-supplied: what the portal's infobox showed for this filing,
# captured by hand (the entity name was supplied separately, after the first
# capture redacted it).
#
# The same four values are saved as assets/status_30120260014978.json, which is
# what the PRODUCTION tool reads -- the gates read this dict instead, so they
# never depend on what an operator has or has not saved. Two copies means two
# places to drift, so the selftest asserts they are identical.
RECORDED_STATUS = {
    "30120260014978": StatusInfo(
        entity_title=("Трговец поединец за поставување на подни и ѕидни облоги "
                      "ДРАГИ АЛЕКСА ЈОВАНОВ-БЕКАТОН ЕЛИТ ТП Пробиштип"),
        publication_date="18.9.2026 18:45",
        document_description="Упис на основање", status="Одлучен - одобрен"),
}


class _FixtureStatusSource:
    name = "fixture"

    def load(self, document_id: str):
        return RECORDED_STATUS.get(document_id)


def fixture_status(**kw):
    return fetch_status(**kw, source=_FixtureStatusSource())


class _FixtureEntityIndex:
    """Entities the gates know about: the recorded profile, plus the entity the
    recorded decision names. The second is the case that matters -- ТОПОЛЧАН
    АГРАР is findable by name with no profile saved, so `has_profile` is False
    and the agent is expected to say so rather than fetch."""
    name = "fixture"

    def entities(self):
        hits = {}
        for embs, profile in RECORDED_PROFILES.items():
            hits[embs.lstrip("0")] = EntityHit(
                embs=profile.embs, full_name=profile.full_name,
                has_profile=True, seen_in=["профил"])
        for decision in RECORDED_DECISIONS.values():
            key = decision.embs.lstrip("0")
            if key in hits:
                hits[key].seen_in.append("решение")
            else:
                hits[key] = EntityHit(embs=decision.embs,
                                      full_name=decision.full_name,
                                      has_profile=False, seen_in=["решение"])
        return list(hits.values())


def fixture_entity_search(**criteria):
    return search_entity_profile(**criteria, index=_FixtureEntityIndex())


class _FixtureIndex:
    """The searchable archive during gates: the recordings above, nothing else.

    Deliberately the same content the fixture decision source serves, so a
    search hit is always fetchable in the same run -- the property the live
    offline index has by construction, kept true here instead of assumed.
    """
    name = "fixture"

    def all_decisions(self):
        return list(RECORDED_DECISIONS.values())


def fixture_search(**criteria):
    return search_announcements(**criteria, index=_FixtureIndex())


DISPATCH = {
    "check_entity_size": lambda **kw: fixture_entity_size(**kw),
    "get_entity_profile": lambda **kw: fixture_profile(**kw),
    "get_registration_decision": lambda **kw: fixture_decision(**kw),
    "search_announcements": lambda **kw: fixture_search(**kw),
    "search_entity_profile": lambda **kw: fixture_entity_search(**kw),
    "get_status_info": lambda **kw: fixture_status(**kw),
}
