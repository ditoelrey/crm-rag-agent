"""
selftest.py  --  offline checks for everything in the agent except the model call.
==================================================================================
Dedup, ambiguity detection, citation validation, clarification matching and the
full multi-turn orchestration are all deterministic, so they are tested with a
stub retriever and a stub OpenAI client: no API key, no network, no cost.

What this cannot check is answer quality -- that needs the model, and belongs in
the answer-eval layer that will extend src/eval.

Run: python -m agent.selftest        (from src/)
"""
from __future__ import annotations

import traceback
from types import SimpleNamespace

from eval.corpus import load as load_corpus
from eval.retriever import Hit

from .agent import PRICING, CRMAgent, is_followup, match_variation
from .context import (_collapse_dotted, _detect_from_evidence, _is_targeted,
                      _is_universal,
                      _known_form_names, _names_absent_form, build_context,
                      build_documents, named_forms,
                      detect_ambiguity, detect_intent, render_ambiguity,
                      validate_citations)

_failures: list[str] = []


def check(label: str, got, want) -> None:
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got!r}, want {want!r}")
    if not ok:
        _failures.append(label)


def check_true(label: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{(' -- ' + detail) if detail else ''}")
    if not cond:
        _failures.append(label)


def _hits(ids) -> list[Hit]:
    return [Hit(cid, 1.0 - i * 0.01) for i, cid in enumerate(ids)]


class StubClient:
    """Records the messages it was given and replies with a canned answer."""

    def __init__(self, reply: str = "Одговор [X]."):
        self.reply = reply
        self.calls: list[list[dict]] = []
        self.offered_tools: list | None = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    # `tools` is accepted because the real client is now called with it. A stub
    # whose signature drifts from the thing it stands in for stops testing the
    # code path it is supposed to cover.
    def _create(self, *, model, messages, temperature, tools=None):
        self.offered_tools = tools
        self.calls.append(messages)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))


class StubRetriever:
    name = "stub"

    def __init__(self, ids):
        self.ids = ids
        self.queries: list[tuple[str, dict | None]] = []

    def search(self, query, k, *, filters=None):
        self.queries.append((query, filters))
        return _hits(self.ids)[:k]


def _dedup_tests(c) -> None:
    print("[dedup]")
    # The same FAQ text exists verbatim under 14 services.
    copies = c.identical_to("srv_2185_shared_faq_1")
    check_true("corpus really does duplicate this block", len(copies) > 5,
               f"{len(copies)} identical copies")

    docs = build_documents(_hits(copies[:6]), c, dedup=True)
    check("6 identical blocks collapse to 1", len(docs), 1)
    check_true("collapsed copies keep provenance", len(docs[0].also_in) >= 1,
               f"also_in={docs[0].also_in[:2]}")

    kept = build_documents(_hits(copies[:6]), c, dedup=False)
    check("dedup can be turned off", len(kept), 6)

    mixed = ["srv_2162_v11050_tariffs_1", *copies[:4], "srv_2063_v11187_process_1"]
    docs = build_documents(_hits(mixed), c)
    check("dedup frees slots but keeps distinct blocks", len(docs), 3)
    check("rank order is preserved", docs[0].chunk_id, "srv_2162_v11050_tariffs_1")


def _ambiguity_tests(c) -> None:
    print("[ambiguity]")
    # Registration fees genuinely differ per legal form -> must ask.
    tariffs = [b.chunk_id for b in c.by_service[2135] if b.type == "tariffs"][:5]
    docs = build_documents(_hits(tariffs), c)
    amb = detect_ambiguity(docs, "Колку чини упис на основање?", c)
    check_true("fires when fees differ across legal forms", amb is not None)
    if amb:
        check("names the right service", amb.id_service, 2135)
        check("detector source recorded", amb.source, "evidence")
        check("flags the conflicting section", amb.section_types, ["tariffs"])
        # Observed live: a question that retrieved two forms offered the user
        # only those two, leaving the other seven unreachable.
        check("offers every form, not only the retrieved ones",
              len(amb.variations), 11)

    # Same documents, but the user already said which form.
    amb2 = detect_ambiguity(docs, "Колку чини упис на основање на Здружение?", c)
    check_true("does not fire when the user named the form", amb2 is None)

    # All five variations share ONE identical access block -> dedup collapses
    # them, so there is nothing to disambiguate.
    access = [b.chunk_id for b in c.by_service[2111] if b.type == "access"]
    docs3 = build_documents(_hits(access), c)
    amb3 = detect_ambiguity(docs3, "Дали годишна сметка може онлајн?", c)
    check("near-identical access blocks collapse", len(docs3), 2)
    check_true("does not fire when the forms agree", amb3 is None)

    # A single-variation service is never ambiguous.
    solo = [b.chunk_id for b in c.by_service[2029] if b.type == "tariffs"]
    amb4 = detect_ambiguity(build_documents(_hits(solo), c), "Колку чини?", c)
    check_true("does not fire for a single-variation service", amb4 is None)

    xml, _, _ = build_context(_hits(tariffs), c, "Колку чини упис на основање?")
    check_true("context carries the ambiguity block", "<ambiguity>" in xml)
    check_true("context escapes and labels documents",
               '<document id="' in xml and 'variant="' in xml)
    check_true("documents are not mislabelled as legal forms",
               'legal_form="' not in xml)


def _structure_ambiguity_tests(c) -> None:
    """Regression fixture from a real failure.

    `python -m agent.cli -q "Колку чини регистрација?"` retrieved these exact
    five blocks and NO tariff rows: "регистрација" matched documents about
    registering a user account in the e-system, while the fee rows say "упис на
    основање". The evidence-based detector had nothing to see, so the agent
    never asked which legal form -- on a question where АД and ПДОО cost 0 МКД
    and the other seven forms cost 2452 МКД.
    """
    print("[structure-based ambiguity]")
    observed = ["srv_2155_v0_terminology_5", "srv_2135_shared_terminology_2",
                "srv_2135_v11116_documents_17", "srv_2194_shared_faq_1",
                "srv_2183_shared_faq_42"]
    docs = build_documents(_hits(observed), c)

    check_true("evidence-only detector stays silent (the original bug)",
               detect_ambiguity(docs, "Колку чини регистрација?") is None)

    amb = detect_ambiguity(docs, "Колку чини регистрација?", c)
    check_true("structure detector fires on the real failing case", amb is not None)
    if amb:
        check("attributes the right service", amb.id_service, 2135)
        check("detector source recorded", amb.source, "structure")
        check("asks about the section the user asked about",
              amb.section_types, ["tariffs"])
        check_true("offers every legal form", len(amb.variations) == 11,
                   f"{len(amb.variations)}: {', '.join(amb.labels[:3])}...")

    check("intent detection reads the question", detect_intent("Колку чини регистрација?"),
          ["tariffs"])
    check_true("'начин' does not trigger the 'чин' fee cue",
               "tariffs" not in detect_intent("На кој начин се поднесува?"))

    # Must NOT fire: single-variation service.
    solo = build_documents(_hits(["srv_2063_v11187_process_1"]), c)
    check_true("silent for a single-variation service",
               detect_ambiguity(solo, "Како да регистрирам залог?", c) is None)

    # Must NOT fire: the forms agree, so the choice changes nothing.
    same = build_documents(_hits(["srv_2111_v11176_access"]), c)
    check_true("silent when the forms share one identical answer",
               detect_ambiguity(same, "Дали може онлајн?", c) is None)

    # Must NOT fire: user already named the form.
    check_true("silent when the user named the legal form",
               detect_ambiguity(docs, "Колку чини регистрација на Здружение?", c) is None)

    # Must NOT fire: no readable intent -> do not guess which section differs.
    check_true("silent when the intent is unclear",
               detect_ambiguity(docs, "Информации за субјект", c) is None)

    # Attribution must ignore shared boilerplate: the rank-1 block belongs to
    # 2155 (zero variations) but is shared text.
    only_shared = build_documents(_hits(["srv_2155_v0_terminology_5",
                                         "srv_2194_shared_faq_1"]), c)
    check_true("no attribution from boilerplate alone",
               detect_ambiguity(only_shared, "Колку чини регистрација?", c) is None)

    # --- fixtures from the first answer-eval run -------------------------- #
    # Both were real clarifications the agent should never have asked.
    from index.structured import attribute, fetch_sections

    # "Како да регистрирам залог?" -- alias expansion appended "упис/основање",
    # the dense arm returned six Фондација process rows, attribution locked onto
    # service 2135 and the agent answered a pledge question with foundation
    # steps, then asked which legal form.
    drift = ["srv_2063_shared_terminology_2",
             "srv_2135_v11117_process_1", "srv_2135_v11117_process_2",
             "srv_2135_v11117_process_3", "srv_2135_v11117_process_4",
             "srv_2063_v11187_forms_5", "srv_2135_v11117_process_5",
             "srv_2063_v11187_forms_6", "srv_2135_v11117_process_6",
             "srv_2063_v11187_documentsLocations_2"]
    hits = _hits(drift)
    check("unanchored attribution follows the alias drift",
          attribute(hits, c).id_service, 2135)
    check("anchoring keeps attribution on what was asked",
          attribute(hits, c, anchor_query="Како да регистрирам залог?").id_service, 2063)
    fetched = fetch_sections(c, "Како да регистрирам залог?", hits)
    check_true("...and the right section is fetched",
               bool(fetched) and all("2063" in cid for cid in fetched),
               f"{len(fetched)} rows, first={fetched[0] if fetched else None}")
    check_true("no clarification is asked for a single-variation service",
               detect_ambiguity(build_documents(hits, c),
                                "Како да регистрирам залог?", c) is None)

    # "...годишна сметка ЗА БАНКА?" -- the user named the variant, but its label
    # is "Банки и финансиски институции", so substring matching missed it.
    bank = ["srv_2111_v11176_deadlines_1", "srv_2111_v11176_deadlines_2",
            "srv_2111_v11179_deadlines_1"]
    check_true("a named variant is recognised through its head noun",
               detect_ambiguity(build_documents(_hits(bank), c),
                                "Кој е крајниот рок за поднесување годишна "
                                "сметка за банка?", c) is None)
    check_true("...but an unnamed one still asks",
               detect_ambiguity(build_documents(_hits(bank), c),
                                "Кој е крајниот рок за поднесување годишна "
                                "сметка?", c) is not None)

    # --- universal rows are agreement, not conflict ----------------------- #
    # Live: the agent answered "what do I do with an English-language document?"
    # from documents_1 of service 2135 -- byte-identical across all NINE legal
    # forms -- and then asked the user to choose between the nine. The corpus
    # files one physical copy per variation, so is_shared is False on every one,
    # and build_documents() dedups them to a single arbitrarily-labelled copy.
    universal = "srv_2135_v11113_documents_1"
    variant_only = "srv_2135_v11118_documents_6"
    check_true("a row identical across every variation reads as universal",
               _is_universal(c, 2135, c.by_id[universal]))
    check_true("...while a one-form row does not",
               not _is_universal(c, 2135, c.by_id[variant_only]))
    english = ("Имам доказ за регистрација кој е на англиски јазик. "
               "Што точно треба да направам со него?")
    # Two forms contributing `documents` rows is the shape that used to fire, but
    # one of the two rows is universal, so only ONE form actually has a say.
    check_true("a universal row does not count as a form disagreeing",
               _detect_from_evidence(
                   build_documents(_hits([universal, variant_only]), c),
                   english, c) is None)
    check_true("...and a context of only universal rows asks nothing at all",
               detect_ambiguity(build_documents(_hits([universal]), c),
                                english, c) is None)
    # The counterweight, kept adjacent on purpose: the cheap way to pass the
    # check above is to stop asking about `documents` at all, which breaks this.
    # Заедница на сопственици and Приватна установа have no electronic-pickup row,
    # so "електронски или на шалтер" is false for two of the eight forms.
    pickup = [b.chunk_id for b in c.of_type(2140, "documentsLocations", 11166)][:2]
    check_true("...but a genuinely variant-specific section still asks",
               detect_ambiguity(build_documents(_hits(pickup), c),
                                "Каде можам да го подигнам документот", c)
               is not None)

    # --- naming a form the service does not have ------------------------- #
    # This fixture used to use ТП, on the evidence that no service offered it.
    # That was never true of the REGISTRY -- ТП variations existed all along and
    # the scraper had silently skipped them, so the test encoded a data gap as
    # if it were a rule. It now uses ПДОО, which service 2135 offers and 2140
    # genuinely does not, and asserts ТП in the opposite direction so a
    # regression in the scrape would be caught here rather than by a user.
    forms_2140 = [c.variation_name.get((2140, v), "")
                  for v in c.variations_of.get(2140, ())]
    check_true("ТП is a real variant of the change service",
               "ТП" in forms_2140, ", ".join(sorted(forms_2140))[:70])
    check_true("ПДОО is a form the corpus knows",
               "ПДОО" in _known_form_names(c))
    check_true("...but is not a variant of the attributed service",
               "ПДОО" not in forms_2140)
    check_true("naming an absent form suppresses the question",
               _names_absent_form("промена за ПДОО", c, forms_2140))
    check_true("...while naming a form the service HAS does not",
               not _names_absent_form("упис на основање на Здружение", c,
                                      [c.variation_name.get((2140, v), "")
                                       for v in c.variations_of.get(2140, ())]))
    check_true("...and naming no form at all does not",
               not _names_absent_form("Колку чини регистрација?", c,
                                      [c.variation_name.get((2140, v), "")
                                       for v in c.variations_of.get(2140, ())]))

    # --- injected rows are context, not evidence -------------------------- #
    # Structured fetch resolves ONE variation and pastes its whole section in.
    # Counting those rows made the detector answer a question it had asked
    # itself: five Фондација `documents` rows arrived by injection, one
    # Подружница row by search, and it reported that two forms disagreed.
    fond = [b.chunk_id for b in c.of_type(2135, "documents", 11117)][:4]
    other = "srv_2135_v11118_documents_6"
    english = ("Имам доказ за регистрација кој е на англиски јазик. "
               "Што точно треба да направам со него?")
    injected = build_documents(_hits(fond + [other]), c, limit=10, pin=fond)
    check("injected rows are flagged",
          sum(1 for d in injected if d.injected), len(fond))
    check_true("a manufactured disagreement is not evidence",
               _detect_from_evidence(injected, english, c) is None)
    check_true("...and the same rows retrieved by SEARCH still are",
               _detect_from_evidence(build_documents(_hits(fond + [other]), c),
                                     english, c) is not None)
    # Dedup merges a searched copy into an injected one; the row is then still
    # evidence, because search did find it.
    universal = "srv_2135_v11113_documents_1"          # identical to v11117_1
    merged = build_documents(_hits([fond[0], universal]), c, pin=[fond[0]])
    check("identical copies collapse to one", len(merged), 1)
    check_true("a row search also found is not marked injected",
               not merged[0].injected)

    # --- channels are variants too ---------------------------------------- #
    # Live: offered "Web сервис" / "Хартиено на шалтер", the user answered
    # "преку интернет", was asked again, answered "шалтер", and was asked a
    # third time. The resolver only accepted the registry's own label, so the
    # menu could not be escaped. And a channel named in the ORIGINAL question
    # ("како да ја добијам преку интернет") has to scope it without asking.
    channels = [(1, "Web сервис"), (2, "Хартиено на шалтер")]
    for reply, want in (("преку интернет", "Web сервис"),
                        ("онлајн", "Web сервис"),
                        ("шалтер", "Хартиено на шалтер"),
                        ("хартиено", "Хартиено на шалтер")):
        got = match_variation(reply, channels, c)
        check(f"a menu reply of {reply!r} resolves", got[1] if got else None, want)
    check_true("an unrelated reply still resolves to nothing",
               match_variation("АД", channels, c) is None)

    # People punctuate these abbreviations; the registry never does. Latin
    # spellings matter too -- Macedonian keyboards are not universal.
    forms_menu = [(11113, "АД"), (11111, "ДОО, ДООЕЛ"), (11112, "ТП")]
    for reply, want in (("а.д.", "АД"), ("a.d.", "АД"), ("ad", "АД"),
                        ("д.о.о.", "ДОО, ДООЕЛ"), ("doo", "ДОО, ДООЕЛ"),
                        ("dooel", "ДОО, ДООЕЛ"), ("тп", "ТП"), ("tp", "ТП")):
        got = match_variation(reply, forms_menu, c)
        check(f"punctuated/latin {reply!r} resolves", got[1] if got else None, want)
    check("...and a dotted form is recognised up front",
          named_forms("Колку чини упис за а.д.?", c), ["АД"])
    # Scoped to single-letter runs so it cannot mangle a real dotted string.
    check("collapsing leaves URLs alone",
          _collapse_dotted("линк: e-submit.crm.com.mk"), "линк: e-submit.crm.com.mk")
    check_true("a channel named up front is recognised",
               "Web сервис" in named_forms(
                   "Ми треба тековна состојба. Како да ја добијам преку интернет?", c))
    check("...and an ordinary question names no form",
          named_forms("Колку чини регистрација?", c), [])

    # --- targeted vs enumerative ------------------------------------------ #
    check_true("a pointed question is targeted", _is_targeted(english))
    check_true("...and a list question is not",
               not _is_targeted("Кои документи ми требаат за да регистрирам фирма?"))
    check_true("'каде' asks for a set, not one row",
               not _is_targeted("Каде можам да го подигнам документот"))

    xml = render_ambiguity(detect_ambiguity(docs, "Колку чини регистрација?", c))
    check_true("prompt warns the rows may be absent from the context",
               "may contain none of them" in xml)


def _citation_tests(c) -> None:
    print("[citations]")
    docs = build_documents(_hits(["srv_2162_v11050_tariffs_1"]), c)
    valid, stale, invalid = validate_citations(
        "Тарифата е 295 МКД [srv_2162_v11050_tariffs_1].", docs)
    check("real citation accepted", valid, ["srv_2162_v11050_tariffs_1"])
    check("no false positives", (stale, invalid), ([], []))

    valid, stale, invalid = validate_citations("Измислено [srv_9999_v0_tariffs_1].", docs)
    check("fabricated citation caught", invalid, ["srv_9999_v0_tariffs_1"])
    check("fabricated citation not counted as valid", valid, [])

    # Observed live: the model answered a repeated question by re-citing a
    # tariff row from two turns earlier that the current retrieval had not
    # returned. Shown-before is not made-up, and must not raise the same alarm.
    valid, stale, invalid = validate_citations(
        "Пак е 0 МКД [srv_2135_v11113_tariffs_1].", docs,
        seen={"srv_2135_v11113_tariffs_1"})
    check("citation from an earlier turn is stale, not fabricated",
          (valid, stale, invalid), ([], ["srv_2135_v11113_tariffs_1"], []))

    # Regression: a markdown link used to be read as a fabricated chunk_id,
    # which flagged a correct, properly-abstaining answer as unverified.
    md = ("Проверете ја тарифата на "
          "[https://www.crm.com.mk/api/files/f0f00e9b?ln=1](https://www.crm.com.mk/api/files/f0f00e9b?ln=1).")
    valid, stale, invalid = validate_citations(md, docs)
    check("markdown link is not a citation", (valid, stale, invalid), ([], [], []))
    valid, stale, invalid = validate_citations(
        f"Тарифата е 295 МКД [srv_2162_v11050_tariffs_1]. Види [линк](http://x).", docs)
    check("real citation survives alongside a markdown link",
          (valid, stale, invalid), (["srv_2162_v11050_tariffs_1"], [], []))


def _clarification_tests(c) -> None:
    print("[clarification matching]")
    tariffs = [b.chunk_id for b in c.by_service[2135] if b.type == "tariffs"][:5]
    amb = detect_ambiguity(build_documents(_hits(tariffs), c), "Колку чини?", c)
    assert amb is not None

    picked = match_variation("АД", amb.variations, c)
    check_true("short label matched", picked is not None and picked[1] == "АД",
               str(picked))
    check_true("word boundary respected (адреса is not АД)",
               match_variation("не знам адресата", amb.variations, c) is None)
    picked2 = match_variation("за здружение станува збор", amb.variations, c)
    check_true("long label matched inside a sentence",
               picked2 is not None and picked2[1] == "Здружение", str(picked2))
    check_true("unrelated reply matches nothing",
               match_variation("не знам", amb.variations, c) is None)

    # Observed live: "за АД" resolved, "акционерско друштво" did not, because
    # the corpus labels are abbreviations and users write them out.
    spelled = match_variation("а колку е за акционерско друштво", amb.variations, c)
    check_true("spelled-out legal form resolves via the glossary",
               spelled is not None and spelled[1] == "АД", str(spelled))
    # ПДОО ("Поедноставено ДОО") is a SIMPLIFIED fast-track and a different
    # procedure. While the real "ДОО, ДООЕЛ" variation was missing from the
    # corpus, "доо" landed on ПДОО and this fixture asserted that as correct.
    doo = match_variation("ме интересира ДОО", amb.variations, c)
    check_true("ДОО resolves to its OWN variation, not the simplified one",
               doo is not None and doo[1] == "ДОО, ДООЕЛ", str(doo))

    print("[follow-up detection]")
    check_true("discourse opener marks a follow-up",
               is_followup("а колку е за акционерско друштво"))
    check_true("bare reply marks a follow-up", is_followup("за АД"))
    check_true("anaphor marks a follow-up", is_followup("а тоа колку трае"))
    check_true("a short real question stays self-contained",
               not is_followup("Како да регистрирам залог?"))
    check_true("a long real question stays self-contained",
               not is_followup("Кои документи се потребни за упис на основање?"))
    # Observed in a multi-turn eval: turn 3 asked for examples of the agents
    # just listed, was read as a new question, retrieved nothing and claimed to
    # have no information about them -- one turn after listing forty-five.
    check_true("a continuation request is a follow-up",
               is_followup("Дај ми неколку како пример"))
    check_true("...but a request that names its own subject is not",
               not is_followup("Дај ми го ЗП образецот"))
    check_true("...nor is one that names a place",
               not is_followup("Дај ми список на адвокати во Прилеп"))


def _tool_tests(c) -> None:
    """The live-lookup loop, driven entirely from fixtures.

    No network and no API key: DISPATCH is injectable for exactly this reason.
    A gate that needs a government portal to be up is a gate people stop running.
    """
    print("[tools]")
    from types import SimpleNamespace

    from .tools.entity_size import (EntitySize, EntitySizeArgs, TOOL_SPEC,
                                    parse_entity_size)
    from bs4 import BeautifulSoup

    # --- parsing, against the markup the live form actually returns --------- #
    html = ('<span id="ctl00_cphMain_ucCheckLeSize_LblResult">'
            'Големината на правниот субјект е: мал        </span>')
    got = parse_entity_size(BeautifulSoup(html, "html.parser"),
                            "07696876", "2026-09-18T00:00:00+00:00")
    check("the size class is read out", got.size, "мал")
    check("trailing whitespace is collapsed", got.message,
          "Големината на правниот субјект е: мал")
    # An unknown entity must NOT be confused with a broken integration: this
    # returns found=False, while a missing span raises. See parse_entity_size.
    unknown = parse_entity_size(BeautifulSoup(
        html.replace("Големината на правниот субјект е: мал",
                     "Не е пронајден субјект"), "html.parser"), "9999999", "t")
    check("an unknown entity is not a size", unknown.size, None)
    check_true("...and is reported as not found", not unknown.found)
    try:
        parse_entity_size(BeautifulSoup("<div>nothing</div>", "html.parser"),
                          "1234567", "t")
        check_true("a missing result span raises rather than returning empty", False)
    except RuntimeError:
        check_true("a missing result span raises rather than returning empty", True)

    # --- the ЕМБС guard: an invented id must never reach the portal --------- #
    for bad in ("123", "abcdefgh", "123456789"):
        try:
            EntitySizeArgs(embs=bad)
            check_true(f"{bad!r} is rejected before any request", False)
        except Exception:
            check_true(f"{bad!r} is rejected before any request", True)
    check("a leading zero is never stripped", EntitySizeArgs(embs="07696876").embs,
          "07696876")

    # --- strict-mode schema hygiene ---------------------------------------- #
    params = TOOL_SPEC["function"]["parameters"]
    check_true("strict mode needs additionalProperties=false",
               params.get("additionalProperties") is False)
    check_true("the class docstring does not leak to the model",
               "description" not in params)

    # --- the loop ----------------------------------------------------------- #
    fixture = EntitySize(embs="07696876", size="мал",
                         message="Големината на правниот субјект е: мал",
                         fetched_at="2026-09-18T00:00:00+00:00", found=True)
    calls = {"n": 0}

    def fake_lookup(**kw):
        calls["n"] += 1
        return fixture

    class ToolThenAnswer:
        """Asks for the lookup once, then answers citing what it was given."""

        def __init__(self):
            self.turn = 0
            self.seen_tool_content: list[str] = []
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create))

        def _create(self, *, model, messages, temperature, tools=None):
            self.turn += 1
            usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20)
            if self.turn == 1:
                check_true("tools are offered to the model", bool(tools))
                call = SimpleNamespace(
                    id="call_1",
                    function=SimpleNamespace(name="check_entity_size",
                                             arguments='{"embs": "07696876"}'))
                msg = SimpleNamespace(content=None, tool_calls=[call])
                return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                                       usage=usage)
            self.seen_tool_content = [m["content"] for m in messages
                                      if isinstance(m, dict) and m.get("role") == "tool"]
            msg = SimpleNamespace(
                content="Големината е мал [tool:entity_size:07696876].",
                tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)

    client = ToolThenAnswer()
    agent = CRMAgent(corpus=c, retriever=StubRetriever(
        [b.chunk_id for b in c.by_service[2135] if b.type == "tariffs"][:3]),
        client=client, dispatch={"check_entity_size": fake_lookup},
        tools=[TOOL_SPEC])
    a = agent.ask("Која е големината на субјектот со ЕМБС 07696876?")

    check("the tool ran exactly once", calls["n"], 1)
    check("the call is recorded on the Answer", a.tool_calls, ["check_entity_size"])
    check_true("the model is told the citation id",
               any("tool:entity_size:07696876" in c_ for c_ in client.seen_tool_content),
               "without it the answer cannot be cited and the guard deletes it")
    # The whole point: a live result must survive validate_citations AND the
    # hard-abstention guard, which replaces any answer carrying no valid id.
    check("the live citation validates", a.citations, ["tool:entity_size:07696876"])
    check("...and nothing is flagged fabricated", a.invalid_citations, [])
    check_true("the answer is not replaced by the abstention guard",
               a.text.startswith("Големината е мал"), a.text[:60])
    check_true("the tool doc joins the context",
               any(d.chunk_id == "tool:entity_size:07696876" for d in a.docs))
    check_true("a repeat citation next turn would be stale, not fabricated",
               "tool:entity_size:07696876" in agent.seen_docs)
    check("both rounds are billed", a.prompt_tokens, 200)

    # State isolation: a tool_calls message in history breaks the NEXT request,
    # because OpenAI requires every tool_call_id to be answered in the same list
    # and history is both trimmed and truncated.
    check("history keeps only the user/assistant pair", len(agent.history), 2)
    check_true("no tool scaffolding leaks into history",
               all(isinstance(m, dict) and m.get("role") in ("user", "assistant")
                   and "tool_calls" not in m for m in agent.history))


def _profile_tool_tests(c) -> None:
    """Form 2: the vision extraction's self-checks, with no model and no network.

    What cannot be tested here is whether gpt-4o reads the PNG correctly -- that
    needs the model and a real image. What CAN be tested, and matters more, is
    everything that decides what happens when it reads it wrong.
    """
    print("[entity profile]")
    from .tools.entity_profile import (ASSETS, TEMPLATE_PATH, LocalImageSource,
                                       ProfileMismatch, ProfileResult,
                                       ProfileUnavailable, TOOL_SPEC,
                                       _same_embs, fetch_profile,
                                       profile_image_from_file, verify_profile)
    from .tools.fixtures import RECORDED_PROFILES, _FixtureSource

    # The real read of agent/tools/assets/basic_profile_7696876.png, verified
    # field by field against the picture and kept in fixtures.py. Reused here so
    # there is ONE recording of that entity: a second copy in the test file
    # would drift from the one the eval replays and nothing would notice.
    good = RECORDED_PROFILES["7696876"]

    check("a clean read raises no shape problems", verify_profile(good, "7696876"), [])
    check_true("a short ЕДБ is caught",
               bool(verify_profile(good.model_copy(update={"edb": "405802"}), "7696876")))
    check_true("a reformatted date is caught",
               bool(verify_profile(good.model_copy(update={"founded": "2023-09-19"}),
                                   "7696876")))
    check_true("an invented size class is caught",
               bool(verify_profile(good.model_copy(update={"size": "средно"}),
                                   "7696876")))

    # Zero padding: the registry prints 7696876 and accepts 07696876 for the
    # same company, so a strict comparison would reject a CORRECT profile.
    check_true("zero padding does not count as a different entity",
               _same_embs("07696876", "7696876"))
    check_true("...but a different number does",
               not _same_embs("7405855", "7696876"))

    # The check that earns the vision step: a profile for the wrong company is
    # worse than no profile, because every field in it looks plausible.
    def wrong_entity(image_bytes, **kw):
        return good.model_copy(update={"embs": "7405855"})

    import agent.tools.entity_profile as ep
    png = ASSETS / "basic_profile_7696876.png"
    real_extract = ep.extract_profile
    ep.extract_profile = wrong_entity
    try:
        fetch_profile("7696876", image_path=png)
        check_true("a mismatched profile is refused, not returned", False)
    except ProfileMismatch:
        check_true("a mismatched profile is refused, not returned", True)
    finally:
        ep.extract_profile = real_extract

    # ACQUISITION. Form 2 sits behind reCAPTCHA v3 and this module does not go
    # looking for a token, so the offline source is the working one and a miss
    # has to be an ANSWER rather than a crash or an unauthorised retry.
    check_true("the saved-PNG path reads the asset",
               profile_image_from_file(png).startswith(b"\x89PNG"))
    try:
        profile_image_from_file(TEMPLATE_PATH)
        check_true("a non-PNG is rejected", False)
    except ValueError:
        check_true("a non-PNG is rejected", True)
    check_true("the render template is on disk for the live path",
               TEMPLATE_PATH.exists() and "[^$.LEID^]" in
               TEMPLATE_PATH.read_text(encoding="utf-8"))

    # An entity nobody has saved. This is the common case in production, so it
    # is the one that must not raise: the tool loop hands whatever comes back to
    # the model, and an exception here takes down the whole turn.
    miss = fetch_profile("1234567", source=_FixtureSource())
    check_true("an unheld entity returns a value, not an exception",
               isinstance(miss, ProfileUnavailable))
    check("...flagged unavailable", miss.available, False)
    check("...with a machine-readable reason", miss.reason, "no_local_image")
    check_true("...and a Macedonian message to relay",
               "не е достапен" in miss.message)
    # The guard deletes any answer that cites nothing, so the miss must be
    # citable or an honest "we do not hold that" gets replaced by boilerplate.
    mdoc = miss.as_context_doc()
    check("an unavailable profile is still citable", mdoc.chunk_id,
          "tool:entity_profile:1234567:unavailable")
    check_true("...and cannot be mistaken for profile data",
               "ЕДБ:" not in mdoc.block.content
               and "Големина:" not in mdoc.block.content)
    check_true("both result shapes expose `available` for the model to branch on",
               "available" in miss.model_dump()
               and ProfileResult(profile=good, fetched_at="x").available is True)

    # PADDING. The registry prints 7696876, users type 07696876, and the
    # operator saving the PNG picks one. Making the human guess which spelling
    # the cache wants is how a profile that is present reads as missing.
    src = LocalImageSource(ASSETS)
    for spelling in ("7696876", "07696876"):
        check_true(f"{spelling} resolves to the saved image",
                   any(p.is_file() for p in src.candidates(spelling)))

    # THE MIGRATION POINT. A source that returns fields with no image and no
    # vision call must satisfy the same contract, or "swap the transport later"
    # is a promise the seam cannot keep.
    class _FieldsOnly:
        name = "api:test"

        def load(self, embs):
            return good.model_copy(update={"embs": embs})

    api = fetch_profile("07696876", source=_FieldsOnly())
    check("a fields-only transport needs no vision step", api.source, "api:test")
    check_true("...and produces the same citable shape",
               api.as_context_doc().chunk_id.startswith("tool:entity_profile:")
               and "Големина: мал" in api.as_context_doc().block.content)
    # Validation is source-independent: an API can transpose a digit too.
    bad_api = type("_BadApi", (), {
        "name": "api:bad",
        "load": lambda self, embs: good.model_copy(update={"edb": "405"})})()
    try:
        fetch_profile("7696876", source=bad_api)
        check_true("shape checks run whatever the source", False)
    except ValueError:
        check_true("shape checks run whatever the source", True)

    # Everything the user could be told must be in the citable text, or
    # numeric_groundedness reads a correctly-copied ЕДБ as a fabrication.
    doc = ProfileResult(profile=good, fetched_at="2026-09-18T12:00:00+00:00"
                        ).as_context_doc()
    check("the profile is citable", doc.chunk_id, "tool:entity_profile:7696876")
    for label, value in (("ЕМБС", "7696876"), ("ЕДБ", "4058023546097"),
                         ("датум", "19.09.2023"), ("големина", "мал"),
                         ("целосен назив", "ЛОРА КОМПАНИ 2023 ДОО Скопје")):
        check_true(f"{label} survives into the citable text", value in doc.block.content)
    # The user's padded spelling must be groundable too -- the image prints
    # 7696876, and an answer opening "субјектот со ЕМБС 07696876" was scored
    # as citing a number its source never contained.
    padded = ProfileResult(profile=good, fetched_at="x",
                           requested_embs="07696876").as_context_doc()
    check_true("the ЕМБС as asked is in the citable text",
               "07696876" in padded.block.content
               and "ЕМБС: 7696876" in padded.block.content)
    check_true("...and is not repeated when it matches the image",
               "Побаран" not in doc.block.content)
    # An absent row prints nothing rather than an empty label: a bare
    # "Дополнителни податоци:" reads as a fact the registry stated and did not.
    check_true("an absent row leaves no empty label behind",
               "Дополнителни податоци" not in doc.block.content)

    params = TOOL_SPEC["function"]["parameters"]
    check_true("strict schema on the profile tool too",
               params.get("additionalProperties") is False
               and "description" not in params)

    # REGISTRATION. The live registry and the fixture registry must offer the
    # same tools: a gate that replays a tool production does not expose, or
    # misses one it does, is measuring a different agent than the one shipping.
    from .tools import DISPATCH as LIVE_DISPATCH, REGISTRY
    from .tools.fixtures import DISPATCH as FIXTURE_DISPATCH
    names = [t["function"]["name"] for t in REGISTRY]
    check_true("Form 2 is registered", "get_entity_profile" in names)
    check("every registered tool is dispatchable",
          sorted(names), sorted(LIVE_DISPATCH))
    check("...and every one has a fixture for the gates",
          sorted(FIXTURE_DISPATCH), sorted(LIVE_DISPATCH))
    # NO REAL IDENTIFIERS IN WHAT THE MODEL READS. Every tool once shipped a
    # real company's ЕМБС or filing number as its "e.g.", and the system prompt
    # did too -- so a model that copied an example instead of asking the user
    # would have looked up a real company nobody mentioned. Any run of 7+
    # digits (ЕМБС 7-8, ЕДБ 13, деловоден број 14) in a contract is one of those.
    import json as _json
    import re as _re

    from .prompts import SYSTEM_PROMPT
    leaked = {
        name: _re.findall(r"\d{7,}", text) for name, text in
        [(t["function"]["name"], _json.dumps(t, ensure_ascii=False))
         for t in REGISTRY] + [("SYSTEM_PROMPT", SYSTEM_PROMPT)]}
    check("no tool contract or prompt carries a real identifier",
          {k: v for k, v in leaked.items() if v}, {})
    check_true("the tool tells the model to prefer the cheaper size lookup",
               "check_entity_size" in TOOL_SPEC["function"]["description"])
    check_true("...and not to retry an unavailable profile",
               "available=false" in TOOL_SPEC["function"]["description"])


def _decision_tool_tests(c) -> None:
    """Form 3: a published Решение, keyed by деловоден број. No model, no network."""
    print("[registration decision]")
    from pydantic import ValidationError

    from .tools.registration_decision import (ASSETS, TOOL_SPEC, DecisionArgs,
                                              DecisionMismatch, DecisionResult,
                                              DecisionUnavailable,
                                              LocalDecisionSource,
                                              RegistrationDecision,
                                              fetch_decision, verify_decision)
    from .tools.fixtures import RECORDED_DECISIONS, _FixtureDecisionSource
    from .tools.schema import strict_schema

    number = "30120260014967"
    good = RECORDED_DECISIONS[number]

    # THE IDENTIFIER. An ЕМБС handed to this tool must fail validation, so the
    # model is told why and asks for the real number instead of the lookup
    # running on something that was never a filing number.
    check("a 14-digit деловоден број is accepted",
          DecisionArgs(deloveden_broj=f" {number} ").deloveden_broj, number)
    for wrong, why in (("7405855", "an ЕМБС"), ("3012026001496", "13 digits"),
                       ("301-2026-0014967", "punctuation")):
        try:
            DecisionArgs(deloveden_broj=wrong)
            check_true(f"{why} is refused as a деловоден број", False)
        except ValidationError:
            check_true(f"{why} is refused as a деловоден број", True)

    check("the recorded read passes every check", verify_decision(good), [])

    def with_row(label: str, value: str) -> RegistrationDecision:
        """`good` with one row's value replaced -- a single simulated misread."""
        sections = [s.model_copy(update={"rows": [
            r.model_copy(update={"value": value}) if r.label == label else r
            for r in s.rows]}) for s in good.sections]
        return good.model_copy(update={"sections": sections})

    # CONSISTENCY. Each identifier is read twice -- as a field and as a table
    # row -- and a misread in either copy fails the document.
    check_true("a misread деловоден број row is caught",
               bool(verify_decision(with_row("Деловоден број", "30120260014961"))))
    check_true("a misread ЕМБС row is caught",
               bool(verify_decision(with_row("ЕМБС", "7405856"))))
    check_true("...but ЕМБС zero padding is not a misread",
               verify_decision(with_row("ЕМБС", "07405855")) == [])
    check_true("an invented size class is caught",
               bool(verify_decision(with_row("Големина на субјектот", "мало"))))
    check_true("a short деловоден број field is caught", bool(verify_decision(
        good.model_copy(update={"deloveden_broj": "3012026001496"}))))
    check_true("a document with no tables is caught",
               bool(verify_decision(good.model_copy(update={"sections": []}))))

    # THE PREAMBLE IS NOT READ. Three misreads in one paragraph, none of them
    # checkable; the schema must not quietly grow the field back.
    fields = set(RegistrationDecision.model_fields)
    check_true("the schema carries no preamble or prose-derived date",
               not fields & {"preamble", "decision_date", "registrar"})

    # SELF-CHECK. The wrong filing is refused, not returned.
    class _Other:
        name = "test"

        def load(self, n):
            return good.model_copy(update={"deloveden_broj": "30120260014978"})

    try:
        fetch_decision(number, source=_Other())
        check_true("another filing's decision is refused", False)
    except DecisionMismatch:
        check_true("another filing's decision is refused", True)

    # A MISS IS AN ANSWER.
    miss = fetch_decision("30120260014978", source=_FixtureDecisionSource())
    check_true("an unheld filing returns a value, not an exception",
               isinstance(miss, DecisionUnavailable))
    check("...flagged unavailable", miss.available, False)
    mdoc = miss.as_context_doc()
    check("...and is citable", mdoc.chunk_id,
          "tool:registration_decision:30120260014978:unavailable")
    check_true("...in wording the abstention guard recognises",
               "не е достапно" in mdoc.block.content)
    check_true("...and cannot be mistaken for decision data",
               "ЕМБС" not in mdoc.block.content and "микро" not in mdoc.block.content)

    # CITABLE TEXT. Everything a user could be told, verbatim.
    hit = fetch_decision(number, source=_FixtureDecisionSource())
    check_true("a held filing returns the decision", isinstance(hit, DecisionResult))
    doc = hit.as_context_doc()
    check("the decision is citable", doc.chunk_id,
          f"tool:registration_decision:{number}")
    for label, value in (("деловоден број", number), ("ЕМБС", "7405855"),
                         ("вид на упис", "главна приходна шифра"),
                         ("датум", "17 септ. 2026"), ("големина", "микро"),
                         ("дејност", "01.250")):
        check_true(f"{label} survives into the citable text", value in doc.block.content)
    check_true("tables keep their titles in the citable text",
               "[Дејности]" in doc.block.content)

    # OFFLINE SOURCE: the saved image resolves by its number, nothing else does.
    src = LocalDecisionSource(ASSETS)
    check_true("the saved image resolves by деловоден број",
               src.path_for(number).is_file())
    check_true("an unsaved number does not", not src.path_for("30120260014978").is_file())

    # SCHEMA. Nested models live in $defs, and strict mode holds every one of
    # them to the same rules -- a nested object without additionalProperties
    # false is rejected by the API at call time, not here, unless we check.
    schema = strict_schema(RegistrationDecision)
    objects = [schema, *schema.get("$defs", {}).values()]
    check_true("every nested object is strict", all(
        o.get("additionalProperties") is False
        and sorted(o["required"]) == sorted(o["properties"])
        and "description" not in o for o in objects),
        f"{len(objects)} object schemas")

    description = TOOL_SPEC["function"]["description"]
    check_true("the tool says an ЕМБС is not a деловоден број",
               "НЕ е деловоден број" in description)
    check_true("...and never to invent one", "НИКОГАШ не измислувај" in description)

    # THE GUARD AND A REQUEST FOR INPUT. Given only an ЕМБС, the right reply
    # asks for the number and cites nothing -- and the abstention guard used to
    # replace it with "нема информация". Both halves are checked: the request
    # gets through, and an uncited FACTUAL answer still does not, because the
    # second is the hallucination the guard was built for.
    from .agent import HARD_ABSTENTION
    single = [b.chunk_id for b in c.by_service[2029] if b.type == "tariffs"]
    q = "Дај ми го решението за упис за субјектот со ЕМБС 7405855."
    ask_back = ("За да го пронајдам решението, потребен ми е деловодниот број "
                "(14 цифри). ЕМБС не е доволен.")
    a = CRMAgent(corpus=c, retriever=StubRetriever(single),
                 client=StubClient(ask_back)).ask(q)
    check("an uncited request for the number reaches the user", a.text, ask_back)
    made_up = "За основање на ДОО потребни се Статут и Програма за работа."
    a = CRMAgent(corpus=c, retriever=StubRetriever(single),
                 client=StubClient(made_up)).ask(q)
    check("...while an uncited factual claim is still replaced",
          a.text, HARD_ABSTENTION)


def _search_tool_tests(c) -> None:
    """Resolving an entity to the деловоден број Form 3 needs."""
    print("[announcement search]")
    from pydantic import ValidationError

    from .tools.announcement_search import (DOC_TYPE_CODES, LocalDecisionIndex,
                                            SearchArgs, build_payload,
                                            date_ticks, search_announcements)
    from .tools.fixtures import _FixtureDecisionSource, _FixtureIndex
    from .tools.matching import as_date
    from .tools.registration_decision import fetch_decision

    _as_date = as_date

    def find(**criteria):
        base = {"embs": "", "name": "", "doc_type": "", "date": ""}
        return search_announcements(**{**base, **criteria}, index=_FixtureIndex())

    # ARGUMENTS. A search with no criteria would return the whole archive and
    # read as "these are the entity's filings".
    try:
        find()
        check_true("a search with no criteria is refused", False)
    except ValidationError:
        check_true("a search with no criteria is refused", True)
    for bad, why in (({"embs": "740585"}, "a 6-digit ЕМБС"),
                     ({"date": "2026-09-17"}, "an ISO date")):
        try:
            find(**bad)
            check_true(f"{why} is refused", False)
        except ValidationError:
            check_true(f"{why} is refused", True)

    # DATES. The decision prints "17 септ. 2026"; the user types "17.09.2026".
    check("an abbreviated Macedonian month normalises", _as_date("17 септ. 2026"),
          "17.09.2026")
    check("...as does the full month name", _as_date("1 септември 2026"),
          "01.09.2026")
    check("...and a numeric date", _as_date("7.9.2026"), "07.09.2026")
    check("a non-date is not a date", _as_date("По службена должност"), None)

    # RESOLUTION, which is the whole point: entity in, filing number out.
    for label, criteria in (("ЕМБС", {"embs": "7405855"}),
                            ("zero-padded ЕМБС", {"embs": "07405855"}),
                            ("name", {"name": "тополчан"}),
                            ("date", {"date": "17.09.2026"}),
                            ("name and date", {"name": "ТОПОЛЧАН",
                                               "date": "17.09.2026"})):
        hits = [h.deloveden_broj for h in find(**criteria).hits]
        check(f"resolves by {label}", hits, ["30120260014967"])

    # AND, not OR. Returning everything named ТОПОЛЧАН for a search that also
    # named a date is a wrong answer in the shape of a right one.
    check("every criterion must hold",
          find(name="ТОПОЛЧАН", date="01.01.2020").hits, [])
    check("...including the kind of decision",
          find(embs="7405855", doc_type="ликвидација").hits, [])

    # THE LIVE PAYLOAD. Not called, but reproduced exactly from the capture --
    # including the date, which the portal sends as local midnight in Skopje
    # expressed in UTC ticks. A naive conversion is two hours out in September
    # and would silently ask about the previous day.
    check("the captured request is reproduced", build_payload(SearchArgs(
        embs="7696876", name="", doc_type="", date="10.09.2026")),
        {"LEID": 7696876, "partOfName": None, "docTypeID": None,
         "announcementDateTicks": 639245880000000000, "ut": False})
    # How far the sent value sits from treating local midnight AS UTC: two
    # hours in summer, one in winter. A fixed offset would be wrong for five
    # months of the year, and wrong by a whole day at the boundary.
    from datetime import datetime as _dt
    HOUR = 36_000_000_000

    def naive(d):
        return int((_dt.strptime(d, "%d.%m.%Y")
                    - _dt(1, 1, 1)).total_seconds()) * 10_000_000
    check("a September date is sent as UTC+2",
          naive("10.09.2026") - date_ticks("10.09.2026"), 2 * HOUR)
    check("a January date is sent as UTC+1",
          naive("15.01.2026") - date_ticks("15.01.2026"), HOUR)
    check("a Macedonian entry type maps to the form's code",
          build_payload(SearchArgs(embs="", name="x", doc_type="ликвидација",
                                   date=""))["docTypeID"], 201)
    check_true("an unknown entry type is sent as null, not guessed",
               build_payload(SearchArgs(embs="", name="x",
                                        doc_type="нешто друго",
                                        date=""))["docTypeID"] is None)
    check_true("the code table is marked incomplete in the source",
               len(DOC_TYPE_CODES) == 5)

    # NOTHING FOUND is a result, and must not overstate itself: we searched our
    # archive, not the registry.
    empty = find(embs="7696876")
    check("an entity with nothing saved finds nothing", empty.found, False)
    text = empty.as_context_doc().block.content
    check_true("...and the reply says the search was of the offline archive",
               "офлајн архивата" in text)
    check_true("...and points at the portal", "порталот" in text)
    check_true("...and never claims the entity HAS no decisions",
               "нема решенија" not in text and "нема објави" not in text)
    check_true("an empty search is still citable",
               empty.as_context_doc().chunk_id.startswith("tool:announcement_search:"))

    # THE INVARIANT: anything found can then be fetched. A resolver that hands
    # back a number Form 3 cannot open has only moved the failure one step.
    for hit in find(embs="7405855").hits:
        got = fetch_decision(hit.deloveden_broj, source=_FixtureDecisionSource())
        check_true(f"{hit.deloveden_broj} is fetchable", getattr(got, "available", False))

    # The live index reads its numbers from filenames -- no vision call to list.
    # Asserted as membership, not equality: the operator adds saved documents
    # to assets/ as they go, and a gate that fails when they do teaches people
    # to ignore it.
    numbers = LocalDecisionIndex().numbers()
    check_true("the offline index sees the saved decisions",
               "30120260014967" in numbers, str(numbers))

    # A TRANSIENT MISREAD. Measured on a real founding decision: the same image
    # at temperature 0 transposed two digits of its 14-digit number in one read
    # out of four, and the cross-check inside the document did NOT catch it --
    # the field and the table row were misread identically. The filename is the
    # independent claim that does catch it, so a disagreement triggers one
    # re-read; a disagreement that survives means the file cannot be identified.
    import agent.tools.registration_decision as _rd
    from .tools.fixtures import RECORDED_DECISIONS as _REC

    real = _rd.extract_decision
    number = "30120260014967"
    good = _REC[number]
    bad = good.model_copy(update={"deloveden_broj": "30120206014967"})

    def flaky(reads):
        """Returns each read in turn, then repeats the last one. The index
        walks every saved decision, so a fixed-length script runs out."""
        pending = list(reads)

        def fake(image_bytes, **kw):
            return pending.pop(0) if len(pending) > 1 else pending[0]
        return fake

    index = LocalDecisionIndex()
    try:
        _rd.extract_decision = flaky([bad, good])
        got = _rd.LocalDecisionSource().load(number)
        check("a misread number is re-read, and the good read wins",
              got.deloveden_broj, number)

        _rd.extract_decision = flaky([bad, bad])
        index.source = _rd.LocalDecisionSource()
        listed = [d.deloveden_broj for d in index.all_decisions()
                  if d.deloveden_broj == number]
        check("a file that stays unidentifiable is not offered by search",
              listed, [])
        check_true("...and is reported, not silently dropped",
                   number in index.rejected, str(index.rejected))
    finally:
        _rd.extract_decision = real

    check_true("the tool warns against inventing a number",
               "НЕ измислувај" in
               __import__("agent.tools.announcement_search", fromlist=["x"])
               .TOOL_SPEC["function"]["description"])


def _entity_search_tests(c) -> None:
    """Resolving a company NAME to the ЕМБС Form 2 needs."""
    print("[entity search]")
    from pydantic import ValidationError

    from .tools.entity_profile import fetch_profile
    from .tools.entity_search import (LocalEntityIndex, search_entity_profile)
    from .tools.fixtures import _FixtureEntityIndex, _FixtureSource

    def find(**criteria):
        base = {"name": "", "embs": ""}
        return search_entity_profile(**{**base, **criteria},
                                     index=_FixtureEntityIndex())

    try:
        find()
        check_true("a search with neither name nor ЕМБС is refused", False)
    except ValidationError:
        check_true("a search with neither name nor ЕМБС is refused", True)

    # RESOLUTION. Partial, case-insensitive, and across the whole legal name --
    # users type the trading name, the registry stores the full one.
    for label, name in (("the full trading name", "ЛОРА КОМПАНИ"),
                        ("lowercase", "лора компани"),
                        ("one word", "ЛОРА"),
                        ("a name buried in the legal form", "КОМПАНИ 2023")):
        check(f"resolves {label}", [h.embs for h in find(name=name).hits],
              ["7696876"])

    # An entity known only from a DECISION. It has no saved profile, so the
    # search must find it and say so -- the alternative is the agent resolving
    # a name, fetching, and hitting "not available" a round later.
    topolchan = find(name="ТОПОЛЧАН АГРАР")
    check("a decision names an entity no profile covers",
          [h.embs for h in topolchan.hits], ["7405855"])
    check("...flagged as having no saved profile",
          [h.has_profile for h in topolchan.hits], [False])
    check("...and attributed to the decision",
          [h.seen_in for h in topolchan.hits], [["решение"]])
    check_true("...which the citable text states",
               "не е зачуван" in topolchan.as_context_doc().block.content)
    # And the flag is true: fetching it really does come back unavailable.
    got = fetch_profile("7405855", source=_FixtureSource())
    check("...matching what a fetch would return", got.available, False)

    check("a profile-backed entity is flagged as available",
          [h.has_profile for h in find(name="ЛОРА").hits], [True])

    # NOTHING FOUND, with the same scope discipline as announcement_search.
    missing = find(name="НЕПОСТОЕЧКА ФИРМА")
    check("an unknown name finds nothing", missing.found, False)
    text = missing.as_context_doc().block.content
    check_true("...and says the archive was searched, not the registry",
               "офлајн архивата" in text and "порталот" in text)
    check_true("...and never claims the entity does not exist",
               "не постои" not in text.replace("може да постои", ""))
    check_true("an empty entity search is still citable",
               missing.as_context_doc().chunk_id.startswith("tool:entity_search:"))

    # THE INVARIANT, as for announcements: what is found can then be fetched.
    for hit in find(name="ЛОРА").hits:
        got = fetch_profile(hit.embs, source=_FixtureSource())
        check_true(f"{hit.embs} is fetchable", got.available)

    # The live index lists ЕМБС values from filenames, with no vision call.
    numbers = LocalEntityIndex().profile_numbers()
    check_true("the offline index sees the saved profiles",
               "7696876" in numbers and len(numbers) >= 3, f"{numbers}")


def _status_tool_tests(c) -> None:
    """Form 4: Статус на предмет. Text in, no vision anywhere."""
    print("[status info]")
    from pydantic import ValidationError

    from .tools.fixtures import RECORDED_STATUS, _FixtureStatusSource
    from .tools.status_info import (TOOL_SPEC, LocalStatusSource, StatusArgs,
                                    StatusResult, StatusUnavailable,
                                    fetch_status, parse_infobox, parse_status)

    number = "30120260014978"

    check("a 14-digit деловоден број is accepted",
          StatusArgs(document_id=f" {number} ").document_id, number)
    for wrong, why in (("7405855", "an ЕМБС"), ("3012026001497", "13 digits")):
        try:
            StatusArgs(document_id=wrong)
            check_true(f"{why} is refused", False)
        except ValidationError:
            check_true(f"{why} is refused", True)

    # THE PARSER, against the structure the portal renders. Labels are matched,
    # not positions, so an extra row or a reordering does not shift every field
    # by one -- which, for a box whose last line is the STATUS, would mean
    # reporting a date as the outcome of someone's application.
    html = ('<div class="infobox"><h2>ТЕСТ ДОО Скопје</h2>'
            '<h3>Предметни податоци:</h3>'
            '<p>Датум на објава: <b>18.9.2026 18:45</b></p>'
            '<p>Опис на документ: <b>Упис на основање</b></p>'
            '<p>Статус: <b>Одлучен - одобрен</b></p></div>')
    parsed = parse_infobox(html)
    check("the entity name comes off the heading", parsed.entity_title,
          "ТЕСТ ДОО Скопје")
    check("the publication date is read", parsed.publication_date, "18.9.2026 18:45")
    check("the document description is read", parsed.document_description,
          "Упис на основање")
    check("the status is read", parsed.status, "Одлучен - одобрен")

    reordered = ('<div class="infobox"><h2>ТЕСТ ДОО Скопје</h2>'
                 '<p>Статус: <b>Одлучен - одобрен</b></p>'
                 '<p>Нешто ново: <b>вредност</b></p>'
                 '<p>Датум на објава: <b>18.9.2026 18:45</b></p></div>')
    shuffled = parse_infobox(reordered)
    check("a reordered box with an unknown row still reads correctly",
          (shuffled.status, shuffled.publication_date),
          ("Одлучен - одобрен", "18.9.2026 18:45"))
    check("...and the absent field is empty, not shifted",
          shuffled.document_description, "")

    # The saved file may be the API's JSON instead of the page.
    api = parse_status('{"title": "ТЕСТ ДОО", "publishDate": "18.9.2026 18:45",'
                       ' "documentDescription": "Упис на основање",'
                       ' "status": "Одлучен - одобрен"}')
    check("the portal's own JSON field names are accepted",
          (api.entity_title, api.status), ("ТЕСТ ДОО", "Одлучен - одобрен"))

    # RESULTS.
    got = fetch_status(number, source=_FixtureStatusSource())
    check_true("a held filing returns its status", isinstance(got, StatusResult))
    doc = got.as_context_doc()
    check("the status is citable", doc.chunk_id, f"tool:status_info:{number}")
    for value in (number, "Одлучен - одобрен", "Упис на основање", "18.9.2026 18:45",
                  "БЕКАТОН ЕЛИТ ТП Пробиштип"):
        check_true(f"{value!r} survives into the citable text",
                   value in doc.block.content)
    untitled = StatusResult(
        document_id=number, fetched_at="x",
        info=RECORDED_STATUS[number].model_copy(update={"entity_title": ""}))
    check_true("an empty field prints no bare label",
               "Субјект:" not in untitled.as_context_doc().block.content)

    miss = fetch_status("30120260014967", source=_FixtureStatusSource())
    check_true("an unheld filing returns a value, not an exception",
               isinstance(miss, StatusUnavailable))
    check("...flagged unavailable", miss.available, False)
    check("...and citable", miss.as_context_doc().chunk_id,
          "tool:status_info:30120260014967:unavailable")
    check_true("...in wording the abstention guard recognises",
               "не е достапен" in miss.as_context_doc().block.content)

    # PRODUCTION vs GATES. The eval replays RECORDED_STATUS; the CLI reads
    # assets/. They diverged once -- the eval scored this filing 1.000 while the
    # CLI reported it unavailable, because no asset had been saved -- and the
    # eval could not notice, since it never touches the production source.
    # Where a filing exists in both, they must say the same thing.
    saved = LocalStatusSource().load(number)
    check_true("the production source serves the recorded filing", saved is not None)
    check("...with exactly the recorded fields", saved, RECORDED_STATUS[number])

    description = TOOL_SPEC["function"]["description"]
    check_true("the tool routes an ЕМБС through the search first",
               "search_announcements" in description)
    check_true("...and forbids inventing a number",
               "НИКОГАШ не измислувај" in description)


def _orchestration_tests(c) -> None:
    print("[orchestration]")
    tariffs = [b.chunk_id for b in c.by_service[2135] if b.type == "tariffs"][:5]
    retriever = StubRetriever(tariffs)
    client = StubClient("За каква правна форма станува збор?")
    agent = CRMAgent(corpus=c, retriever=retriever, client=client)

    a1 = agent.ask("Колку чини регистрација?")
    check_true("turn 1 detects ambiguity", a1.asked_for_clarification)
    # GATE 3 is structural: an ambiguous turn never reaches the model, so there
    # is no answer, no coverage hedge and no menu-under-an-answer to leak. Live,
    # the model produced all three at once -- a hallucinated document list, a
    # hedge saying the list was incomplete, and the options underneath.
    check("an ambiguous turn does not call the model", len(client.calls), 0)
    check("...so it costs nothing", a1.cost_usd, 0.0)
    check("...and cites nothing", a1.citations, [])
    check_true("...and the reply is the question itself",
               a1.text.startswith("За каква правна форма") and "- АД" in a1.text,
               a1.text.splitlines()[0])

    # Turn 2: the reply alone ("АД") is meaningless as a query -- it must be
    # combined with the original question and turned into a filter.
    client.reply = "Тарифата е 2452 МКД [%s]." % tariffs[0]
    a2 = agent.ask("АД")
    q2, f2 = retriever.queries[-1]
    check_true("clarification is merged into the retrieval query",
               "Колку чини регистрација?" in q2 and "АД" in q2, q2)
    check("clarification becomes a variation filter",
          (f2 or {}).get("variation_scope") is not None, True)
    # The service was settled when we asked; the reply only picks a form. Live,
    # answering "АД" to a clarification about вистински сопственик (2183) sent
    # the search after "АД" on its own and returned incorporation-of-AD (2140).
    check("...and pins the service we asked about", (f2 or {}).get("service_scope"), 2135)
    check_true("resolved turn is no longer ambiguous", not a2.asked_for_clarification)
    check("history carries both turns", len(agent.history), 4)
    # The message shape is asserted on the first turn that actually calls the model.
    check_true("system rules go first",
               client.calls[0][0]["role"] == "system"
               and "ALWAYS respond in Macedonian" in client.calls[0][0]["content"])
    check_true("documents go in their own system message",
               any(m["role"] == "system" and "<documents>" in m["content"]
                   for m in client.calls[0][1:]))
    check_true("the user turn is last", client.calls[0][-1]["role"] == "user")
    # Priced from the table for whichever model is the default, so switching it
    # is a one-line change and not a test edit as well.
    in_price, out_price = PRICING[agent.model]
    check("cost is accounted", round(a2.cost_usd, 8),
          round((100 * in_price + 20 * out_price) / 1_000_000, 8))

    # A long answer must not survive into history as a data source. Live, a
    # 36-agent list from turn 1 was still there six turns later and the model
    # built a non-existent lawyer out of it -- a Струмица name with a Гостивар
    # address -- citing nothing.
    from .agent import HISTORY_ANSWER_CHARS, _for_history
    long_answer = "\n".join(f"{i}. АДВОКАТ ИМЕ {i} | адреса: Ул. X Бр.{i}"
                            for i in range(1, 40))
    trimmed = _for_history(long_answer)
    check_true("a long answer is trimmed before entering history",
               len(trimmed) < len(long_answer) and "повторно да се побараат" in trimmed,
               f"{len(long_answer)} -> {len(trimmed)} chars")
    check("a short answer is kept whole", _for_history("Кратко."), "Кратко.")

    client3 = StubClient(long_answer)
    agent3 = CRMAgent(corpus=c, retriever=StubRetriever(tariffs), client=client3)
    agent3.ask("Кои се агентите во Гостивар?")
    check_true("history holds the trimmed version, not the full list",
               len(agent3.history[-1]["content"]) <= HISTORY_ANSWER_CHARS + 120,
               f"{len(agent3.history[-1]['content'])} chars")

    agent.reset()
    check("reset clears history", len(agent.history), 0)

    # End to end on the real failure: the same five blocks the live agent got.
    observed = ["srv_2155_v0_terminology_5", "srv_2135_shared_terminology_2",
                "srv_2135_v11116_documents_17", "srv_2194_shared_faq_1",
                "srv_2183_shared_faq_42"]
    client2 = StubClient("За каква правна форма станува збор?")
    agent2 = CRMAgent(corpus=c, retriever=StubRetriever(observed), client=client2)
    a = agent2.ask("Колку чини регистрација?")
    check_true("live-failure query now asks for clarification",
               a.asked_for_clarification and a.ambiguity.source == "structure")
    # The structure detector fires precisely when the differing rows may be
    # absent from the context, so there is nothing to answer from and the model
    # is not asked. The user still gets every form to choose between.
    check("the structure-detected turn does not call the model", len(client2.calls), 0)
    check_true("...and every legal form is offered",
               all(f"- {lbl}" in a.text for lbl in ("Здружение", "Фондација")),
               a.text.replace("\n", " ")[:90])
    # And the follow-up still resolves to a filtered retrieval.
    client2.reply = "Тарифата е 0 МКД [srv_2135_v11113_tariffs_1]."
    a2 = agent2.ask("АД")
    q, f = agent2.retriever.queries[-1]
    check_true("follow-up filters to the named form",
               (f or {}).get("variation_scope") == 11113 and "регистрација" in q,
               f"{f} | {q}")


def _structured_fetch_tests(c) -> None:
    """The layer that fixed the залог answer.

    Live, the agent produced a confident four-step procedure for
    "Како да регистрирам залог?" while the eval scored hit@10 = 0.000 -- not one
    of the registry's six `process` rows had been retrieved, so three steps
    (receipt, processing, approval) were silently missing from a grounded,
    fully-cited answer.
    """
    print("[structured fetch]")
    from eval.baselines import BM25Retriever
    from index.intent import detect_intent
    from index.structured import SectionFetchRetriever, attribute, fetch_sections

    r = SectionFetchRetriever(BM25Retriever(c), c)

    hits = r.search("Како да регистрирам залог?", 10)
    got = [h.chunk_id for h in hits]
    process = [b.chunk_id for b in c.of_type(2063, "process", 11187)]
    check_true("the залог answer rows are fetched, not searched for",
               any(cid in got for cid in process),
               f"{sum(1 for cid in process if cid in got)}/{len(process)} steps in top 10")
    check_true("all six steps are reachable",
               set(process) <= set(h.chunk_id for h in r.search("Како да регистрирам залог?", 40)),
               f"{len(process)} steps")

    # Precedence: a pickup question must not pull in three wrong sections.
    check("specific cue beats vague interrogatives",
          detect_intent("Каде и како го подигнувам документот?", top_only=True),
          ["documentsLocations"])
    check_true("a service name alone is not a procedure question",
               "process" not in detect_intent("Самостојна пријава за упис на основање"))

    # Refusals: each returns nothing rather than something wrong.
    no_intent = fetch_sections(c, "Информации за субјект", hits)
    check("no readable intent -> no injection", no_intent, [])

    # 2135 has nine legal forms; without knowing which, injecting fees would be
    # picking one at random (АД is 0 МКД, most others are 2452 МКД).
    amb_hits = [Hit("srv_2135_v11116_documents_17", 1.0),
                Hit("srv_2135_shared_terminology_2", 0.9)]
    check("unpinned legal form -> no injection",
          fetch_sections(c, "Колку чини регистрација?", amb_hits), [])
    # ...but an explicit filter pins it.
    pinned = fetch_sections(c, "Колку чини регистрација?", amb_hits,
                            filters={"variation_scope": 11116})
    check_true("an explicit variation filter unlocks the fetch", len(pinned) >= 1,
               str(pinned[:2]))

    # A single hit must not attribute a variation of a multi-form service.
    solo_vote = attribute([Hit("srv_2135_v11116_documents_17", 1.0)], c)
    check("one hit is not enough to pick a legal form",
          (solo_vote.id_service, solo_vote.id_variation), (2135, None))
    two_votes = attribute([Hit("srv_2135_v11116_documents_17", 1.0),
                           Hit("srv_2135_v11116_documents_1", 0.9)], c)
    check("two agreeing hits are",
          (two_votes.id_service, two_votes.id_variation), (2135, 11116))

    # Coverage: the model must be able to tell a whole section from a fragment.
    from .context import render_coverage, section_coverage
    whole = build_documents(_hits([b.chunk_id for b in c.of_type(2063, "process", 11187)]), c)
    cov = section_coverage(whole, c)
    check("a fully fetched section reports COMPLETE",
          [(t, p_, tot) for t, _, p_, tot in cov], [("process", 6, 6)])

    # Observed live: one of six `documents` rows, presented under a heading that
    # implied the full list.
    fragment = build_documents(_hits(["srv_2063_v11187_documents_3"]), c)
    cov = section_coverage(fragment, c)
    check_true("a fragment reports PARTIAL with the true total",
               cov and cov[0][2] == 1 and cov[0][3] > 1, str(cov))
    check_true("the warning reaches the prompt",
               "PARTIAL" in render_coverage(fragment, c))
    faq_only = build_documents(_hits(["srv_2063_shared_faq_3"]), c)
    check("lookup sections are not reported as partial",
          section_coverage(faq_only, c), [])

    # --- agent directory ------------------------------------------------- #
    from index.structured import detect_municipalities, fetch_directory

    both = fetch_directory(c, "Кои се овластените агенти во Гостивар?")
    lists = {c.by_id[cid].list_key for cid in both}
    check("a municipality question returns both agent lists", lists,
          {"advocates", "doo_tp"})
    check_true("and only that municipality",
               all(c.by_id[cid].municipality == "ГОСТИВАР" for cid in both))

    # A municipality named in a SERVICE question is not a directory request.
    check("municipality without an agent cue -> no directory",
          fetch_directory(c, "Колку чини регистрација во Гостивар?"), [])

    # Skopje is ten municipalities; the literal bucket alone is a third of them.
    skopje = detect_municipalities("адвокати агенти во Скопје", c)
    check_true("Скопје expands to its municipalities", len(skopje) >= 8,
               f"{len(skopje)}: {', '.join(skopje[:4])}...")

    # Homoglyph cleanup: the advocates sheet spells one Аеродром with a Latin M.
    check("Аеродром is one bucket, not two",
          sum(1 for m in c.by_municipality if m.startswith("АЕРОДР")), 1)

    # Oversized municipalities must announce that they are a fragment.
    centar = [b for b in c.by_municipality["ЦЕНТАР"] if b.list_key == "advocates"]
    check_true("a split municipality states its part and true total",
               any("дел 1 од" in b.content and str(b.n_agents) in b.content
                   for b in centar),
               f"{len(centar)} parts, {centar[0].n_agents} agents")

    # A fetched section must survive the trim to k distinct documents. Live,
    # five of six pledge-deletion steps reached the model and it correctly
    # reported that one was missing -- the section was fetched whole and then
    # cut apart by semantic competition in the fused ranking.
    steps = [b.chunk_id for b in c.of_type(2115, "process", 11125)]
    noise = [b.chunk_id for b in c.by_service[2183]][:5]
    interleaved = _hits([steps[0], noise[0], steps[1], noise[1], steps[2],
                         noise[2], steps[3], noise[3], steps[4], noise[4],
                         steps[5]])
    # limit=10 is what the agent actually uses (DEFAULT_K), and at that size this
    # fixture reproduces the live failure exactly: five of the six steps arrive.
    plain = build_documents(interleaved, c, limit=10)
    kept = build_documents(interleaved, c, limit=10, pin=steps)
    check("without pinning the section tail is trimmed away",
          sum(1 for d in plain if d.chunk_id in steps), 5)
    check("pinning keeps every fetched row",
          sum(1 for d in kept if d.chunk_id in steps), len(steps))
    check("...without exceeding the limit", len(kept), 10)
    # ...but injection can never take the whole context: the semantic arm is
    # what recovers a wrong attribution.
    flood = [b.chunk_id for b in c.of_type(2135, "documents", 11117)][:12]
    crowded = build_documents(_hits(flood + noise), c, limit=10, pin=flood)
    check("injection leaves room for the semantic arm",
          sum(1 for d in crowded if d.chunk_id not in set(flood)), 3)
    # Rank order across sections, registry order WITHIN one. Sorting purely by
    # fused rank handed the model a numbered procedure as 4, 1, 5, 2, 3, and it
    # answered 1, 2, 4, 5 -- losing a step while reassembling the sequence.
    check("a procedure reads in its own order",
          [d.chunk_id for d in kept if d.chunk_id in steps], steps)

    # The budget keeps injection from taking every slot at small k.
    r.search("Кои документи се потребни?", 10)
    check_true("injection is capped so semantic keeps half the slots",
               len(r.last_sections) <= 6, f"{len(r.last_sections)} rows at k=10")


def _injection_tests(c) -> None:
    """Corpus text must never restructure the context it is rendered into.

    Offline half of agent/injection.py -- the behavioural half needs a model
    call and lives behind `python -m agent.injection --live`.
    """
    print("[prompt injection -- structural]")
    from .injection import PAYLOADS, check_structural
    for r in check_structural(c):
        check_true(f"{r.payload} is neutralised", r.ok, r.detail)
    check_true("every payload declares why it exists",
               all(p.why for p in PAYLOADS), f"{len(PAYLOADS)} payloads")


def main() -> int:
    _failures.clear()
    c = load_corpus()
    print(f"corpus: {len(c)} blocks, sha={c.sha256[:12]}\n")
    for stage in (_dedup_tests, _ambiguity_tests, _structure_ambiguity_tests,
                  _structured_fetch_tests, _citation_tests, _clarification_tests,
                  _injection_tests, _tool_tests, _profile_tool_tests,
                  _decision_tool_tests, _search_tool_tests,
                  _entity_search_tests, _status_tool_tests,
                  _orchestration_tests):
        try:
            stage(c)
        except Exception:
            traceback.print_exc()
            _failures.append(stage.__name__)
    print()
    if _failures:
        print(f"SELFTEST FAILED: {len(_failures)} check(s): {', '.join(_failures)}")
        return 1
    print("SELFTEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
