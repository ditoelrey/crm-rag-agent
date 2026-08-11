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

from .agent import CRMAgent, is_followup, match_variation
from .context import (build_context, build_documents, detect_ambiguity,
                      detect_intent, render_ambiguity, validate_citations)

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
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model, messages, temperature):
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
              len(amb.variations), 9)

    # Same documents, but the user already said which form.
    amb2 = detect_ambiguity(docs, "Колку чини упис на основање на Здружение?", c)
    check_true("does not fire when the user named the form", amb2 is None)

    # All five variations share ONE identical access block -> dedup collapses
    # them, so there is nothing to disambiguate.
    access = [b.chunk_id for b in c.by_service[2111] if b.type == "access"]
    docs3 = build_documents(_hits(access), c)
    amb3 = detect_ambiguity(docs3, "Дали годишна сметка може онлајн?", c)
    check("identical access blocks collapse to one", len(docs3), 1)
    check_true("does not fire when the forms agree", amb3 is None)

    # A single-variation service is never ambiguous.
    solo = [b.chunk_id for b in c.by_service[2162] if b.type == "tariffs"]
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
        check_true("offers every legal form", len(amb.variations) == 9,
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
    # "ПДОО" is the label; nobody asks for a ПДОО, they ask for a ДОО.
    pdoo = match_variation("ме интересира ДОО", amb.variations, c)
    check_true("ДОО resolves to the ПДОО variation",
               pdoo is not None and pdoo[1] == "ПДОО", str(pdoo))

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


def _orchestration_tests(c) -> None:
    print("[orchestration]")
    tariffs = [b.chunk_id for b in c.by_service[2135] if b.type == "tariffs"][:5]
    retriever = StubRetriever(tariffs)
    client = StubClient("За каква правна форма станува збор?")
    agent = CRMAgent(corpus=c, retriever=retriever, client=client)

    a1 = agent.ask("Колку чини регистрација?")
    check_true("turn 1 detects ambiguity", a1.asked_for_clarification)
    check_true("turn 1 sends system rules first",
               client.calls[0][0]["role"] == "system"
               and "ALWAYS respond in Macedonian" in client.calls[0][0]["content"])
    check_true("documents go in their own system message",
               any(m["role"] == "system" and "<documents>" in m["content"]
                   for m in client.calls[0][1:]))
    check_true("the user turn is last", client.calls[0][-1]["role"] == "user")
    check("cost is accounted", round(a1.cost_usd, 8),
          round((100 * 0.15 + 20 * 0.60) / 1_000_000, 8))

    # Turn 2: the reply alone ("АД") is meaningless as a query -- it must be
    # combined with the original question and turned into a filter.
    client.reply = "Тарифата е 2452 МКД [%s]." % tariffs[0]
    a2 = agent.ask("АД")
    q2, f2 = retriever.queries[-1]
    check_true("clarification is merged into the retrieval query",
               "Колку чини регистрација?" in q2 and "АД" in q2, q2)
    check("clarification becomes a variation filter",
          (f2 or {}).get("variation_scope") is not None, True)
    check_true("resolved turn is no longer ambiguous", not a2.asked_for_clarification)
    check("history carries both turns", len(agent.history), 4)

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
    check_true("the model is told the forms and that rows may be missing",
               any("<ambiguity>" in m["content"] and "may contain none of them"
                   in m["content"] for m in client2.calls[0] if m["role"] == "system"))
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
                  _injection_tests, _orchestration_tests):
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
