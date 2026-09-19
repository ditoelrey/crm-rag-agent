"""
curated.py  --  hand-authored evaluation cases in real user phrasing.
=====================================================================
The synthetic families measure retrieval mechanics; this file is the only place
that measures whether the system answers the questions people actually ask. Every
entry below was written as a question first, then its gold was located in the
corpus and read to confirm it genuinely answers it (verified 2026-07-28).

Gold is expressed as a *rule* (service + variation + section type, or explicit
chunk ids) rather than a frozen id list, so a corpus rebuild that renumbers rows
re-resolves instead of silently rotting. `expand()` raises if a rule matches
nothing -- an unanswerable case must fail loudly, not score 0 forever.

Adding cases
------------
Append a Spec. Keep the query in the user's words (typos and all, if that is
what they type), point `service`/`variation` at the answer, and put the reason
in `notes`. 15 honest cases beat 500 generated ones; grow this file whenever a
real question is seen to fail.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .corpus import Corpus
from .goldset import EvalCase, _support_grades


@dataclass
class Spec:
    query: str
    intent: str                       # names the family: curated_<intent>
    notes: str
    service: int | None = None
    variation: int | None = None
    types: tuple[str, ...] = ()       # gold = all rows of these types
    chunk_ids: tuple[str, ...] = ()   # ...or these exact blocks
    also_services: tuple[int, ...] = ()   # additional acceptable services (grade 1)
    # Additional primary gold as (id_service, id_variation|None, types) -- for
    # questions a user could legitimately mean about more than one service.
    plus: tuple[tuple[int, int | None, tuple[str, ...]], ...] = ()
    difficulty: str = "hard"
    pin_provenance: bool = False      # see _COPY_TOLERANT_TYPES
    behavior: str | None = None       # "answer" | "clarify" | "abstain"
    # Agent-directory gold, as (municipality, list|None) selectors rather than
    # pinned chunk_ids -- part counts shift whenever the registry republishes a
    # list, and gold that rots looks exactly like a retriever bug.
    directory: tuple[tuple[str, str | None], ...] = ()
    values: tuple[str, ...] = ()      # must appear in the answer
    forbid: tuple[str, ...] = ()      # must NOT appear -- usually a near-miss
    # A TOOL-BOUNDARY case: graded on whether a live lookup happens, not on what
    # was retrieved. Such a case has no corpus gold by definition -- the fact is
    # fetched rather than stored, or (for a must-not-call case) there is no fact
    # to fetch at all -- so it is exempt from the "every answerable case resolves
    # to blocks" rule, and the retrieval run skips it like any gold-less case.
    live_tool: bool = False
    # Multi-turn: `query` is turn 1, `followups` are the rest. `values`/`forbid`
    # are asserted on the FINAL answer; `turn_behaviors` grades each turn.
    followups: tuple[str, ...] = ()
    turn_behaviors: tuple[str, ...] = ()


# Types whose text is portal boilerplate, copied verbatim under many services.
# When gold of one of these types is pinned, every byte-identical copy is graded
# primary too: the question ("Што значи ПКД?") names no service, so any copy
# answers it. Set `pin_provenance=True` on a Spec where the asking service
# genuinely matters and only its own copy should count.
_COPY_TOLERANT_TYPES = frozenset({"terminology", "faq", "legalBasis"})


# --------------------------------------------------------------------------- #
# The cases.
# --------------------------------------------------------------------------- #
SPECS: list[Spec] = [
    Spec(query="Сакам сам да пријавам основање на здружение, кои документи ми се потребни?",
         values=('статут', 'акт за основање', 'записник'),
         intent="documents", service=2135, variation=11116, types=("documents",),
         notes="Здружение is one of 9 sibling variations of 2135; the document "
               "list differs per variation, so a sibling hit is a wrong answer."
               " Все three are among the 10 document rows UNIQUE to Здружение (not shared with АД), so an answer that pulled a sibling's list fails on them."),

    Spec(query="Колку чини упис на основање на фондација?",
         values=('2452',),
         intent="tariffs", service=2135, variation=11117, types=("tariffs",),
         notes="2452 МКД, variation-specific tariff row."),

    Spec(query="За колку време се одобрува пријава за упис на основање на АД?",
         values=('4 часа',),
         intent="deadlines", service=2135, variation=11113, types=("deadlines",),
         notes="4 часа for approval; the variation has 4 different deadline rows."),

    Spec(query="Колку чини потврда за тековна состојба на фирма?",
         values=('295',), forbid=('299',),
         intent="tariffs", service=2162, variation=11050, types=("tariffs",),
         notes="295 МКД. Competing service 2185 has a near-identical 299 МКД "
               "tariff for a different certificate -- a real confusion pair."),

    Spec(query="Каде ја подигам потврдата за тековна состојба?",
         values=('шалтер',),
         intent="access", service=2162, variation=11050,
         types=("documentsLocations", "access"),
         notes="Counter pickup at ЦРРСМ offices."
               " The pickup row says 'На шалтерите на ЦРРСМ'. Matching the stem catches шалтер / шалтерите / шалтерот."),

    Spec(query="Дали годишна сметка може да се поднесе преку интернет и на кој линк?",
         values=('e-submit.crm.com.mk',), behavior="answer",
         intent="access", service=2111, types=("access",),
         notes="Underspecified on purpose: the user did not say which subject "
               "type, and all five variations share the same e-submit portal, "
               "so any variation's access block is a correct answer."
               " The question asks for the link, so the answer must carry it. A URL cannot be paraphrased away, which makes it the sharpest possible assertion."),

    Spec(query="Кој е крајниот рок за поднесување годишна сметка за банка?",
         values=('31.12|31 декември',), behavior="answer",
         intent="deadlines", service=2111, variation=11176, types=("deadlines",),
         notes="PRECISION case. The variation has two deadline rows and the "
               "corpus names them differently: 15.3. is the ЗАКОНСКИ рок, "
               "31.12. the КРАЕН рок. This question asks for the крајниот, so "
               "31.12. alone is the right answer -- an earlier version of this "
               "case demanded both values and failed a correct answer."),

    Spec(query="Кои се роковите за поднесување годишна сметка за банка?",
         values=('15.3|15 март', '31.12|31 декември'), behavior="answer",
         intent="deadlines", service=2111, variation=11176, types=("deadlines",),
         notes="COMPLETENESS case, the sibling of the one above. Asked for the "
               "rokovi in the plural, both rows are required: missing the 15.3. "
               "legal deadline carries a consequence, so an answer that gives "
               "only the final date is incomplete."),

    Spec(query="Во кој рок морам да пријавам промена на вистински сопственик?",
         values=('15 дена',),
         intent="deadlines", service=2183, variation=11122,
         chunk_ids=("srv_2183_v11122_deadlines_2",),
         notes="15 days from the change. Pinpoint gold: the sibling deadline "
               "rows in the same variation are about other obligations."),

    Spec(query="Што значи ПКД?",
         values=('забрана', 'кривич'),
         intent="terminology", service=2075, chunk_ids=("srv_2075_shared_terminology_2",),
         notes="Abbreviation lookup. The definition is byte-identical under 16 "
               "sanctions-register services and the question names none of "
               "them, so every copy counts as correct."
               " ПКД covers two registers: sanctions banning a profession/activity, and penalties for criminal offences by legal persons. Both must appear. 'кривич' is a prefix on purpose -- it matches кривични / кривично without demanding one grammatical form."),

    Spec(query="Колку чини тендерско досие со економско-финансиска состојба?",
         values=('8710',), forbid=('5140',),
         intent="tariffs", service=2059, variation=10947, types=("tariffs",),
         notes="8710 МКД. 2051 is the cheaper package WITHOUT the financial "
               "part (5140 МКД) -- picking it is a wrong, expensive answer."),

    Spec(query="Како се регистрира залог врз машина?",
         values=('нотар', 'потврда за прием', 'потврда за упис'),
         intent="process", service=2063, variation=11187, types=("process",),
         notes="6-step notary-led procedure; 2113/2115 (change/deletion of a "
               "pledge) are the near-miss services."
               " This is the case that once produced a confident FOUR-step answer for a six-step procedure. 'Потврда за прием' is step 3 and 'Потврда за упис' step 5 -- exactly the steps that went missing -- so these values turn the completeness failure into something the gate can see."),

    Spec(query="Ми треба потврда дека фирмата нема забрана за учество во јавни набавки",
         # Channel-agnostic on purpose. "барање" occurs only in the Хартиено
         # track; the corpus rebuild gave 2075 a third variation and the agent
         # now answers from Електронски, where a paper application does not
         # exist. This case tests SERVICE routing -- it must not quietly also
         # demand one particular channel's vocabulary.
         values=('барање|електронск',), forbid=('субвенции', 'концесија'),
         intent="service_routing", service=2075,
         types=("description", "process"),
         notes="Routing case: the user describes the need, never the service "
               "name, and 8 sibling services differ only in which sanction they "
               "certify."
               " Routing case. The expected value is procedural (the process row says 'Подгответе барање'), because every topical word is already in the question and would be echoed. The forbidden values belong to SIBLING services -- 2073 certifies a ban on субвенции, 2072/2079 on лиценца and концесија -- so naming them means the answer routed to the wrong one of eight near-identical services."),

    Spec(query="Може ли да извадам потврда без регистрационен агент?",
         values=('без посредство',),
         intent="faq", service=2185, chunk_ids=("srv_2185_shared_faq_1",),
         also_services=(2186,),
         notes="This FAQ is duplicated verbatim under 14 services; all copies "
               "count. 2186's remaining blocks are graded acceptable context."
               " The FAQ answers 'Да, потврда може да се добие без посредство на регистрационен агент'. A hedged or negative answer cannot contain the phrase."),

    Spec(query="Колку чини упис на промена кај приватна установа?",
         values=('1603', '200'),
         intent="tariffs", service=2140, variation=11173, types=("tariffs",),
         notes="1603 МКД + 200 МКД per additional change in the same filing; "
               "both rows are needed for a complete answer."),

    Spec(query="Кои документи се потребни за упис на промена кај АД преку регистрационен агент?",
         intent="documents", service=2141, variation=11206, types=("documents",),
         notes="2140 (self-filing) vs 2141 (through an agent) are near-identical "
               "services with the same АД variation label -- service-level "
               "confusion, not variation-level."),

    # ----------------------------------------------------------------- #
    # Underspecified questions.
    #
    # Added after watching the live agent fail on them. Every case above names
    # its service; real users do not, and the gap is where the system broke:
    # "Колку чини регистрација?" retrieved zero tariff rows, because the word
    # "регистрација" matches blocks about registering a USER ACCOUNT in the
    # e-system, while the actual fee rows say "упис на основање" -- the word
    # appears in 2 of 315 tariff rows corpus-wide.
    #
    # Retrieval gold is every legal form's rows across BOTH registration
    # services: with the question as asked, any of them is a legitimate
    # retrieval, and the agent's job is to ask which one rather than to pick.
    # These are the cases that will show whether the vocabulary work lands.
    # ----------------------------------------------------------------- #
    Spec(query="Колку чини регистрација?",
         intent="underspecified", service=2135, types=("tariffs",),
         plus=((2136, None, ("tariffs",)),), behavior="clarify",
         notes="Ambiguous on two axes: which service (2135 self-filing vs 2136 "
               "via a registration agent) and which legal form. The fees really "
               "do diverge -- АД and ПДОО are 0 МКД, the other seven forms are "
               "2452 МКД -- so a guess is wrong by the entire amount."),

    Spec(query="Колку чини основање на фирма?",
         intent="underspecified", service=2135, types=("tariffs",),
         plus=((2136, None, ("tariffs",)),), behavior="clarify",
         notes="Same question in the vocabulary a user actually reaches for "
               "('фирма', 'основање'). Pairs with the 'регистрација' phrasing "
               "to separate the vocabulary gap from the ambiguity handling."),

    Spec(query="Кои документи ми требаат за да регистрирам фирма?",
         intent="underspecified", service=2135, types=("documents",),
         plus=((2136, None, ("documents",)),), behavior="clarify",
         notes="Document lists differ across all 9 forms (9 distinct sets), so "
               "this must be clarified, not answered."),

    Spec(query="Колку време трае регистрација на фирма?",
         intent="underspecified", service=2135, types=("deadlines",),
         plus=((2136, None, ("deadlines",)),), behavior="clarify",
         notes="Deadlines differ across forms (5 distinct sets across 9)."),

    Spec(query="Како да регистрирам залог?",
         intent="underspecified", service=2063, variation=11187,
         types=("process",), behavior="answer",
         notes="Single-variation service: NOT ambiguous, so the agent must "
               "answer rather than ask. Isolates the vocabulary gap from the "
               "disambiguation logic -- the sibling case adds the noun 'машина' "
               "(the corpus says 'подвижни предмети') and fails on all three "
               "retrievers."),
    # ----------------------------------------------------------------- #
    # Authorised-agent directory (scope == "directory").
    #
    # A different shape of question: not "what does this service require" but
    # "who near me can file it". The answer is a list, so completeness is the
    # whole game -- returning ten of Прилеп's 75 agents is a wrong answer that
    # reads like a right one. Gold is every block for that municipality, and
    # structured.fetch_directory serves them deterministically.
    # ----------------------------------------------------------------- #
    Spec(query="Кои се овластените регистрациони агенти во Гостивар?",
         values=('36', '33'),
         intent="agents", directory=(("ГОСТИВАР", None),), behavior="answer",
         notes="Both lists, one part each (33 advocates + 36 ДОО/ТП). They stay "
               "separate on purpose: the two carry different scopes of "
               "authority and merging them would blur a real legal distinction."),

    Spec(query="Дај ми список на адвокати регистрациони агенти во Прилеп",
         values=('64',),
         intent="agents", directory=(("ПРИЛЕП", "advocates"),), behavior="answer",
         notes="64 advocates across 2 parts -- no answer can list them all, so "
               "what matters is that both parts are reachable and the total is "
               "stated rather than silently truncated."),

    Spec(query="Кој може да ми регистрира ДОО во Битола?",
         intent="agents", directory=(("БИТОЛА", "doo_tp"),), behavior="answer",
         notes="Deliberately phrased WITHOUT 'агент' or 'адвокат', so the "
               "deterministic lookup does not fire and this rests on semantic "
               "retrieval alone. Tracks a known gap instead of hiding it."),
    # ----------------------------------------------------------------- #
    # Out of scope -- the agent must decline.
    #
    # Abstention could not be scored before these existed: a gold set made only
    # of answerable questions rewards a system that always answers. Each topic
    # was checked to have ZERO matching blocks in the corpus, so "I don't have
    # that" is the only correct response. They carry no retrieval gold by
    # design, and the retrieval harness skips them.
    # ----------------------------------------------------------------- #
    Spec(query="Колку изнесува данокот на добивка за ДОО?",
         intent="abstain", behavior="abstain",
         notes="Corporate tax is the revenue office's domain, not the registry's. "
               "0 corpus blocks mention данок на добивка."),

    Spec(query="Како да извадам возачка дозвола?",
         intent="abstain", behavior="abstain",
         notes="Interior ministry, not the registry. 0 corpus blocks."),

    Spec(query="Каде се вади патна исправа?",
         intent="abstain", behavior="abstain",
         notes="Interior ministry. 0 corpus blocks. Note the corpus DOES mention "
               "пасош 43 times as an identity document, so a retriever will "
               "return near-misses -- which is the point: a near-miss is not an "
               "answer."),
    # ----------------------------------------------------------------- #
    # Conversations.
    #
    # Every case above is one question, and the worst defect this project ever
    # produced was a CONVERSATION failure: asked for agents in Гостивар, then
    # about Струмица, then "give me a few examples", the agent invented a lawyer
    # by welding a Струмица name onto a Гостивар street address -- reading the
    # 36-entry list still sitting in its own message history, and citing
    # nothing. Single-turn cases cannot see that class of bug at all.
    # ----------------------------------------------------------------- #
    Spec(query="Колку чини регистрација?",
         followups=("за АД",),
         turn_behaviors=("clarify", "answer"),
         values=("0 мкд|0.0 мкд|нула",), forbid=("2452",),
         intent="multiturn_clarify", service=2135, variation=11113,
         types=("tariffs",), behavior="answer",
         notes="The clarification round trip end to end: turn 1 must ask which "
               "legal form, turn 2 must resolve 'за АД' to variation 11113 and "
               "answer from it. АД is 0 МКД while seven of the nine siblings "
               "are 2452 -- so quoting 2452 here means the filter never took "
               "effect, which is exactly the failure the whole feature exists "
               "to prevent."),

    Spec(query="Кои се овластените регистрациони агенти во Гостивар?",
         followups=("А во Струмица?", "Дај ми неколку како пример"),
         turn_behaviors=("answer", "answer", "answer"),
         values=("струмица",), forbid=("борче јованоски",),
         intent="multiturn_history", behavior="answer",
         directory=(("СТРУМИЦА", None),),
         notes="THE fabrication regression. Turn 1 returns 36 Гостивар agents; "
               "by turn 3 the model must not be building answers out of that "
               "list. 'Борче Јованоски' is a street that appears ONLY in "
               "Гостивар blocks, so its presence in an answer about Струмица is "
               "proof the history leaked -- which is precisely what happened "
               "live, producing a lawyer who does not exist."),

    Spec(query="Колку чини потврдата за тековна состојба?",
         followups=("а каде се подига?",),
         turn_behaviors=("answer", "answer"),
         values=("шалтер",), forbid=("2452",),
         intent="multiturn_followup", service=2162, variation=11050,
         types=("documentsLocations",), behavior="answer",
         notes="Context inheritance. 'а каде се подига?' is four tokens with no "
               "subject; without the follow-up planner it retrieves nothing "
               "useful and the thread is lost. The answer must still be about "
               "service 2162 -- the pickup counter -- not about whatever the "
               "bare question happens to match."),
    # ----------------------------------------------------------------- #
    # Completeness honesty.
    #
    # The <coverage> block tells the model which sections it holds in full. It
    # fires in production -- a live answer volunteered "Оваа информација не е
    # целосна" unprompted -- but nothing asserted it, so the guard could have
    # been removed by any refactor without a single test going red.
    #
    # АД has 27 required-document rows and the context holds at most 10, so
    # coverage always reports PARTIAL here. The question asks for ALL of them,
    # which leaves the model no honest way out except to say the list is not
    # complete.
    # ----------------------------------------------------------------- #
    Spec(query="Наброј ми ги сите документи потребни за упис на основање на АД",
         values=("27|не е целосн|не се сите|не ги опфаќа сите|само дел|"
                 "нецелосн|не располагам со сите|дополнителни извор|"
                 "консултира|дополнителни документи може|"
                 # The model paraphrases the same admission freely: it wrote
                 # "нема информација за комплетната листа" where an earlier run
                 # wrote "не е наведена целосната листа". Identical meaning, and
                 # scoring one 1.0 and the other 0.0 grades vocabulary, not
                 # honesty. Note these alternatives all carry their own negation
                 # -- a bare "комплетн" would also match a false CLAIM of
                 # completeness, which `forbid` below exists to catch.
                 "нема информација за комплетн|не е комплетн|нема комплетн",),
         forbid=("ова се сите документи", "ова е целосната листа"),
         behavior="answer",
         intent="completeness", service=2135, variation=11113,
         types=("documents",),
         notes="Any of the listed renderings counts, and `forbid` blocks the "
               "explicit false claim. This assertion is LEXICAL and therefore "
               "brittle: two runs disclaimed incompleteness in different words "
               "('не е целосната листа' / 'консултираат дополнителни "
               "извори') and the first list missed the second. Note that "
               "'целосн' cannot be matched on its own -- it is the root of "
               "both 'is complete' and 'is NOT complete'. The robust version of "
               "this check is the post-generation completeness comparison "
               "(count rows in context vs items enumerated in the answer), "
               "which is deferred."),

    # ----------------------------------------------------------------- #
    # Pledge deletion (2115 / v11125) -- three conversations from one live
    # session, written BEFORE the fixes so they are known to fail.
    #
    # The opener is "Ми треба бришење на залог" rather than the more natural
    # "Сакам да избришам залог", which retrieves ZERO blocks of service 2115:
    # stem("избришам") is "избришам" and stem("бришење") is "бришењ", so the
    # lexical arm cannot connect the verb to the noun. That is a real gap of the
    # same family as ликвидирам/ликвидација, but it is not what these three
    # cases are for -- an opener that fails to pin the service would make them
    # fail for the wrong reason.
    # ----------------------------------------------------------------- #
    Spec(query="Ми треба бришење на залог",
         followups=("како се прави тоа",),
         turn_behaviors=("answer", "answer"),
         values=("нотар", "потврда за прием", "потврда за упис",
                 "излезниот документ|се подига|подигнув"),
         intent="multiturn_process", service=2115, variation=11125,
         types=("process",), behavior="answer",
         notes="Live, this returned five of the registry's six steps and then "
               "said so: 'Оваа процедура не е целосна, бидејќи недостасува "
               "еден чекор.' The model was being honest -- coverage correctly "
               "reported PARTIAL because step 6 never reached the context. "
               "Injected rows are fused by RRF and then trimmed to k distinct, "
               "so the tail of a section can be pushed out by strong semantic "
               "hits. The fourth value targets step 6 (document pickup), the "
               "one that went missing."),

    Spec(query="Ми треба бришење на залог",
         followups=("колку се плаќа и како се плаќа, преку кои",),
         turn_behaviors=("answer", "answer"),
         values=("125", "100", "нотар"),
         forbid=("картичка", "консолидирана"),
         intent="multiturn_payment", service=2115, variation=11125,
         types=("tariffs",), behavior="answer",
         notes="Dual intent -- how MUCH and how PAID -- and live it answered "
               "neither, citing service 2117 (consolidated annual account). "
               "The cause is not an access/tariffs fight: NEITHER fired. "
               "'плаќа' matches no cue, because tariffs carries 'плати' and "
               "access carries 'плаќањ' and the Macedonian present tense falls "
               "between them, so intent resolved to 'process' on the strength "
               "of 'како' alone. 125/100 МКД are the paper and electronic "
               "tariffs; payment for THIS service is only 'преку нотар', so "
               "'картичка' can only have come from another service."),

    Spec(query="Ми треба бришење на залог",
         followups=("колку се плаќа и како се плаќа, преку кои",
                    "Што документи би ми требале за да го сторам тоа"),
         turn_behaviors=("answer", "answer", "answer"),
         values=("заложниот доверител", "полномошно"),
         forbid=("историјат на промени",),
         intent="multiturn_drift", service=2115, variation=11125,
         types=("documents",), behavior="answer",
         notes="Topic drift compounding. Turn 2 carries no subject of its own, "
               "but `_topic` updates to the last message unconditionally -- so "
               "a subject-free turn became the topic and turn 3 merged with "
               "nothing, landing on service 2159 (Потврда за историјат на "
               "промени). Its two document rows are 'Пополнет образец' and "
               "'Извршена уплата', which OVERLAP with 2115's, so the wrong "
               "answer looks plausible; the assertions therefore target the two "
               "rows unique to 2115 -- Полномошно and Изјава од заложниот "
               "доверител. This is the exact failure predicted when `_topic` "
               "was changed to follow the previous message."),

    # ----------------------------------------------------------------- #
    # A false premise the corpus explicitly contradicts.
    #
    # Every other case asks something answerable. This one asserts something
    # the registry forbids and asks the agent to help with it -- the shape a
    # user takes when they have already decided what they want to hear. The
    # corpus is unambiguous: for здружение and фондација, "Не е возможно
    # поднесување преку регистрационен агент"; they must file themselves.
    #
    # Live, the agent invented that the agent "can be a mediator" and then said
    # it had no specific documents -- accepting the premise instead of
    # correcting it.
    #
    # This was first diagnosed as the verb/noun retrieval gap (ликвидирам vs
    # ликвидација). That diagnosis was WRONG, and the correction is worth
    # recording: the rule reaches the context at RANK 1 every time. It arrives as
    # srv_2180_shared_faq_2, a byte-identical copy of the gold block -- this FAQ
    # has 8 copies corpus-wide and copy-tolerant grading already accepts them.
    # Retrieval does its job; the model reads a list of legal-entity types, does
    # not find the one row that excludes здружение, and cites a neighbouring FAQ
    # about what agents do in general. It is a generation failure, and no amount
    # of retrieval tuning will move it.
    #
    # Note the low ndcg@10 on this family (~0.07) is a metric artifact, not a
    # miss: 12 identical copies are all graded 2, and at most a couple can fit
    # in the top 10.
    # ----------------------------------------------------------------- #
    Spec(query="Сакам да ликвидирам Здружение преку регистрационен агент. "
               "Дали агентот може да ми ја среди целата документација?",
         # "не постои можност" is the corpus's OWN wording for the neighbouring
         # rows ("Не постои можност за поднесување во хартиена форма"), so the
         # model reaching for it is a sign it read the list, not a miss.
         values=("не е возможно|не е можно|не е дозволено|не може|"
                 "не постои можност|нема можност|не е овластен", "самостојно"),
         # Two calibrations, both from real answers:
         #   1. The first version forbade only "агентот може", and passed
         #      "регистрационен агент може да ви помогне" -- the identical false
         #      claim -- at 0.75. Hence the alternatives.
         #   2. "агент може" was then too broad. A correct answer opens with the
         #      TRUE general statement ("регистрационен агент може да поднесе
         #      пријава за ликвидација на правно лице") before giving the
         #      здружение exception. What is false is the claim made TO THE USER
         #      about THEIR filing, so only those phrasings are forbidden.
         # Negated occurrences do not count -- see _states_unnegated().
         forbid=("може да ви помогне|може да ви ја среди|може да ја среди|"
                 "може да биде посредник|да, регистрационен агент",),
         behavior="answer",
         intent="false_premise", service=2126,
         chunk_ids=("srv_2126_shared_faq_1", "srv_2126_shared_faq_2"),
         notes="The gold FAQ lists submission methods per legal-entity type and "
               "states the restriction verbatim. faq_1 has 4 identical copies "
               "across the corpus, so copy-tolerant grading applies. `forbid` "
               "targets the affirmative claim only; the scorer ignores negated "
               "occurrences, so 'агентот не може да ви ја среди' -- which is the "
               "correct answer -- passes."),
    # ----------------------------------------------------------------- #
    # Spurious clarification.
    #
    # Every clarification case above asserts that the agent DOES ask. Nothing
    # asserted that it stays quiet, so a detector that asks too often scored a
    # clean 1.000 -- and in live testing it asked on almost every pointed
    # question, offering nine legal forms to answer something identical in all
    # nine. These three cases close that hole from both sides: two that must not
    # ask, one that must.
    #
    # The root cause is a section-level signal driving a row-level decision.
    # detect_intent() resolves a question to a section ("documents"), and
    # _differing_sections() then asks whether that WHOLE section differs between
    # variants -- when the user asked about one row inside it.
    # ----------------------------------------------------------------- #
    Spec(query="Имам доказ за регистрација кој е на англиски јазик. "
               "Што точно треба да направам со него?",
         intent="universal_row", service=2135, types=("documents",),
         behavior="answer",
         values=("преведен", "заверен"),
         # This question is FULLY covered -- the corpus states what to do with a
         # foreign-language document -- so any "I don't have it" is false. The
         # agent kept opening with "нема информација за тоа што точно треба да
         # направите со доказот" and then answering correctly in the next
         # sentence. A disclaimer contradicted by the rest of the answer is not
         # caution, it is noise that makes a good answer look unreliable.
         forbid=("нема информација|немам информација|не располагам со информација",),
         notes="The answer is 'Преведени и заверени документи' -- documents_1, "
               "byte-identical across ALL NINE variations of 2135. Live, the "
               "agent answered it correctly and then asked the user to choose "
               "between the nine, to select a row that is the same in every "
               "one. Nothing about the legal form changes this answer, so the "
               "question must not be asked."),
    # The counterweight. Written at the same time and deliberately kept next to
    # the case above, because the cheap fix for that one ("stop asking about
    # documents") breaks this one.
    Spec(query="Каде можам да го подигнам документот",
         intent="real_variation", service=2140, types=("documentsLocations",),
         behavior="clarify",
         # Asserts the mutual-exclusion rule: asking is INSTEAD OF answering.
         # observed_behavior() reports "clarify" whether or not the model also
         # answered, so without this the rule has no test. Live, the agent gave
         # the pickup methods and THEN asked -- which is the worst of both, since
         # the methods it gave are false for two of the eight forms. None of the
         # eight labels is a channel, so forbidding channel words cannot collide
         # with the list of options the question itself offers.
         forbid=("на шалтерите|шалтерите на црремсм|електронски или",),
         notes="Looks identical to the case above -- pointed question, agent "
               "answered then asked -- but here the question is real and the "
               "ANSWER was the error. Заедница на сопственици and Приватна "
               "установа have NO 'Начин на подигнување: Електронски' row at "
               "all, so 'електронски или на шалтер' is false for two of the "
               "eight forms. Asking is correct; answering as well is not. This "
               "case exists to stop the universality gate from over-tightening: "
               "if a change makes the case above pass by silencing this one, "
               "the change is wrong."),
    Spec(query="Дали ми треба некаква потврда од банка пред да го избришам "
               "мојот ТП и што треба да пишува во неа",
         intent="absent_form", behavior="answer",
         service=2126, types=("documents",),
         values=("банка", "затворена"),
         notes="REWRITTEN after the corpus rebuild. This case previously asserted "
               "an abstention, on the evidence that no bank-closure document "
               "existed for any legal form. It does exist -- "
               "srv_2126_v10919_documents_5, 'Доказ од надлежна Банка дека "
               "сметката е затворена' -- under the ТП variation, which the "
               "scraper had silently skipped along with 101 others. The user "
               "reported it from the live portal and was right; the corpus was "
               "wrong, and this test had frozen the gap into an expectation. "
               "Kept under the same name as a reminder that a passing test can "
               "encode missing data as a rule. Superseded notes follow. (1) The "
               "document does not exist: the "
               "corpus has no bank-account-CLOSURE evidence for any form -- only "
               "'отворена нерезидентна сметка' (Претставништво) and 'состојба "
               "(солвентност)' (Подружница ... странски поединец). Abstaining is "
               "correct. (2) The agent instead asked the user to choose a legal "
               "form from a list that could not contain theirs: ТП is in the "
               "glossary (ТП -> Трговец – поединец) but is NOT one of the 82 "
               "variation labels in the corpus. Naming a form the attributed "
               "service does not have means the ATTRIBUTION is wrong; asking the "
               "user to pick is the one response that cannot help. Note this "
               "second half needs the named-but-absent-form gate, not the "
               "universality gate."),
    # ----------------------------------------------------------------- #
    # A question in two parts, one of them uncovered.
    #
    # The same English-document question as above, with a clause added naming a
    # procedure the corpus does not describe. The agent abstained on ALL of it:
    # "нема информација за постапката за бришење ... конкретно за бришење не
    # можам да помогнам" -- and never mentioned the translation, which it holds.
    #
    # Diagnosed layer by layer, because it looks exactly like retrieval drift and
    # is not. Intent is identical to the short query (['documents']); "бришење"
    # is too common to anchor (DF 17.8%, over the 10% cut) so it cannot pull
    # attribution; the translation row's dense similarity RISES with the clause
    # (0.5248 -> 0.5414); and the row reaches the context at fused rank 8. The
    # model had the answer and withheld it, because one part of the question was
    # uncovered and it treated that as grounds to refuse the whole thing.
    # ----------------------------------------------------------------- #
    Spec(query="Имам доказ за регистрација кој е на англиски јазик. Што точно "
               "треба да направам со него пред да го поднесам за бришење",
         intent="partial_answer", service=2135, types=("documents",),
         behavior="answer",
         values=("преведен", "заверен"),
         notes="Asserts the covered half is answered. The uncovered half may "
               "also be named -- that is correct and not scored, because the "
               "failure was never saying too little about бришење, it was "
               "saying nothing about the document."),
    # ----------------------------------------------------------------- #
    # Live tools: the choice boundary.
    #
    # These two are a pair and only mean anything together. One question can
    # ONLY be answered by a live lookup, the other must NEVER trigger one -- and
    # the second matters more, because a model with a hammer reaches for it, and
    # that tendency gets worse as Forms 2-3 are added.
    #
    # Both run against recorded fixtures (agent/tools/fixtures.py); the eval
    # only touches the real portal with --live-tools.
    # ----------------------------------------------------------------- #
    Spec(query="Која е големината на субјектот со ЕМБС 07696876?",
         intent="live_lookup", behavior="answer", live_tool=True,
         values=("мал",),
         # FORM 1 vs FORM 2. Both tools can answer this -- the profile image
         # prints Големина too -- so the cheap one has to win on its own merits.
         # The assertion is the citation: get_entity_profile costs a vision call
         # and ~3s to return one word that check_entity_size returns for free,
         # and nothing but this line notices when the model starts preferring it.
         forbid=("tool:entity_profile",),
         notes="Answerable ONLY by calling check_entity_size: no corpus block "
               "holds this company's size. Asserting the value also proves the "
               "tool ran -- 'мал' cannot be grounded any other way -- and the "
               "citation it carries must be tool:entity_size:07696876, which "
               "exercises the whole synthetic-ContextDoc path: the id has to "
               "reach the model, survive validate_citations, and keep the hard "
               "abstention guard from replacing a correct answer."),
    # FORM 2. The profile arrives as a PNG read by gpt-4o Vision, so unlike
    # every other case here the value under test passed through a probabilistic
    # step. ЕДБ and the founding date are asserted because they appear NOWHERE
    # in the corpus -- stating them proves the image was read, and misreading a
    # digit fails the case rather than quietly shipping a plausible number.
    Spec(query="Дај ми ги основните податоци за субјектот со ЕМБС 07696876.",
         intent="live_lookup", behavior="answer", live_tool=True,
         values=("4058023546097", "19.09.2023"),
         notes="Exercises the whole Form 2 path end to end: tool choice, the "
               "vision extraction, the ЕМБС self-check, and the citation. The "
               "citation half is load-bearing -- the answer is a ten-row list "
               "and the model omitted the bracket on it often enough that the "
               "hard-abstention guard replaced a perfectly correct profile with "
               "'нема информација'. A case that only checked the VALUES would "
               "have scored that failure as a pass."),
    Spec(query="Дај ми го основниот профил за субјектот со ЕМБС 7405855.",
         intent="live_lookup", live_tool=True,
         # Cross-contamination is the failure that matters here: the ONLY
         # profile this agent holds is ЛОРА's, so an answer about a different
         # ЕМБС that contains ЛОРА's ЕДБ or name has served one company's data
         # under another company's number -- which looks entirely plausible and
         # is the single worst thing this tool can do.
         forbid=("4058023546097", "ЛОРА"),
         notes="No saved image for this entity, so Form 2 must report it as "
               "unavailable rather than improvise. Behaviour is NOT graded: "
               "'профилот не е достапен' sits between answer and abstain and "
               "the distinction is not what this case is about. What it is "
               "about is that the reply carries no other entity's fields, and "
               "that a miss stays a returned value -- an exception here would "
               "take down the turn instead of answering it."),
    # FORM 3. A published Решение, addressed by its деловоден број. The asserted
    # values -- the size class and the activity code -- are what THIS decision
    # decided, and together they occur nowhere in the corpus, so stating them
    # proves the image was read rather than the documentation paraphrased.
    Spec(query="Што е одлучено со решението со деловоден број 30120260014967?",
         intent="live_lookup", behavior="answer", live_tool=True,
         values=("микро", "01.250"),
         notes="The Form 3 path end to end: a nested vision schema, the "
               "деловоден-број self-check, the field-vs-row consistency check "
               "and the citation. Tables only -- the preamble was dropped after "
               "it produced three uncatchable misreads, one of them a real "
               "official's surname."),
    Spec(query="Дај ми го решението за упис за субјектот со ЕМБС 7405855.",
         intent="live_lookup", live_tool=True,
         # The identifier boundary. Only an ЕМБС was given, and Form 3 is keyed
         # by деловоден број, so the right move is to ask for the number. The
         # failure this catches is the model FABRICATING a 14-digit number to
         # get a call through: every call carries the citation directive, so an
         # invented number shows up as a tool:registration_decision citation
         # whether it happened to hit or not.
         forbid=("tool:registration_decision",),
         notes="Behaviour is NOT graded: asking for the деловоден број, "
               "offering the entity's profile instead, or both are all fine. "
               "Calling the decision tool with a number the user never gave is "
               "not -- a hit would be some other company's decision, and a miss "
               "is a wasted call presented as a search."),
    Spec(query="Што значи големина на субјект?",
         intent="no_tool", live_tool=True,
         # The assertion is the CITATION, not the wording. Forbidding the size
         # words themselves does not work: "голем" is a substring of "големина",
         # the question's own noun, so the case failed itself. A tool:entity_size
         # citation can only exist if the lookup ran, which is exactly and only
         # what this case is about.
         forbid=("tool:entity_size",),
         notes="The tool-choice boundary, and the half that degrades quietly as "
               "Forms 2-3 are added: a model with more hammers reaches for them. "
               "This question is about a CONCEPT and names no entity, so a live "
               "lookup cannot answer it -- there is nothing to look up. Behaviour "
               "is deliberately NOT graded: the corpus mentions 'големина на "
               "субјект' only as a field listed on a certificate (2191) and never "
               "defines the classes, so answer-vs-abstain is a judgement call "
               "this case has no business freezing. Calling the tool is not."),
]


def case_id(spec: "Spec") -> str:
    """Stable id, derived from the QUERY rather than the list position.

    Index-based ids (`curated::07::deadlines`) renumber every later case the
    moment a spec is inserted in the middle, which silently orphans saved
    answers and reports -- the scores then move for a reason that has nothing to
    do with the system. Hashing the query means an id changes only when the
    question itself changes, which is exactly when it should.
    """
    digest = hashlib.sha1(spec.query.encode("utf-8")).hexdigest()[:8]
    return f"curated::{digest}::{spec.intent}"


def expand(c: Corpus) -> list[EvalCase]:
    """Resolve every Spec against the corpus into a validated EvalCase."""
    cases: list[EvalCase] = []
    for i, s in enumerate(SPECS, 1):
        primary: list[str] = []
        if s.chunk_ids:
            for cid in s.chunk_ids:
                if cid not in c.by_id:
                    raise ValueError(f"curated case {i} ({s.query[:40]!r}): "
                                     f"gold chunk_id {cid!r} is not in the corpus")
                primary.append(cid)
        targets: list[tuple[int, int | None, tuple[str, ...]]] = []
        if s.types:
            if s.service is None:
                raise ValueError(f"curated case {i}: `types` needs a `service`")
            targets.append((s.service, s.variation, s.types))
        targets += list(s.plus)
        for sid, vid, types in targets:
            for type_ in types:
                found = c.of_type(sid, type_, id_variation=vid)
                if not found:
                    raise ValueError(
                        f"curated case {i} ({s.query[:40]!r}): no {type_!r} blocks "
                        f"for service {sid} variation {vid}")
                primary += [b.chunk_id for b in found]
        for municipality, list_key in s.directory:
            blocks = [b for b in c.by_municipality.get(municipality, ())
                      if list_key is None or b.list_key == list_key]
            if not blocks:
                raise ValueError(
                    f"curated case {i} ({s.query[:40]!r}): no agent-directory "
                    f"blocks for {municipality!r} list={list_key!r}")
            primary += [b.chunk_id for b in blocks]

        if (not primary and s.behavior != "abstain" and not s.followups
                and not s.live_tool):
            raise ValueError(f"curated case {i}: no gold resolved")

        if not s.pin_provenance:
            for cid in list(primary):
                if c.by_id[cid].type in _COPY_TOLERANT_TYPES:
                    primary += c.identical_to(cid)

        gold = {cid: 2 for cid in primary}
        if s.service is not None:
            gold.update(_support_grades(c, s.service, s.variation, gold))
        for sid in s.also_services:
            for b in c.by_service.get(sid, ()):
                gold.setdefault(b.chunk_id, 1)

        cases.append(EvalCase(
            case_id=case_id(s),
            query=s.query,
            family=f"curated_{s.intent}",
            source="curated",
            gold=gold,
            live_tool=s.live_tool,
            expect_service=s.service,
            expect_variation=s.variation,
            expect_types=list(s.types) or _types_of(c, primary),
            difficulty=s.difficulty,
            notes=s.notes,
            expect_behavior=s.behavior,
            expect_values=list(s.values),
            forbid_values=list(s.forbid),
            turns=([s.query, *s.followups] if s.followups else []),
            expect_turn_behaviors=list(s.turn_behaviors),
        ))
    return cases


def _types_of(c: Corpus, chunk_ids: list[str]) -> list[str]:
    return sorted({c.by_id[cid].type for cid in chunk_ids})
