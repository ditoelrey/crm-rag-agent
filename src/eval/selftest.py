"""
selftest.py  --  known-answer tests for the harness itself.
===========================================================
An eval harness is the instrument every later decision is read off, so it needs
its own calibration check. Three layers:

  1. metrics    hand-computed nDCG/MRR/coverage values, so a refactor of the
                formulas cannot silently shift every score.
  2. gold set   the validator must reject stale chunk_ids -- the failure mode
                that looks identical to a broken retriever.
  3. end-to-end oracle must score a perfect 1.0 and random must score near 0 on
                the real corpus. If the oracle is not 1.0, the harness is wrong,
                not the retriever.

Run: python -m eval.cli selftest        (from src/)
"""
from __future__ import annotations

import math
import os
import traceback

from . import curated, goldset, metrics as M
from .baselines import BM25Retriever, OracleRetriever, RandomRetriever
from .corpus import DEFAULT_CORPUS, load as load_corpus
from .goldset import EvalCase
from .harness import evaluate
from .text import stem, tokenize

_failures: list[str] = []


def check(label: str, got, want, tol: float = 1e-9) -> None:
    ok = (abs(got - want) <= tol) if isinstance(want, float) else (got == want)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got!r}, want {want!r}")
    if not ok:
        _failures.append(label)


def check_true(label: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{(' -- ' + detail) if detail else ''}")
    if not cond:
        _failures.append(label)


def _metrics_tests() -> None:
    print("[metrics]")
    ranked = ["c1", "c2", "c3", "c4"]

    check("hit@1 when rank1 is gold", M.hit_at_k(ranked, {"c1": 2}, 1), 1.0)
    check("hit@1 when rank3 is gold", M.hit_at_k(ranked, {"c3": 2}, 1), 0.0)
    check("hit@3 when rank3 is gold", M.hit_at_k(ranked, {"c3": 2}, 3), 1.0)
    check("grade-1 alone never counts as a hit", M.hit_at_k(ranked, {"c1": 1}, 4), 0.0)

    check("rr for rank 3", M.rr_at_k(ranked, {"c3": 2}, 10), 1 / 3)
    check("rr outside cutoff", M.rr_at_k(ranked, {"c4": 2}, 3), 0.0)

    gold4 = {"a": 2, "b": 2, "c": 2, "d": 2}
    mixed = ["a", "x", "y", "z", "w"]
    check("recall@2 with 4 gold", M.recall_at_k(mixed, gold4, 2), 0.25)
    check("coverage@2 normalizes by reachable", M.coverage_at_k(mixed, gold4, 2), 0.5)
    check("coverage@k>|gold| equals recall", M.coverage_at_k(mixed, gold4, 10), 0.25)
    check("precision@2", M.precision_at_k(mixed, gold4, 2), 0.5)

    # DCG = 0 + 1/log2(3) + 3/log2(4);  IDCG = 3/log2(2) + 1/log2(3)
    dcg = 1 / math.log2(3) + 3 / math.log2(4)
    idcg = 3 / math.log2(2) + 1 / math.log2(3)
    check("ndcg@3 graded", M.ndcg_at_k(["c1", "c2", "c3"], {"c3": 2, "c2": 1}, 3),
          dcg / idcg, tol=1e-9)
    check("ndcg is 1.0 for a perfect ranking",
          M.ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, 2), 1.0)
    check("ndcg is 0.0 when nothing relevant is returned",
          M.ndcg_at_k(["x", "y"], {"a": 2}, 2), 0.0)
    check("empty gold scores 0", M.ndcg_at_k(ranked, {}, 5), 0.0)


def _text_tests() -> None:
    print("[text]")
    check("definite article stripped", stem("документите"), "документ")
    check("short root protected", stem("дом"), "дом")
    check("unstemmed tokenization keeps surface forms",
          tokenize("Колку чини АД регистрација?", do_stem=False),
          ["колку", "чини", "ад", "регистрација"])
    check_true("stopwords dropped", "на" not in tokenize("документи на субјект"))
    check_true("intent word survives stopword filtering",
               "колку" in tokenize("колку чини", do_stem=False))
    check_true("inflected forms merge to one term",
               stem("документи") == stem("документот") == stem("документите"),
               f"{stem('документи')} / {stem('документот')} / {stem('документите')}")


def _goldset_tests(c) -> None:
    print("[goldset]")
    bogus = EvalCase(case_id="x", query="q", family="f", source="curated",
                     gold={"srv_does_not_exist_v0_process_1": 2})
    try:
        goldset.validate_cases([bogus], c)
        check_true("validator rejects a stale gold chunk_id", False)
    except ValueError as e:
        check_true("validator rejects a stale gold chunk_id", "not in corpus" in str(e))

    real_id = c.blocks[0].chunk_id
    no_primary = EvalCase(case_id="y", query="q", family="f", source="curated",
                          gold={real_id: 1})
    try:
        goldset.validate_cases([no_primary], c)
        check_true("validator rejects gold with no primary", False)
    except ValueError as e:
        check_true("validator rejects gold with no primary", "primary" in str(e))

    cases_a = goldset.generate(c, seed=7)
    cases_b = goldset.generate(c, seed=7)
    check_true("generation is deterministic for a fixed seed",
               [x.case_id for x in cases_a] == [x.case_id for x in cases_b] and
               [x.query for x in cases_a] == [x.query for x in cases_b])
    goldset.validate_cases(cases_a, c)
    check_true("generated gold validates", True, f"{len(cases_a)} cases")

    cur = curated.expand(c)
    goldset.validate_cases(cur, c)
    check("every curated spec resolves", len(cur), len(curated.SPECS))
    copy_tolerant = [x for x in cur if x.family.endswith(("faq", "terminology"))]
    check_true("boilerplate gold accepts its duplicate copies",
               all(sum(1 for g in x.gold.values() if g >= 2) > 1 for x in copy_tolerant),
               f"{len(copy_tolerant)} copy-tolerant curated cases")

    var_cases = [x for x in cases_a if x.expect_variation is not None]
    check_true("variation family is populated", len(var_cases) > 20,
               f"{len(var_cases)} variation-targeted cases")
    leaked = [x for x in var_cases
              if any(c.by_id[cid].id_variation not in (None, x.expect_variation)
                     for cid, g in x.gold.items() if g >= 2)]
    check_true("no sibling variation is graded primary", not leaked,
               f"{len(leaked)} leaky cases")


def _end_to_end_tests(c) -> None:
    print("[end-to-end]")
    cases = goldset.generate(c, seed=7)[:120]

    oracle = evaluate(OracleRetriever(cases), cases, c)
    check("oracle hit@1", oracle.overall["hit@1"], 1.0)
    check("oracle mrr@10", oracle.overall["mrr@10"], 1.0)
    check("oracle ndcg@20", oracle.overall["ndcg@20"], 1.0, tol=1e-6)

    noisy = evaluate(OracleRetriever(cases, noise=3, corpus=c), cases, c)
    check_true("noise degrades the oracle", noisy.overall["ndcg@10"] < oracle.overall["ndcg@10"],
               f"{noisy.overall['ndcg@10']:.3f} < {oracle.overall['ndcg@10']:.3f}")

    rnd = evaluate(RandomRetriever(c), cases, c)
    check_true("random is near the floor", rnd.overall["ndcg@10"] < 0.02,
               f"ndcg@10={rnd.overall['ndcg@10']:.4f}")

    bm25 = evaluate(BM25Retriever(c), cases, c)
    check_true("bm25 clears random by a wide margin",
               bm25.overall["ndcg@10"] > rnd.overall["ndcg@10"] + 0.2,
               f"bm25={bm25.overall['ndcg@10']:.3f} random={rnd.overall['ndcg@10']:.3f}")
    check_true("report carries the corpus hash", len(bm25.corpus_sha256) == 64)


def _answer_tests(c) -> None:
    """Every scorer is checked against a defect this project actually produced.

    A scorer that cannot catch the failures we already know about is decoration.
    """
    print("[answer eval]")
    from . import answers as A
    from . import curated

    cases = {x.query: x for x in curated.expand(c)}

    def rec(text, cites=(), docs=(), clarify=False, invalid=()):
        return A.AnswerRecord(case_id="t", query="q", text=text,
                              citations=list(cites), docs=list(docs or cites),
                              asked_clarification=clarify,
                              invalid_citations=list(invalid))

    fee_case = cases["Колку чини потврда за тековна состојба на фирма?"]
    tariff = "srv_2162_v11050_tariffs_1"

    good = rec("Тарифата изнесува 295 МКД [%s]." % tariff, [tariff])
    check("correct fee: value found", A.value_recall(good, fee_case), 1.0)
    check("correct fee: near-miss absent", A.forbidden_absent(good, fee_case), 1.0)
    check("correct fee: numbers grounded", A.numeric_groundedness(good, c), 1.0)
    check("correct fee: the citation supports it",
          A.value_citation(good, fee_case, c), 1.0)

    # 299 МКД is a REAL tariff -- for a different certificate. The most
    # expensive class of error this system can make.
    wrong = rec("Тарифата изнесува 299 МКД [srv_2185_v10943_tariffs_1].",
                ["srv_2185_v10943_tariffs_1"])
    check("near-miss fee is caught", A.forbidden_absent(wrong, fee_case), 0.0)
    check("...and the expected value is missing", A.value_recall(wrong, fee_case), 0.0)

    invented = rec("Тарифата изнесува 999 МКД [%s]." % tariff, [tariff])
    check("a fabricated amount is ungrounded",
          A.numeric_groundedness(invented, c), 0.0)

    # Observed live: correct details, cited to part 1 of a list; the agent named
    # is in part 2. Retrieval-level checking cannot see this.
    both = ["agents_doo_tp_centar_p1", "agents_doo_tp_centar_p2"]
    misattributed = rec("Телефонот е 077-860-715 [agents_doo_tp_centar_p1].",
                        ["agents_doo_tp_centar_p1"], both)
    phone_case = A_case(cases, "Кои се овластените регистрациони агенти во Гостивар?",
                        ["077-860-715"])
    check("citation to the wrong part is caught",
          A.value_citation(misattributed, phone_case, c), 0.0)
    check("...while the fact itself is grounded in the context",
          A.numeric_groundedness(misattributed, c), 1.0)
    right = rec("Телефонот е 077-860-715 [agents_doo_tp_centar_p2].",
                ["agents_doo_tp_centar_p2"], both)
    check("citation to the right part passes",
          A.value_citation(right, phone_case, c), 1.0)

    # Behaviour classification.
    check("abstention is recognised",
          A.observed_behavior(rec("Во документацијата со која располагам нема "
                                  "информација за тоа.")), "abstain")
    check("clarification is recognised",
          A.observed_behavior(rec("За каква правна форма станува збор?")), "clarify")
    check("the ambiguity flag alone marks a clarification",
          A.observed_behavior(rec("Која е вашата намера?", clarify=True)), "clarify")
    check("a normal answer is an answer",
          A.observed_behavior(rec("Тарифата изнесува 295 МКД [x].")), "answer")

    # An answer that states a fee and notes one gap is an ANSWER, not a refusal
    # -- grading it as one would reward hedging.
    hedged = rec("Тарифата изнесува 295 МКД [%s]. " % tariff
                 + "Дополнителни детали за роковите не се достапни во "
                 + "документацијата со која располагам, па препорачувам да "
                 + "проверите директно кај Централниот регистар за останатите "
                 + "чекори од постапката и потребните документи." )
    check("a hedged answer is still an answer", A.observed_behavior(hedged), "answer")

    # Enumeration markers must not be read as factual numbers.
    check("list markers are not claims",
          A.numbers_in("1. Прв чекор\n2. Втор чекор\n3. Трет чекор"), [])
    check("real numbers are claims",
          A.numbers_in("Тарифата е 2452 МКД, рокот е 15 дена."), ["2452", "15"])

    check("a fabricated citation fails integrity",
          A.score_case(rec("Текст [srv_9999_v0_x_1].", invalid=["srv_9999_v0_x_1"]),
                       fee_case, c)[0]["citation_integrity"], 0.0)

    # Behaviour gold: an abstain case answered is a behaviour miss.
    abstain_case = cases["Како да извадам возачка дозвола?"]
    scores, _ = A.score_case(rec("Возачка дозвола се вади во МВР."), abstain_case, c)
    check("answering an out-of-scope question fails behaviour",
          scores["behavior_match"], 0.0)
    scores, _ = A.score_case(
        rec("Во документацијата со која располагам нема информација за тоа."),
        abstain_case, c)
    check("declining an out-of-scope question passes behaviour",
          scores["behavior_match"], 1.0)

    # A correct paraphrase must not be graded as a miss. The corpus writes
    # "15.3."; gpt-4o wrote "15 март" and an earlier scorer gave it 0.0 on both
    # value_recall and numeric_groundedness -- rewarding regurgitation.
    dl = cases["Кои се роковите за поднесување годишна сметка за банка?"]
    verbatim = rec("Роковите се 15.3. и 31.12. [srv_2111_v11176_deadlines_1].",
                   ["srv_2111_v11176_deadlines_1"],
                   ["srv_2111_v11176_deadlines_1", "srv_2111_v11176_deadlines_2"])
    natural = rec("Роковите се до 15 март и до 31 декември "
                  "[srv_2111_v11176_deadlines_1].",
                  ["srv_2111_v11176_deadlines_1"],
                  ["srv_2111_v11176_deadlines_1", "srv_2111_v11176_deadlines_2"])
    check("verbatim dates score full marks", A.value_recall(verbatim, dl), 1.0)
    check("a natural rendering scores the same", A.value_recall(natural, dl), 1.0)
    check("...and is not read as ungrounded",
          A.numeric_groundedness(natural, c), 1.0)
    invented_date = rec("Рокот е до 27 март [srv_2111_v11176_deadlines_1].",
                        ["srv_2111_v11176_deadlines_1"],
                        ["srv_2111_v11176_deadlines_1"])
    check("but an invented date is still caught",
          A.numeric_groundedness(invented_date, c), 0.0)

    # Form 3's identifier boundary: given only an ЕМБС, the right reply asks for
    # the 14-digit деловоден број and cites nothing. Its numbers are a digit
    # count from the tool contract and the user's own ЕМБС -- neither is a
    # fabricated registry fact, and the scorer read both as one.
    def asked(query, text, docs=()):
        return A.AnswerRecord(case_id="t", query=query, text=text, citations=[],
                              docs=list(docs))
    q = "Дај ми го решението за упис за субјектот со ЕМБС 7405855."
    check("a digit count is not a claim",
          A.numbers_in("потребен ми е деловодниот број, кој е 14-цифрен број"), [])
    check("...nor '7 или 8 цифри'", A.numbers_in("ЕМБС има 7 или 8 цифри"), [])
    check("...but '14 дена' still is", A.numbers_in("рокот е 14 дена"), ["14"])
    check("the user's own ЕМБС repeated back is grounded", A.numeric_groundedness(
        asked(q, "ЕМБС (7405855) не е доволен; потребен ми е деловодниот број."),
        c), 1.0)
    check("...but a DIFFERENT identifier is not", A.numeric_groundedness(
        asked(q, "Решението за ЕМБС 7405856 е донесено."), c), 0.0)
    # The guard on the exemption: a short number from the question is the shape
    # of a false premise ("the fee is 999?" -> "yes, 999"), and must still fail.
    check("an amount echoed from a false premise is still ungrounded",
          A.numeric_groundedness(asked("Дали таксата е 999 денари?",
                                       "Да, таксата е 999 денари.", [tariff]), c),
          0.0)

    # --- conversations ---------------------------------------------------- #
    mt = cases["Кои се овластените регистрациони агенти во Гостивар?"]
    check("the fabrication case is multi-turn", len(mt.conversation), 3)
    check_true("its assertions target the FINAL answer",
               "струмица" in mt.expect_values and "борче јованоски" in mt.forbid_values)

    # The exact leak seen live: a Гостивар street in an answer about Струмица.
    leaked = A.AnswerRecord(
        case_id=mt.case_id, query=mt.query,
        text="Еве неколку агенти: АДВОКАТ ТОМЕ ЃОРЃЕВИЌ, Ул. БОРЧЕ ЈОВАНОСКИ "
             "Бр.56 ГОСТИВАР.",
        turns=[{"text": "...", "citations": ["agents_doo_tp_gostivar_p1"]},
               {"text": "...", "citations": ["agents_advocates_strumica_p1"]},
               {"text": "Еве неколку...", "citations": []}])
    check("the history leak is caught", A.forbidden_absent(leaked, mt), 0.0)

    clean = A.AnswerRecord(
        case_id=mt.case_id, query=mt.query,
        text="Во Струмица се достапни следниве агенти "
             "[agents_advocates_strumica_p1].",
        citations=["agents_advocates_strumica_p1"],
        turns=[{"text": "a", "citations": ["agents_doo_tp_gostivar_p1"]},
               {"text": "b", "citations": ["agents_advocates_strumica_p1"]},
               {"text": "c", "citations": ["agents_advocates_strumica_p1"]}])
    check("a clean conversation passes", A.forbidden_absent(clean, mt), 1.0)
    check("...and states the right municipality", A.value_recall(clean, mt), 1.0)

    # Per-turn behaviour: ending well does not excuse asking the wrong thing.
    cl = cases["Колку чини регистрација?"]
    good = A.AnswerRecord(
        case_id=cl.case_id, query=cl.query, text="Тарифата е 0 МКД [x].",
        citations=["srv_2135_v11113_tariffs_1"],
        turns=[{"text": "За каква правна форма станува збор?", "citations": [],
                "asked_clarification": True},
               {"text": "Тарифата е 0 МКД.", "citations": ["srv_2135_v11113_tariffs_1"]}])
    check("a correct round trip scores 1.0",
          A.score_case(good, cl, c)[0]["turn_behavior_match"], 1.0)
    guessed = A.AnswerRecord(
        case_id=cl.case_id, query=cl.query, text="Тарифата е 2452 МКД [x].",
        citations=["srv_2135_v11117_tariffs_1"],
        turns=[{"text": "Тарифата е 2452 МКД.", "citations": ["srv_2135_v11117_tariffs_1"]},
               {"text": "Тарифата е 2452 МКД.", "citations": ["srv_2135_v11117_tariffs_1"]}])
    check("answering instead of asking fails the round trip",
          A.score_case(guessed, cl, c)[0]["turn_behavior_match"], 0.5)
    check("...and quoting the wrong form's fee is caught",
          A.forbidden_absent(guessed, cl), 0.0)

    # Round trip through disk, so `--from` re-scoring is safe.
    import tempfile
    tmp = os.path.join(tempfile.mkdtemp(), "a.jsonl")
    A.write_answers([good], tmp)
    check("answers survive a save/load round trip",
          A.read_answers(tmp)[0].text, good.text)


def A_case(cases, query, values):
    """A copy of a curated case with different expected values, for fixtures."""
    import copy
    case = copy.deepcopy(cases[query])
    case.expect_values = list(values)
    return case


def main(corpus_path: str = DEFAULT_CORPUS) -> int:
    _failures.clear()
    c = load_corpus(corpus_path)
    print(f"corpus: {c.path}\n        {len(c)} blocks, sha={c.sha256[:12]}\n")
    for stage in (_metrics_tests, _text_tests):
        stage()
    for stage in (_goldset_tests, _answer_tests, _end_to_end_tests):
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
